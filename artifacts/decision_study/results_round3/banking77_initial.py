"""Frozen-encoder domain adaptation, lexical controls, and candidate-budget ablation.

Run with the pinned upstream checkout and checkpoint. Test inference starts only
after selection.json is persisted. All methods use the same deduplicated rows.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np

from metrics import paired_accuracy, probabilities, quality

REVISION = "7a77439b0923d6efe00052d50a7f787840bfce98"
MODEL_SHA = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
BLOBS = {
    "train.csv": "98e2543cf482d0dca7bfb175ebe35d98efad95be",
    "test.csv": "799687a8367359432985b8b13d85a2baf73f92dd",
    "categories.json": "cdd2a5c77a4079a455f8fb7e751d1ecee0e2a5a4",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


def freeze(data):
    for name, expected in BLOBS.items():
        raw = (data / name).read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    labels = sorted(json.loads((data / "categories.json").read_text()))
    if len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("Expected 77 unique categories")
    seen, source, excluded = {}, {}, []
    for split in ("train", "test"):
        source[split] = []
        with (data / f"{split}.csv").open(newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                text_sha = hashlib.sha256(" ".join(row["text"].lower().split()).encode()).hexdigest()
                item = dict(id=f"{split}:{index}", text=row["text"],
                            label=labels.index(row["category"]), text_sha256=text_sha)
                if text_sha in seen:
                    excluded.append(dict(id=item["id"], duplicates=seen[text_sha]["id"],
                                         label_conflict=item["label"] != seen[text_sha]["label"]))
                    continue
                seen[text_sha] = item
                source[split].append(item)
    fit, select = [], []
    for label in range(77):
        group = [r for r in source["train"] if r["label"] == label]
        group.sort(key=lambda r: hashlib.sha256(("banking77-r3:" + r["id"]).encode()).hexdigest())
        if len(group) <= 10:
            raise ValueError("Too few unique fitting rows for a class")
        select.extend(group[:10])
        fit.extend(group[10:])
    splits = {"fit": fit, "select": select, "test": source["test"]}
    for name, rows in splits.items():
        for row in rows:
            row["split"] = name
    return labels, splits, excluded


def choose(scores):
    return min(scores, key=lambda name: (-scores[name]["accuracy"], scores[name]["nll"], name))


def top_candidates(p, k=10):
    # Stable tie break; canonical order avoids putting the classifier winner first.
    return np.sort(np.argsort(-p, axis=1, kind="stable")[:, :k], axis=1)


def rerank_distribution(logits, candidate_ids, total_labels, prior=None, alpha=1.0, temperature=1.0):
    z = np.asarray(logits, dtype=float) * alpha / temperature
    if prior is not None:
        z += np.log(np.maximum(np.take_along_axis(prior, candidate_ids, axis=1), 1e-12))
    local = probabilities(z)
    full = np.zeros((len(z), total_labels), dtype=float)
    np.put_along_axis(full, candidate_ids, local, axis=1)
    return full


def infer_decisions(agent, rows, labels, candidate_ids, max_len, head_max_len, out, name):
    """Vary per-row criteria while reusing upstream encoding, collation, and forward."""
    from laya.common import collate_items

    logits, usages, batch_times = [], [], []
    started = time.perf_counter()
    with (out / f"{name}.jsonl").open("x") as stream:
        for start in range(0, len(rows), 8):
            batch_start = time.perf_counter()
            items = []
            for row, indices in zip(rows[start:start + 8], candidate_ids[start:start + 8]):
                question = {"type": "choice",
                            "instructions": "Which banking intent does `message` express?",
                            "criteria": {labels[i].replace("_", " "): None for i in indices}}
                agent._check_question("intent", question)
                internal = {"intent": agent._to_internal(question)}
                items.append(agent._encode_state({"message": row["text"]}, ["intent"], internal,
                                                 max_len=max_len, head_max_len=head_max_len))
            batch = collate_items(items, agent.tok.pad_token_id)
            z, _ = agent._forward(batch)
            for offset, (row, per_row) in enumerate(zip(rows[start:start + 8], items)):
                indices = candidate_ids[start + offset]
                value = z[offset, :len(indices)].astype(float).tolist()
                stats = dict(state=per_row[0]["state_stats"], options=per_row[0]["options"],
                             input_tokens=len(per_row[0]["ids"]))
                logits.append(value)
                usages.append(stats)
                stream.write(json.dumps(dict(id=row["id"], label=row["label"],
                                             candidates=indices.tolist(), logits=value, usage=stats)) + "\n")
            stream.flush()
            batch_times.append(time.perf_counter() - batch_start)
            if (start // 8) % 32 == 0:
                event("decision_progress", arm=name, rows=min(start + 8, len(rows)))
    timing = dict(seconds=time.perf_counter() - started, rows=len(rows),
                  batch_size=8, batch_seconds=batch_times,
                  max_len=max_len, head_max_len=head_max_len)
    write_json(out / f"{name}_timing.json", timing)
    return np.asarray(logits), usages, timing


def embed(agent, rows, out, name):
    from laya.shortlist import embed_fn_from_agent
    from sklearn.preprocessing import normalize

    fn = embed_fn_from_agent(agent, max_length=128, batch_size=32)
    parts, started = [], time.perf_counter()
    for start in range(0, len(rows), 256):
        parts.append(fn([r["text"] for r in rows[start:start + 256]]))
        event("embedding_progress", split=name, rows=min(start + 256, len(rows)))
    values = normalize(np.concatenate(parts))
    np.save(out / f"{name}_embeddings.npy", values)
    token_lengths = [len(agent.tok.encode(r["text"])) for r in rows]
    timing = dict(seconds=time.perf_counter() - started, rows=len(rows),
                  max_length=128, truncated=sum(n > 128 for n in token_lengths))
    write_json(out / f"{name}_embedding_timing.json", timing)
    return values, timing


def lexical_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import Normalizer

    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                    sublinear_tf=True, dtype=np.float64)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                    max_features=50000, sublinear_tf=True, dtype=np.float64)),
        ])),
        ("normalize", Normalizer()),
    ])


def train_heads(x, y, validation, target, strengths, name, out):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    models, preds, scores, traces = {}, {}, {}, {}
    for c in strengths:
        key = f"{name}_C{c}"
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(C=c, max_iter=400, solver="lbfgs", random_state=20261025)
            model.fit(x, y)
        if not np.array_equal(model.classes_, np.arange(77)):
            raise ValueError("Classifier label ordering changed")
        p = model.predict_proba(validation)
        models[key], preds[key], scores[key] = model, p, quality(p, target)
        traces[key] = dict(seconds=time.perf_counter() - started, iterations=model.n_iter_.tolist(),
                           warnings=[str(w.message) for w in caught])
        event("classifier_trained", arm=key, metrics=scores[key], timing=traces[key])
    winner = choose(scores)
    model = models[winner]
    np.savez(out / f"{name}_head.npz", coef=model.coef_, intercept=model.intercept_,
             classes=model.classes_)
    write_json(out / f"{name}_selection.json", dict(selected=winner, scores=scores, training=traces))
    return model, preds[winner], winner


def export_lexical(vectorizer, out):
    for name, vec in vectorizer.named_steps["features"].transformer_list:
        write_json(out / f"{name}_vocabulary.json", {k: int(v) for k, v in vec.vocabulary_.items()})
        np.save(out / f"{name}_idf.npy", vec.idf_)


def final_arms(agent, rows, labels, classifier_p, out, split):
    all_ids = np.tile(np.arange(len(labels)), (len(rows), 1))
    default_z, default_usage, default_timing = infer_decisions(
        agent, rows, labels, all_ids, 512, 192, out, f"{split}_laya_default")
    wide_z, wide_usage, wide_timing = infer_decisions(
        agent, rows, labels, all_ids, 1024, 768, out, f"{split}_laya_wide")
    shortlist = top_candidates(classifier_p)
    short_z, short_usage, short_timing = infer_decisions(
        agent, rows, labels, shortlist, 512, 256, out, f"{split}_laya_top10")
    t77 = agent.temperature_by_options["choice:11+"]
    t10 = agent.temperature_by_options["choice:6-10"]
    arms = {
        "laya_default": probabilities(default_z, t77),
        "laya_wide": probabilities(wide_z, t77),
        "supervised": classifier_p,
        "laya_top10": rerank_distribution(short_z, shortlist, 77, temperature=t10),
    }
    for alpha in (.25, 1):
        arms[f"fusion_{alpha}"] = rerank_distribution(
            short_z, shortlist, 77, prior=classifier_p, alpha=alpha, temperature=t10)
    diagnostics = dict(
        shortlist_recall=float(np.any(shortlist == np.array([r["label"] for r in rows])[:, None], axis=1).mean()),
        decision_timings=dict(default=default_timing["seconds"], wide=wide_timing["seconds"],
                              top10=short_timing["seconds"]),
        usage_first_row=dict(default=default_usage[0], wide=wide_usage[0], top10=short_usage[0]),
        truncated_rows={key: sum(u["state"]["truncated"] for u in usages)
                        for key, usages in (("default", default_usage), ("wide", wide_usage),
                                            ("top10", short_usage))},
    )
    return arms, diagnostics


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "upstream", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION or digest(args.model / "model.safetensors") != MODEL_SHA:
        raise ValueError("Checkpoint or upstream differs from protocol")
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite an experiment")
    args.output.mkdir(parents=True)
    labels, splits, excluded = freeze(args.data)
    protocol = dict(
        dataset_commit="57ec275d8078af65b7731c2a98be812d844a6d6b", data_git_blobs=BLOBS,
        source_sha256={name: digest(args.data / name) for name in BLOBS},
        model_sha256=MODEL_SHA, upstream_revision=REVISION, labels=labels,
        split_ids={key: [r["id"] for r in rows] for key, rows in splits.items()},
        exclusions=excluded, device="cpu", threads=8, seed=20261025,
        implementation_sha256={name: digest(Path(__file__).with_name(name))
                               for name in ("banking77.py", "metrics.py", "PROTOCOL_ROUND3.md")},
    )
    write_json(args.output / "protocol.json", protocol)
    with (args.output / "rows.jsonl").open("x") as stream:
        for rows in splits.values():
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    event("protocol_frozen", sizes={k: len(v) for k, v in splits.items()}, exclusions=len(excluded))
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(args.upstream))
    import torch
    import laya
    import sklearn
    torch.set_num_threads(8)
    started = time.perf_counter()
    agent = laya.load(str(args.model), device="cpu", backend="eager")
    write_json(args.output / "environment.json", dict(
        torch=torch.__version__, laya=laya.__version__, sklearn=sklearn.__version__,
        numpy=np.__version__, dtype=str(agent.dtype), load_seconds=time.perf_counter() - started))
    fit, select, test = (splits[k] for k in ("fit", "select", "test"))
    yfit, yselect = np.array([r["label"] for r in fit]), np.array([r["label"] for r in select])
    vectorizer = lexical_vectorizer()
    sparse_fit = vectorizer.fit_transform([r["text"] for r in fit])
    sparse_select = vectorizer.transform([r["text"] for r in select])
    sparse_head, sparse_p, sparse_name = train_heads(
        sparse_fit, yfit, sparse_select, yselect, (1, 4, 16), "lexical", args.output)
    export_lexical(vectorizer, args.output)
    dense_fit, _ = embed(agent, fit, args.output, "fit")
    dense_select, _ = embed(agent, select, args.output, "select")
    dense_head, dense_p, dense_name = train_heads(
        dense_fit, yfit, dense_select, yselect, (1, 10, 100), "dense", args.output)
    mixed = {str(w): (1 - w) * sparse_p + w * dense_p for w in (0, .25, .5, .75, 1)}
    mix_scores = {name: quality(p, yselect) for name, p in mixed.items()}
    weight = float(choose(mix_scores))
    validation_arms, validation_diagnostics = final_arms(
        agent, select, labels, mixed[str(int(weight)) if weight in (0, 1) else str(weight)],
        args.output, "select")
    val_scores = {name: quality(p, yselect) for name, p in validation_arms.items()}
    winner = choose(val_scores)
    selection = dict(selected=winner, lexical=sparse_name, dense=dense_name,
                     dense_weight=weight, mix_scores=mix_scores,
                     validation=val_scores, diagnostics=validation_diagnostics,
                     no_test_metrics_computed=True)
    write_json(args.output / "selection.json", selection)
    event("selection_locked", selected=winner, dense_weight=weight, validation=val_scores)
    # Only now execute the untouched test set.
    dense_test, _ = embed(agent, test, args.output, "test")
    sparse_test = vectorizer.transform([r["text"] for r in test])
    sparse_test_p, dense_test_p = sparse_head.predict_proba(sparse_test), dense_head.predict_proba(dense_test)
    classifier_p = (1 - weight) * sparse_test_p + weight * dense_test_p
    arms, diagnostics = final_arms(agent, test, labels, classifier_p, args.output, "test")
    arms.update(lexical=sparse_test_p, dense=dense_test_p)
    ytest = np.array([r["label"] for r in test])
    results = dict(
        selected_on_validation_only=winner,
        test_metrics={name: quality(p, ytest) for name, p in arms.items()},
        selected_vs_default=paired_accuracy(arms["laya_default"], arms[winner], ytest),
        selected_vs_lexical=paired_accuracy(arms["lexical"], arms[winner], ytest),
        selected_vs_supervised=paired_accuracy(arms["supervised"], arms[winner], ytest),
        diagnostics=diagnostics, selection_sha256=digest(args.output / "selection.json"),
        scope="Target-supervised Banking77 fixed-schema study; not a Jev comparison or SOTA claim",
    )
    with (args.output / "test_predictions.jsonl").open("x") as stream:
        for i, row in enumerate(test):
            stream.write(json.dumps(dict(
                id=row["id"], label=row["label"], text_sha256=row["text_sha256"],
                probabilities={name: p[i].tolist() for name, p in arms.items()})) + "\n")
    write_json(args.output / "results.json", results)
    event("complete", results=results)


if __name__ == "__main__":
    main()
