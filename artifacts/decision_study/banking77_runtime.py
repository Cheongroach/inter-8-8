"""Load the validation-selected Banking77 route without fitting on serving data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from banking77 import MODEL_SHA, REVISION, digest, lexical_vectorizer, rerank_distribution, top_candidates
from metrics import probabilities


def restore_vectorizer(root):
    vectorizer = lexical_vectorizer()
    for name, vec in vectorizer.named_steps["features"].transformer_list:
        vec.set_params(vocabulary=json.loads((root / f"{name}_vocabulary.json").read_text()))
        vec.idf_ = np.load(root / f"{name}_idf.npy", allow_pickle=False)
    return vectorizer


def head_predict(root, name, x):
    with np.load(root / f"{name}_head.npz", allow_pickle=False) as parameters:
        if not np.array_equal(parameters["classes"], np.arange(77)):
            raise ValueError("Unsupported classifier label order")
        return probabilities(x @ parameters["coef"].T + parameters["intercept"])


class BankingRouter:
    def __init__(self, artifact, upstream, model):
        self.root = Path(artifact)
        self.selection = json.loads((self.root / "selection.json").read_text())
        self.labels = json.loads((self.root / "protocol.json").read_text())["labels"]
        self.selected = self.selection["selected"]
        if self.selected not in ("laya_default", "laya_wide", "supervised", "laya_top10",
                                 "fusion_0.25", "fusion_1"):
            raise ValueError("Unknown selected routing method")
        actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if actual != REVISION or digest(Path(model) / "model.safetensors") != MODEL_SHA:
            raise ValueError("Pinned encoder or upstream mismatch")
        sys.path.insert(0, str(upstream))
        import torch
        import laya
        from laya.shortlist import embed_fn_from_agent

        torch.set_num_threads(8)
        self.agent = laya.load(str(model), device="cpu", backend="eager")
        self.embed = embed_fn_from_agent(self.agent, max_length=128, batch_size=32)
        self.vectorizer = restore_vectorizer(self.root)

    def predict(self, texts):
        from sklearn.preprocessing import normalize

        texts = list(texts)
        if not texts:
            return np.zeros((0, 77))
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("Each request needs nonempty text")
        classifier = None
        if self.selected not in ("laya_default", "laya_wide"):
            w = self.selection["dense_weight"]
            classifier = np.zeros((len(texts), 77))
            if w < 1:
                classifier += (1 - w) * head_predict(
                    self.root, "lexical", self.vectorizer.transform(texts))
            if w > 0:
                classifier += w * head_predict(self.root, "dense", normalize(self.embed(texts)))
        if self.selected == "supervised":
            return classifier
        ids = (np.tile(np.arange(77), (len(texts), 1)) if classifier is None
               else top_candidates(classifier))
        wide = self.selected == "laya_wide"
        max_len = 1024 if wide else 512
        head_len = 768 if wide else (192 if classifier is None else 256)
        captured = []
        original = self.agent._forward

        def capture(batch):
            logits, act = original(batch)
            captured.extend(logits.astype(float).tolist())
            return logits, act

        self.agent._forward = capture
        try:
            for text, order in zip(texts, ids):
                questions = {"intent": {
                    "type": "choice", "instructions": "Which banking intent does `message` express?",
                    "criteria": {self.labels[i].replace("_", " "): None for i in order},
                }}
                self.agent.predict_batch([{"message": text}], questions, batch_size=1,
                                         max_len=max_len, head_max_len=head_len)
        finally:
            self.agent._forward = original
        if len(captured) != len(texts):
            raise RuntimeError("Decision rows do not match request rows")
        if classifier is None:
            return probabilities(captured, self.agent.temperature_by_options["choice:11+"])
        alpha = float(self.selected.split("_", 1)[1]) if self.selected.startswith("fusion_") else 1
        prior = classifier if self.selected.startswith("fusion_") else None
        return rerank_distribution(
            captured, ids, 77, prior=prior, alpha=alpha,
            temperature=self.agent.temperature_by_options["choice:6-10"])


def main():
    parser = argparse.ArgumentParser()
    for name in ("artifact", "upstream", "model", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    requests = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("Request IDs must be unique")
    router = BankingRouter(args.artifact, args.upstream, args.model)
    p = router.predict([r["text"] for r in requests])
    with args.output.open("x") as stream:
        for request_id, distribution in zip(ids, p):
            stream.write(json.dumps(dict(
                id=request_id, choice=router.labels[int(distribution.argmax())],
                probabilities=dict(zip(router.labels, distribution.tolist())),
                model=router.selected, scope="Banking77 fixed 77-label schema",
            )) + "\n")


if __name__ == "__main__":
    main()
