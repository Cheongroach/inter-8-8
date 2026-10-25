"""Second calibration round: unused target-domain data and a fresh holdout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from evaluate import CRITERIA, EXPECTED_REVISION, freeze_splits, sha256
from metrics import apply_affine, fit_affine, fit_temperature, paired_accuracy, probabilities, quality


def main():
    parser = argparse.ArgumentParser()
    for key in ('upstream', 'model', 'data', 'previous', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != EXPECTED_REVISION:
        raise ValueError('Upstream revision changed')
    old = json.loads(args.previous.read_text())
    if sha256(args.model / 'model.safetensors') != old['model_sha256']:
        raise ValueError('Checkpoint changed')
    for name in ('train', 'test'):
        if sha256(args.data / f'{name}.csv') != old['data_sha256'][name]:
            raise ValueError('Source data changed')
    pool = freeze_splits(args.data, 1, 1, 2304)['test']
    excluded = set(old['split_ids']['test'])
    if {row['id'] for row in pool[:256]} != excluded:
        raise ValueError('Previous test set differs from excluded prefix')
    fresh = pool[256:]
    splits = {'fit': fresh[:1024], 'select': fresh[1024:1536], 'test': fresh[1536:2048]}
    if any(row['id'] in excluded for subset in splits.values() for row in subset):
        raise ValueError('Previously inspected test item reused')
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite a second-round experiment')
    args.output.mkdir(parents=True)
    rows = [dict(row, split=name) for name, subset in splits.items() for row in subset]
    protocol = {
        'round': 2, 'upstream_revision': revision, 'model_sha256': old['model_sha256'],
        'data_sha256': old['data_sha256'],
        'split_ids': {name: [row['id'] for row in subset] for name, subset in splits.items()},
        'excluded_previous_test_ids': sorted(excluded),
        'source_pool': 'unused official AG News test rows; calibration/validation/holdout split',
        'scope': 'custom target-domain calibration; not official full-test benchmark',
        'selection': 'validation NLL, subject to no validation accuracy loss vs shipped',
        'affine_prior': 'identity on already temperature-scaled logits',
        'candidates': ['shipped', 'temperature', 'affine_0.1', 'affine_1.0', 'affine_10.0'],
        'implementation_sha256': {name: sha256(Path(__file__).with_name(name))
                                  for name in ('iterate.py', 'evaluate.py', 'metrics.py')},
        'device': 'cpu', 'threads': 8, 'batch_size': 8,
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    sys.path.insert(0, str(args.upstream))
    import torch
    import laya
    torch.set_num_threads(8)
    agent = laya.load(str(args.model), device='cpu', backend='eager')
    questions = {'topic': {'type': 'choice', 'instructions': 'What is the topic of `article`?',
                           'criteria': CRITERIA}}
    agent.predict_batch([{'article': rows[0]['state']}], questions, batch_size=1)
    captured = []
    original = agent._forward
    def capture(batch):
        z, action = original(batch)
        captured.extend(z[:, :4].copy().tolist())
        if len(captured) % 256 == 0:
            print(json.dumps({'event': 'inference_progress', 'rows': len(captured)}), flush=True)
        return z, action
    agent._forward = capture
    started = time.perf_counter()
    try:
        outputs = agent.predict_batch([{'article': row['state']} for row in rows], questions,
                                      batch_size=8, sort_by_length=False)
    finally:
        agent._forward = original
    seconds = time.perf_counter() - started
    if len(captured) != 2048 or len(outputs) != 2048:
        raise ValueError('Inference lost or duplicated rows')
    with (args.output / 'predictions.jsonl').open('w') as stream:
        for row, z, output in zip(rows, captured, outputs):
            stream.write(json.dumps({'id': row['id'], 'label': row['label'], 'split': row['split'],
                                     'text_sha256': row['text_sha256'], 'logits': z,
                                     'usage': output.get('usage')}) + '\n')
    logits = np.asarray(captured, dtype=float)
    y = np.asarray([row['label'] for row in rows])
    masks = {name: np.asarray([row['split'] == name for row in rows]) for name in splits}
    cfg = json.loads((args.model / 'rl_agent_config.json').read_text())
    shipped_t = cfg['temperature_by_options']['choice:3-5']
    candidates = {'shipped': probabilities(logits, shipped_t)}
    parameters = {'shipped': {'temperature': shipped_t}}
    fitted_t = fit_temperature(logits[masks['fit']], y[masks['fit']])
    candidates['temperature'] = probabilities(logits, fitted_t)
    parameters['temperature'] = {'temperature': fitted_t}
    for penalty in (0.1, 1.0, 10.0):
        # The identity prior reproduces the shipped calibrated distribution.
        # Raw logits would instead regularize toward temperature one.
        w, b = fit_affine(logits[masks['fit']] / shipped_t, y[masks['fit']], penalty)
        name = f'affine_{penalty}'
        candidates[name] = apply_affine(logits / shipped_t, w, b)
        parameters[name] = {'input_temperature': shipped_t, 'penalty': penalty,
                            'weight': w.tolist(), 'bias': b.tolist()}
    validation = {name: quality(p[masks['select']], y[masks['select']]) for name, p in candidates.items()}
    eligible = [name for name in candidates if validation[name]['accuracy'] >= validation['shipped']['accuracy']]
    selected = min(eligible, key=lambda name: (validation[name]['nll'], name != 'shipped', name))
    test = {name: quality(p[masks['test']], y[masks['test']]) for name, p in candidates.items()}
    baseline = candidates['shipped'][masks['test']]
    candidate = candidates[selected][masks['test']]
    labels = y[masks['test']]
    idx = np.arange(len(labels))
    nll_difference = -np.log(np.maximum(candidate[idx, labels], 1e-12)) + np.log(np.maximum(baseline[idx, labels], 1e-12))
    rng = np.random.default_rng(20261025)
    nll_bootstrap = rng.choice(nll_difference, (2000, len(labels)), replace=True).mean(axis=1)
    result = {
        'round': 2, 'selected_on_validation_only': selected, 'selected_parameters': parameters[selected],
        'selection_metrics': validation, 'test_metrics': test,
        'test_paired_selected_vs_shipped': paired_accuracy(baseline, candidate, labels),
        'test_paired_nll_difference': {'mean': float(nll_difference.mean()),
            'ci95': np.quantile(nll_bootstrap, [0.025, 0.975]).tolist(),
            'direction': 'negative favors selected', 'draws': 2000, 'seed': 20261025},
        'inference_seconds': seconds, 'inferred_rows': len(rows),
        'torch': torch.__version__, 'laya': laya.__version__, 'dtype': str(agent.dtype),
        'timing_scope': 'CPU eager, eight-thread/eight-row batches, excluding model load',
        'quality_scope': 'fresh 512-case custom holdout; no Jev or SOTA claim',
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
