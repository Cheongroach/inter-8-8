"""Quality metrics and validation-selected probability recalibration."""

import numpy as np


def probabilities(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    if not np.isfinite(z).all() or temperature <= 0:
        raise ValueError("Finite logits and positive temperature required")
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def quality(p, labels, bins=15):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or len(p) != len(y) or len(y) == 0:
        raise ValueError("Nonempty aligned probability rows and labels required")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Probability rows must sum to one")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Label outside candidate schema")
    correct = p.argmax(axis=1) == y
    confidence = p.max(axis=1)
    ece = 0.0
    for b in range(bins):
        mask = (confidence >= b / bins) & (
            (confidence < (b + 1) / bins) if b < bins - 1 else (confidence <= 1)
        )
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    one_hot = np.eye(p.shape[1])[y]
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
        "brier_multiclass_sum": float(((p - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def fit_temperature(logits, labels):
    candidates = np.geomspace(0.25, 16.0, 161)
    y = np.asarray(labels)
    losses = [
        -np.log(np.maximum(probabilities(logits, t)[np.arange(len(y)), y], 1e-12)).mean()
        for t in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def fit_affine(logits, labels, penalty, steps=400):
    """Fit a small domain calibration map, regularized toward identity.

    This changes output probabilities, not the pretrained encoder weights.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    y = np.asarray(labels, dtype=np.int64)
    k = z.shape[1]
    weight = np.eye(k)
    bias = np.zeros(k)
    targets = np.eye(k)[y]
    moments = [np.zeros_like(weight), np.zeros_like(bias)]
    variances = [np.zeros_like(weight), np.zeros_like(bias)]
    for step in range(1, steps + 1):
        residual = (probabilities(z @ weight + bias) - targets) / len(y)
        grads = [
            z.T @ residual + penalty * (weight - np.eye(k)),
            residual.sum(axis=0) + penalty * bias,
        ]
        for index, (parameter, grad) in enumerate(zip((weight, bias), grads)):
            moments[index] = 0.9 * moments[index] + 0.1 * grad
            variances[index] = 0.999 * variances[index] + 0.001 * grad**2
            parameter -= 0.01 * (moments[index] / (1 - 0.9**step)) / (
                np.sqrt(variances[index] / (1 - 0.999**step)) + 1e-8
            )
    return weight, bias


def apply_affine(logits, weight, bias):
    z = np.asarray(logits, dtype=np.float64)
    return probabilities((z - z.mean(axis=1, keepdims=True)) @ weight + bias)


def paired_accuracy(base, candidate, labels, seed=20261025, draws=2000):
    y = np.asarray(labels)
    delta = (np.asarray(candidate).argmax(axis=1) == y).astype(float) - (
        np.asarray(base).argmax(axis=1) == y
    ).astype(float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "method": "paired case bootstrap; exact-text duplicates removed",
        "seed": seed,
        "draws": draws,
    }
