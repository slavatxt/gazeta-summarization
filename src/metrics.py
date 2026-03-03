"""ROUGE метрики для суммаризации."""
import numpy as np
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def evaluate_rouge(predictions, references):
    """Средний ROUGE F-score."""
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        s = _scorer.score(ref, pred)
        for key in scores:
            scores[key].append(s[key].fmeasure)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def print_rouge(name, sc):
    print(f"{name:<25} R1={sc['rouge1']:.4f}  R2={sc['rouge2']:.4f}  RL={sc['rougeL']:.4f}")
