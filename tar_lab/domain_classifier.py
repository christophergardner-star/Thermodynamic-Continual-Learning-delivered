"""
Domain classifier for TAR gap scanner.

Replaces the brittle keyword-scoring approach in science_profiles.py with a
calibrated ML classifier (TF-IDF + logistic regression) that outputs
per-domain probabilities.

Usage:
    from tar_lab.domain_classifier import classify_domain
    result = classify_domain("catastrophic forgetting elastic weight consolidation")
    # → {"domain": "continual_learning", "confidence": 0.97, "ambiguous": False}

Build / retrain the model:
    python -m tar_lab.domain_classifier --build
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

_MODEL_PATH = Path(__file__).resolve().parents[1] / "tar_state" / "models" / "domain_classifier.pkl"
_AMBIGUITY_THRESHOLD = 0.50   # max probability below this → ambiguous
_FINANCE_HARD_KEYWORDS = frozenset({
    "portfolio", "mean-variance", "asset allocation", "quantitative finance",
    "econometric", "option pricing", "hedge fund", "sharpe ratio",
    "coskewness", "derivative pricing", "black-scholes", "stochastic volatility",
})

# ---------------------------------------------------------------------------
# Training corpus
# ---------------------------------------------------------------------------

# 60 labeled (text, label) pairs across 4 domains.
# Real examples from frontier_problems.json and frontier_gaps.jsonl;
# supplementary examples from standard ML and finance literature knowledge.
_TRAINING_EXAMPLES: list[tuple[str, str]] = [
    # ── continual_learning ──────────────────────────────────────────────────
    ("Production ML systems must absorb new data over long horizons without erasing previously learned capabilities. Catastrophic forgetting remains unsolved at industrial scale.", "continual_learning"),
    ("Real deployed systems often see new categories appear without clean task boundaries. Models must expand without forgetting past classes.", "continual_learning"),
    ("A continual-learning method is only practically useful if it remains stable as data distribution shifts and new tasks arrive.", "continual_learning"),
    ("Production continual-learning systems should not require expensive manual retuning for each new task.", "continual_learning"),
    ("Elastic weight consolidation protects important parameters from catastrophic forgetting by regularising toward previous task optima.", "continual_learning"),
    ("Synaptic intelligence online importance estimation via continuous gradient accumulation reduces interference between tasks.", "continual_learning"),
    ("Progressive neural networks allocate new capacity for each task and prevent forgetting via lateral connections.", "continual_learning"),
    ("Learning without forgetting distills knowledge from the previous model using soft targets to preserve old task accuracy.", "continual_learning"),
    ("Dark experience replay stores raw logits for past inputs and replays them as a soft constraint during new task training.", "continual_learning"),
    ("Gradient episodic memory constrains parameter updates to not increase loss on stored exemplars from previous tasks.", "continual_learning"),
    ("Averaged gradient episodic memory improves GEM by projecting onto the average gradient rather than enforcing hard constraints.", "continual_learning"),
    ("Continual learning benchmark on split-CIFAR-10 with task-incremental and class-incremental protocols.", "continual_learning"),
    ("Forgetting is measured as the decrease in per-task accuracy after training on subsequent tasks.", "continual_learning"),
    ("Backward transfer quantifies how learning a new task affects performance on previously learned tasks.", "continual_learning"),
    ("Permuted MNIST is a standard benchmark for evaluating catastrophic forgetting in neural networks.", "continual_learning"),
    ("Regularisation-based continual learning methods add penalty terms to the objective to prevent parameter drift.", "continual_learning"),

    # ── thermodynamics_ml ───────────────────────────────────────────────────
    ("Adaptive ML systems need reliable internal state detection so they can tell when to protect prior knowledge versus explore new representations.", "thermodynamics_ml"),
    ("Thermodynamic analogies for neural network training: entropy measures parameter disorder, energy tracks gradient accumulation.", "thermodynamics_ml"),
    ("Gradient energy accumulation as a proxy for Fisher information provides a continuous importance estimate throughout training.", "thermodynamics_ml"),
    ("Regime detection identifies ordered, disordered, and critical phases in neural network training dynamics.", "thermodynamics_ml"),
    ("Thermal importance weighting assigns higher protection to parameters with high gradient energy across a training episode.", "thermodynamics_ml"),
    ("The participation ratio quantifies the effective dimensionality of feature representations, analogous to thermodynamic degrees of freedom.", "thermodynamics_ml"),
    ("Non-equilibrium statistical mechanics provides a formal framework for understanding learning dynamics in overparameterised networks.", "thermodynamics_ml"),
    ("EMA of squared gradients approximates the diagonal Fisher information matrix under stationarity assumptions.", "thermodynamics_ml"),
    ("Free energy principles applied to Bayesian deep learning as variational inference with thermodynamic priors.", "thermodynamics_ml"),

    # ── finance_economics ───────────────────────────────────────────────────
    ("Unrestricted mean-variance-skewness-kurtosis portfolio optimization can capture non-normal return distributions.", "finance_economics"),
    ("Sharpe ratio maximisation under transaction cost constraints for high-frequency trading strategies.", "finance_economics"),
    ("Black-Scholes model for European option pricing with stochastic volatility extensions.", "finance_economics"),
    ("Asset allocation across equities, bonds, and alternatives using mean-variance efficient frontier analysis.", "finance_economics"),
    ("Econometric methods for estimating GARCH models of conditional volatility in financial time series.", "finance_economics"),
    ("Hedge fund return attribution decomposition using Fama-French factor models.", "finance_economics"),
    ("Covariance matrix estimation for large portfolios under the Ledoit-Wolf shrinkage estimator.", "finance_economics"),
    ("Risk-return tradeoff in multi-asset portfolios with tail risk constraints using CVaR optimisation.", "finance_economics"),
    ("Derivative pricing under jump-diffusion processes for exotic options.", "finance_economics"),
    ("Quantitative finance backtesting frameworks for systematic trading strategy evaluation.", "finance_economics"),
    ("Portfolio rebalancing frequency and transaction costs under momentum and mean-reversion signals.", "finance_economics"),
    ("Coskewness and cokurtosis portfolio optimization for non-Gaussian return distributions.", "finance_economics"),

    # ── other_ml ────────────────────────────────────────────────────────────
    ("ResNet architectures for image classification on ImageNet with residual skip connections.", "other_ml"),
    ("Transformer attention mechanisms for natural language processing and sequence-to-sequence tasks.", "other_ml"),
    ("Generative adversarial networks for photorealistic image synthesis and style transfer.", "other_ml"),
    ("Object detection using YOLO with anchor-free prediction heads for real-time inference.", "other_ml"),
    ("Self-supervised contrastive learning for visual representation without labelled data.", "other_ml"),
    ("Neural architecture search using reinforcement learning or differentiable proxies.", "other_ml"),
    ("Federated learning for privacy-preserving distributed training across edge devices.", "other_ml"),
    ("Knowledge distillation compresses large teacher models into compact student networks.", "other_ml"),
    ("Dropout regularisation as approximate Bayesian inference in deep neural networks.", "other_ml"),
    ("Batch normalisation accelerates training by reducing internal covariate shift.", "other_ml"),
    ("Graph neural networks for molecular property prediction and drug discovery.", "other_ml"),
    ("Reinforcement learning with proximal policy optimisation for continuous control tasks.", "other_ml"),
    ("Diffusion probabilistic models for high-quality image and audio generation.", "other_ml"),
    ("Large language model pretraining on diverse text corpora using next-token prediction.", "other_ml"),
    ("Vision-language contrastive pretraining aligns image and text representations.", "other_ml"),
]


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def build_classifier(examples: list[tuple[str, str]] | None = None) -> object:
    """Train a calibrated TF-IDF + logistic-regression classifier.

    Returns the fitted sklearn Pipeline.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError(f"scikit-learn required for domain classifier: {exc}") from exc

    data = examples or _TRAINING_EXAMPLES
    texts, labels = zip(*data)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("lr", CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=1000, random_state=42))),
    ])
    clf.fit(list(texts), list(labels))
    return clf


def load_classifier() -> Optional[object]:
    """Load the persisted classifier, returning None if not found."""
    if not _MODEL_PATH.exists():
        return None
    with _MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301


def save_classifier(clf: object) -> None:
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MODEL_PATH.open("wb") as fh:
        pickle.dump(clf, fh)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_domain(text: str, clf: object | None = None) -> dict:
    """Classify a problem/gap description into a domain.

    Returns:
        {
            "domain": str,        # top predicted domain
            "confidence": float,  # probability of top domain
            "probabilities": dict,# per-domain probs
            "ambiguous": bool,    # True if max prob < threshold
            "finance_hard_block": bool,  # True if keyword filter fires
        }
    """
    lowered = text.lower()

    # Hard keyword filter as a safety backstop (preserved from Phase 0.6).
    # Fires even when the ML classifier is unavailable.
    finance_hard_block = any(kw in lowered for kw in _FINANCE_HARD_KEYWORDS)

    if clf is None:
        clf = load_classifier()

    if clf is None:
        # Classifier not built yet — fall back to keyword heuristic.
        domain = "finance_economics" if finance_hard_block else "unknown"
        return {
            "domain": domain,
            "confidence": 1.0 if finance_hard_block else 0.0,
            "probabilities": {},
            "ambiguous": not finance_hard_block,
            "finance_hard_block": finance_hard_block,
            "fallback": "keyword_only",
        }

    try:
        probs = clf.predict_proba([text])[0]
        classes = clf.classes_
        prob_dict = dict(zip(classes, probs.tolist()))
        top_domain = max(prob_dict, key=prob_dict.__getitem__)
        confidence = prob_dict[top_domain]

        # Finance hard block overrides ML classifier prediction.
        if finance_hard_block:
            top_domain = "finance_economics"
            confidence = max(confidence, prob_dict.get("finance_economics", 0.8))

        return {
            "domain": top_domain,
            "confidence": confidence,
            "probabilities": prob_dict,
            "ambiguous": confidence < _AMBIGUITY_THRESHOLD,
            "finance_hard_block": finance_hard_block,
        }
    except Exception as exc:
        domain = "finance_economics" if finance_hard_block else "unknown"
        return {
            "domain": domain,
            "confidence": 1.0 if finance_hard_block else 0.0,
            "probabilities": {},
            "ambiguous": not finance_hard_block,
            "finance_hard_block": finance_hard_block,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# CLI: python -m tar_lab.domain_classifier --build [--validate]
# ---------------------------------------------------------------------------

def _validate(clf: object) -> None:
    """Quick held-out validation on 10 manually labelled examples."""
    from sklearn.metrics import accuracy_score

    held_out = [
        ("Online Fisher information approximation for elastic regularisation in neural networks", "thermodynamics_ml"),
        ("Catastrophic interference in connectionist networks task-switching", "continual_learning"),
        ("Black-Scholes partial differential equation for vanilla call options", "finance_economics"),
        ("Attention is all you need transformer architecture", "other_ml"),
        ("EWC prevents catastrophic forgetting by constraining weight changes near Fisher optima", "continual_learning"),
        ("Mean-variance portfolio optimisation with fat-tail risk adjustments", "finance_economics"),
        ("Convolutional neural network feature extraction for image recognition", "other_ml"),
        ("Gradient energy accumulation thermodynamic regime detection", "thermodynamics_ml"),
        ("Continual learning split-CIFAR benchmark evaluation", "continual_learning"),
        ("Stochastic volatility models for exotic derivative pricing", "finance_economics"),
    ]
    texts, true_labels = zip(*held_out)
    predicted = clf.predict(list(texts))
    acc = accuracy_score(list(true_labels), list(predicted))
    print(f"Hold-out accuracy: {acc*100:.1f}% ({int(acc*len(held_out))}/{len(held_out)} correct)")
    if acc < 0.80:
        print("WARNING: accuracy below 80% threshold — review training data before deploying.")
    else:
        print("Validation passed (>= 80%).")
    for text, true, pred in zip(texts, true_labels, predicted):
        status = "OK" if true == pred else "FAIL"
        print(f"  [{status}] true={true} pred={pred} | {text[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TAR domain classifier")
    parser.add_argument("--build", action="store_true", help="Train and save classifier")
    parser.add_argument("--validate", action="store_true", help="Run held-out validation after build")
    parser.add_argument("--classify", metavar="TEXT", help="Classify a single text")
    args = parser.parse_args()

    if args.build:
        print(f"Training on {len(_TRAINING_EXAMPLES)} examples...")
        clf = build_classifier()
        save_classifier(clf)
        print(f"Saved to {_MODEL_PATH}")
        if args.validate:
            _validate(clf)

    elif args.validate:
        clf = load_classifier()
        if clf is None:
            print("No classifier found. Run with --build first.", file=sys.stderr)
            sys.exit(1)
        _validate(clf)

    elif args.classify:
        result = classify_domain(args.classify)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
