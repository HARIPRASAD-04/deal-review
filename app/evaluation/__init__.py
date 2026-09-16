"""Evaluation package — Module 8.

Infrastructure for running non-agent deterministic evaluation suites.
"""

from app.evaluation.runner import EvaluationCase, EvaluationResult, evaluate_case

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "evaluate_case",
]
