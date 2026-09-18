"""Validation for the TypeSafe-shaped request contract."""
from typing import Any


def validate_request(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in payload:
        raise ValueError("state is required")
    if not isinstance(payload.get("model"), str):
        raise ValueError("model is required and must be a string")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions is required and must be a non-empty object")
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("question ids must be non-empty strings")
        if not isinstance(question, dict):
            raise ValueError(f"questions.{question_id} must be an object")
        kind = question.get("type")
        if kind not in {"choice", "noul", "score"}:
            raise ValueError(f"questions.{question_id}.type must be choice, noul, or score")
        if "instructions" not in question:
            raise ValueError(f"questions.{question_id}.instructions is required")
        criteria = question.get("criteria")
        if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
            raise ValueError(f"questions.{question_id}.criteria must contain at least two choices")
        if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
            raise ValueError(f"questions.{question_id}.criteria must contain at least two levels")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"questions.{question_id}.criteria must be an object when provided")
