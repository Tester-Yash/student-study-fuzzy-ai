"""
models.py
=========
Pydantic data models used across the project.

Using Pydantic provides:
    - strict schema validation
    - automatic type conversion
    - bounds checking (min/max ranges)
    - clean structured output handling for LangChain
"""

from typing import Optional
from pydantic import BaseModel, Field


class ExtractedStudentData(BaseModel):
    """
    Schema returned directly by the LangChain extraction chain in a single LLM call.
    Includes structured relevance detection and qualitative to numeric mapping.
    """

    is_study_related: bool = Field(
        ...,
        description=(
            "Set to True if the student's message describes an actual study/exam situation "
            "with meaningful details (e.g. days until exam, study hours, or preparation level). "
            "Set to False for casual greetings ('hello', 'hi'), chit-chat, off-topic questions, "
            "or text lacking any academic study context."
        ),
    )
    days_until_exam: Optional[int] = Field(
        default=None,
        ge=0,
        le=30,
        description="Days remaining until the exam (0-30). If tomorrow use 1, today use 0, next week use 7. None if not study related.",
    )
    study_hours: Optional[float] = Field(
        default=None,
        ge=0,
        le=12,
        description="Hours the student studied today (0-12). If not explicitly mentioned in a study situation, default to 0. None if not study related.",
    )
    preparation_level: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Self-rated preparation level as a percentage (0-100). Poor ~ 20, average ~ 50, good ~ 80. None if not study related.",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="If is_study_related is False, provide a brief, polite explanation of why (e.g. 'Greeting without study context').",
    )


class StudentStudyData(BaseModel):
    """
    Validated quantitative study data passed directly into the fuzzy logic engine.
    Ensures all numbers are strictly bounded within their respective universes.
    """

    days_until_exam: int = Field(
        ...,
        ge=0,
        le=30,
        description="Number of days remaining until the exam (0-30).",
    )
    study_hours: float = Field(
        ...,
        ge=0,
        le=12,
        description="Number of hours the student studied today (0-12).",
    )
    preparation_level: float = Field(
        ...,
        ge=0,
        le=100,
        description="Self-rated preparation level as a percentage (0-100).",
    )


class ExplanationResult(BaseModel):
    """Structured output of the second LangChain explanation chain."""

    explanation: str = Field(..., description="A friendly, supportive explanation of the recommendation.")
    tips: list[str] = Field(..., description="A list of 3 to 5 practical, actionable study tips.")
