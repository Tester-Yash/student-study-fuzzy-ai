"""
extractor.py
============
FIRST LANGCHAIN CHAIN:
    Natural-language student input -> LangChain prompt -> LLM
    -> structured Pydantic output (ExtractedStudentData -> StudentStudyData)

Performs genuine natural language understanding in a single structured LLM call.
Extracts:
    - days_until_exam (0-30)
    - study_hours (0-12)
    - preparation_level (0-100)

Safely detects and rejects:
    - empty or very short messages
    - casual greetings ("hello", "hi")
    - unrelated questions
    - incomplete descriptions lacking study context
"""

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from models import ExtractedStudentData, StudentStudyData
from llm_config import get_llm


class ExtractionError(Exception):
    """Raised when the student input is invalid, casual, irrelevant, or incomplete."""
    pass


EXTRACTION_SYSTEM_PROMPT = """You are an academic advisor AI analyzing a student's study situation.
Your job is to read their natural-language message and extract 3 quantitative parameters:

1. `days_until_exam`: Days remaining until the exam (integer 0 to 30).
   - "tomorrow" -> 1
   - "today" -> 0
   - "in 5 days" -> 5
   - "next week" -> 7
   - "in two weeks" -> 14

2. `study_hours`: Hours the student already studied today (float 0.0 to 12.0).
   - If not mentioned in an otherwise valid study situation, set to 0.0.

3. `preparation_level`: Self-rated preparation level as a percentage (float 0.0 to 100.0).
   - "poor" / "bad" / "starting now" / "low" -> ~20.0
   - "average" / "okay" / "moderate" / "decent" -> ~50.0
   - "good" / "high" / "confident" / "well prepared" -> ~80.0
   - If an explicit percentage or score is given (e.g. "prepared 40%"), use it.

Relevance Rules:
- If the message is a casual greeting (e.g. "hello", "hi", "hey"), chit-chat, unrelated question (e.g. weather, recipes, code), or lacks any meaningful study timeframe/preparation details, set `is_study_related = False` and provide a helpful `rejection_reason`.
- Only set `is_study_related = True` if the message genuinely describes a student's study situation.
"""


def extract_student_data(user_text: str) -> StudentStudyData:
    """
    Run the LangChain extraction chain in a single structured LLM call.

    Returns:
        StudentStudyData: Validated object with days_until_exam, study_hours, and preparation_level.

    Raises:
        ExtractionError: If input is empty, too short, irrelevant, or incomplete.
    """
    cleaned = (user_text or "").strip()
    if not cleaned:
        raise ExtractionError(
            "Please describe your study situation (e.g. how many days until your exam, "
            "how many hours you studied today, and how prepared you feel)."
        )

    if len(cleaned) < 4:
        raise ExtractionError(
            "That message is too short. Please describe your study situation in a full sentence."
        )

    # Obvious fast-path checks for basic single-word chit-chat
    lower = cleaned.lower()
    if lower in {"hello", "hi", "hey", "hola", "sup", "how are you", "good morning", "good evening"}:
        raise ExtractionError(
            "Hello! Please describe your study situation (e.g., 'My exam is in 5 days, "
            "I studied 2 hours today, and my preparation is poor')."
        )

    # Single LangChain call with Pydantic structured output
    llm = get_llm(temperature=0.0, max_tokens=250)
    structured_llm = llm.with_structured_output(ExtractedStudentData)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "{text}"),
    ])
    messages = prompt.format_messages(text=cleaned)

    try:
        raw: ExtractedStudentData = structured_llm.invoke(messages)
    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "rate_limit" in err_msg.lower():
            raise ExtractionError(
                "AI rate limit reached. Please wait a few seconds and try again."
            ) from exc
        raise ExtractionError(
            "I had trouble understanding your study details. Please mention your exam timeline, "
            "hours studied, and preparation level clearly."
        ) from exc

    if raw is None or not raw.is_study_related:
        reason = raw.rejection_reason if (raw and raw.rejection_reason) else None
        if reason:
            raise ExtractionError(
                f"{reason}. Please describe your study situation with days until exam and preparation level."
            )
        raise ExtractionError(
            "I couldn't find enough study-related information in that message. "
            "Please mention your exam timeline, study hours today, and preparation level."
        )

    # Ensure required parameters were provided
    if raw.days_until_exam is None or raw.preparation_level is None:
        raise ExtractionError(
            "Please include both how many days remain until your exam and how prepared you feel."
        )

    study_hours = raw.study_hours if raw.study_hours is not None else 0.0

    # Bounds clipping to guarantee valid inputs for the fuzzy engine
    safe_days = int(np.clip(raw.days_until_exam, 0, 30))
    safe_hours = float(np.clip(study_hours, 0.0, 12.0))
    safe_prep = float(np.clip(raw.preparation_level, 0.0, 100.0))

    return StudentStudyData(
        days_until_exam=safe_days,
        study_hours=safe_hours,
        preparation_level=safe_prep,
    )
