"""
explainer.py
============
SECOND LANGCHAIN CHAIN:
    Fuzzy result + extracted student information -> LangChain prompt
    -> LLM -> simple explanation + structured, actionable study tips

Turns numeric fuzzy outputs into friendly, human-readable explanations
and categorized study tips with bold key action labels.
"""

from langchain_core.prompts import ChatPromptTemplate
from models import ExplanationResult
from llm_config import get_llm


EXPLANATION_SYSTEM_PROMPT = """You are an experienced, encouraging academic coach helping a student.
You will be provided:
  - The student's current exam situation (days until exam, hours studied today, preparation level)
  - The recommended daily study time (in hours) calculated by a Mamdani fuzzy logic engine
  - The calculated "study need" level (Low / Medium / High)

Generate a response adhering to this format:
1. `explanation`: A concise, supportive paragraph (2-4 sentences) explaining what this fuzzy-logic recommendation means in everyday terms and why it fits their current situation.
2. `tips`: Exactly 4 to 5 high-impact, practical study recommendations. Each tip MUST begin with a 2 to 3 word bold category title followed by a colon and concrete advice.
   Examples of formats to follow:
   - "**Targeted review:** Focus on the top 2-3 topics that carry the highest exam weight..."
   - "**Practice with purpose:** Take a timed mock test under real exam conditions..."
   - "**Spaced repetition:** Review challenging flashcard concepts at expanding intervals..."
   - "**Active recall:** Solve practice questions from memory before checking notes..."

Keep the tone constructive, clear, and motivating. Respond strictly using the structured schema.
"""


class ExplanationError(Exception):
    """Raised if the explanation chain fails (e.g. network/LLM failure)."""
    pass


def explain_recommendation(
    days_until_exam: int,
    study_hours: float,
    preparation_level: float,
    recommended_study_time: float,
    study_need_label: str,
) -> ExplanationResult:
    """
    Run the second LangChain chain to turn the fuzzy system's numeric
    output into a friendly explanation + structured tips.
    """
    llm = get_llm(temperature=0.3, max_tokens=600)
    structured_llm = llm.with_structured_output(ExplanationResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLANATION_SYSTEM_PROMPT),
        ("human",
         "Student situation:\n"
         "- Days until exam: {days}\n"
         "- Hours studied today: {hours}\n"
         "- Preparation level: {prep}%\n\n"
         "Fuzzy system output:\n"
         "- Recommended daily study time: {rec_time} hours/day\n"
         "- Study need level: {need_label}\n"),
    ])
    messages = prompt.format_messages(
        days=days_until_exam,
        hours=study_hours,
        prep=preparation_level,
        rec_time=recommended_study_time,
        need_label=study_need_label,
    )

    try:
        result: ExplanationResult = structured_llm.invoke(messages)
    except Exception as exc:
        raise ExplanationError(
            "I couldn't generate an explanation right now (the AI service "
            "may be unavailable). Here is the recommendation based on the fuzzy logic calculation."
        ) from exc

    if result is None or not result.tips:
        raise ExplanationError(
            "I couldn't generate a full explanation right now. Please try again."
        )

    return result
