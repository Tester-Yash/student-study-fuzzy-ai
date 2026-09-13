"""
tests/test_fuzzy.py
====================
Unit tests for the fuzzy logic engine (fuzzy_system.py) and input validation
behavior of extractor.py.

These tests do NOT require a live API key: they thoroughly test the Mamdani
fuzzy mathematics, rule firing, defuzzification, input coverage, and extractor
validation without relying on external network calls.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Allow running tests from the project root or from inside tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fuzzy_system import (
    FuzzyStudySystem,
    days_membership,
    hours_membership,
    prep_membership,
    study_time_membership,
    study_need_membership,
)
from extractor import extract_student_data, ExtractionError
from models import ExtractedStudentData, StudentStudyData


@pytest.fixture
def system():
    return FuzzyStudySystem()


# ---------------------------------------------------------------------
# 1. Normal student input
# ---------------------------------------------------------------------
def test_normal_student_input(system):
    result = system.run(days_until_exam=5, study_hours=2.0, preparation_level=30.0)
    assert 0.0 <= result["recommended_study_time"] <= 8.0
    assert result["study_need_label"] in ("Low", "Medium", "High")
    assert 0.0 <= result["study_need_score"] <= 100.0


# ---------------------------------------------------------------------
# 2. Key logical rule cases (as required by specification)
# ---------------------------------------------------------------------
def test_exam_near_poor_preparation(system):
    """Near exam AND Poor preparation -> High study time & High need."""
    result = system.run(days_until_exam=1, study_hours=1.0, preparation_level=10.0)
    assert result["recommended_study_time"] >= 5.0, "Near exam + poor prep must recommend high study time"
    assert result["study_need_label"] == "High"


def test_exam_near_low_study_hours(system):
    """Near exam AND Low study hours -> High study time."""
    result = system.run(days_until_exam=2, study_hours=0.5, preparation_level=40.0)
    assert result["recommended_study_time"] >= 4.5


def test_exam_far_good_preparation(system):
    """Far exam AND Good preparation -> Low study time & Low need."""
    result = system.run(days_until_exam=28, study_hours=6.0, preparation_level=90.0)
    assert result["recommended_study_time"] <= 3.2, "Far exam + good prep must recommend low study time"
    assert result["study_need_label"] == "Low"


def test_exam_medium_average_preparation(system):
    """Medium exam distance AND Average preparation -> Medium study time."""
    result = system.run(days_until_exam=10, study_hours=3.0, preparation_level=50.0)
    assert 3.0 <= result["recommended_study_time"] <= 5.5
    assert result["study_need_label"] in ("Medium", "Low")


def test_high_study_hours_good_preparation(system):
    """High study hours today AND Good preparation -> Low study time."""
    result = system.run(days_until_exam=15, study_hours=8.0, preparation_level=85.0)
    assert result["recommended_study_time"] <= 3.5


# ---------------------------------------------------------------------
# 3. Input variation coverage (Low vs High study hours)
# ---------------------------------------------------------------------
def test_low_study_hours(system):
    result = system.run(days_until_exam=6, study_hours=0.0, preparation_level=40.0)
    assert 0.0 <= result["recommended_study_time"] <= 8.0


def test_high_study_hours(system):
    result = system.run(days_until_exam=6, study_hours=10.0, preparation_level=70.0)
    assert 0.0 <= result["recommended_study_time"] <= 8.0


# ---------------------------------------------------------------------
# 4. Boundary values (extremes 0 and maximum for all variables)
# ---------------------------------------------------------------------
def test_boundary_values_minimum(system):
    result = system.run(days_until_exam=0, study_hours=0.0, preparation_level=0.0)
    assert 0.0 <= result["recommended_study_time"] <= 8.0
    assert 0.0 <= result["study_need_score"] <= 100.0


def test_boundary_values_maximum(system):
    result = system.run(days_until_exam=30, study_hours=12.0, preparation_level=100.0)
    assert 0.0 <= result["recommended_study_time"] <= 8.0
    assert 0.0 <= result["study_need_score"] <= 100.0


# ---------------------------------------------------------------------
# 5. Defuzzification and mathematical properties
# ---------------------------------------------------------------------
def test_defuzzification_produces_float_and_curves(system):
    result = system.run(days_until_exam=10, study_hours=3.0, preparation_level=50.0)
    assert isinstance(result["recommended_study_time"], float)
    assert isinstance(result["study_need_score"], float)
    assert "aggregated_time_curve" in result
    assert "aggregated_need_curve" in result
    assert len(result["aggregated_time_curve"]["x"]) == len(result["aggregated_time_curve"]["y"])


def test_centroid_no_division_by_zero(system):
    """Centroid computation must safely fall back when all memberships are zero."""
    zero_membership = np.zeros(50)
    universe = np.linspace(0, 8, 50)
    value = system._centroid(universe, zero_membership, default=2.5)
    assert value == 2.5


def test_rules_coverage_no_orphan_inputs(system):
    """
    Verify that across the entire input space, the improved rule base
    fires at least one rule with firing_strength > 0 (no default fallback needed).
    """
    for d in [0, 2, 5, 10, 15, 20, 25, 30]:
        for h in [0.0, 1.0, 3.0, 6.0, 9.0, 12.0]:
            for p in [0.0, 20.0, 50.0, 75.0, 100.0]:
                res = system.run(days_until_exam=d, study_hours=h, preparation_level=p)
                active_rules = [r for r in res["rules"] if r["firing_strength"] > 0]
                assert len(active_rules) > 0, f"No rule fired for inputs d={d}, h={h}, p={p}"


# ---------------------------------------------------------------------
# 6. Parameterized valid range checks
# ---------------------------------------------------------------------
@pytest.mark.parametrize("days,hours,prep", [
    (0, 0.0, 0.0),
    (30, 12.0, 100.0),
    (5, 2.0, 20.0),
    (15, 6.0, 50.0),
    (3, 0.0, 5.0),
    (29, 11.0, 99.0),
    (10, 4.0, 60.0),
    (2, 1.0, 15.0),
    (7, 3.5, 45.0),
    (21, 5.0, 80.0),
])
def test_output_within_valid_range(system, days, hours, prep):
    result = system.run(days_until_exam=days, study_hours=hours, preparation_level=prep)
    assert 0.0 <= result["recommended_study_time"] <= 8.0
    assert 0.0 <= result["study_need_score"] <= 100.0


# ---------------------------------------------------------------------
# 7. Extractor validation and error handling (without calling real LLM)
# ---------------------------------------------------------------------
def test_extractor_rejects_empty_input():
    with pytest.raises(ExtractionError):
        extract_student_data("")


def test_extractor_rejects_whitespace_input():
    with pytest.raises(ExtractionError):
        extract_student_data("     ")


def test_extractor_rejects_too_short_input():
    with pytest.raises(ExtractionError):
        extract_student_data("hi")


def test_extractor_rejects_casual_greetings():
    for greeting in ["hello", "Hello", "hi", "hey", "how are you"]:
        with pytest.raises(ExtractionError):
            extract_student_data(greeting)


@patch("extractor.get_llm")
def test_extractor_mock_valid_input(mock_get_llm):
    """Test full extraction flow with a mocked LLM structured output."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ExtractedStudentData(
        is_study_related=True,
        days_until_exam=5,
        study_hours=2.0,
        preparation_level=20.0,
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    result = extract_student_data("My exam is in 5 days, studied 2 hours, preparation is poor")
    assert isinstance(result, StudentStudyData)
    assert result.days_until_exam == 5
    assert result.study_hours == 2.0
    assert result.preparation_level == 20.0


@patch("extractor.get_llm")
def test_extractor_mock_irrelevant_rejection(mock_get_llm):
    """Test structured rejection of irrelevant input using mocked LLM."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ExtractedStudentData(
        is_study_related=False,
        rejection_reason="Unrelated recipe question",
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    with pytest.raises(ExtractionError) as exc_info:
        extract_student_data("What is the recipe for chocolate cake?")
    assert "recipe" in str(exc_info.value).lower() or "study" in str(exc_info.value).lower()
