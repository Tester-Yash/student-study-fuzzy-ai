"""
fuzzy_system.py
================
A genuine Mamdani-style Fuzzy Inference System, implemented from scratch
using plain Python + NumPy (NO external fuzzy-logic library).

This module performs the five classic steps of fuzzy inference:
    1. Fuzzification            -> convert crisp inputs into membership degrees
    2. Rule Evaluation (AND)    -> minimum operator on rule antecedents
    3. Implication              -> clip each output membership by firing strength
    4. Aggregation (OR)         -> maximum operator across all rules per output
    5. Defuzzification          -> centroid (center of gravity) method

Inputs:
    - days_until_exam   (0-30 days)
    - study_hours_today (0-12 hours)
    - preparation_level (0-100 %)

Outputs:
    - recommended_study_time (0-8 hours/day)
    - study_need             (0-100, interpreted as Low/Medium/High)

Everything here is plain, readable Python so it can be explained line by
line in a viva.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. MEMBERSHIP FUNCTIONS
# ---------------------------------------------------------------------------
# We use simple triangular and trapezoidal membership functions.
# A triangular function rises from 0 to 1 and falls back to 0.
# A trapezoidal function rises from 0 to 1, stays flat, then falls back to 0.

def trimf(x, a, b, c):
    """Triangular membership function.
    a = left foot, b = peak, c = right foot.
    Value is 0 outside [a, c], rises linearly to 1 at b, then falls to c.
    """
    if b == a:
        left = 1.0 if x <= a else 0.0
    else:
        left = (x - a) / (b - a)

    if c == b:
        right = 1.0 if x >= c else 0.0
    else:
        right = (c - x) / (c - b)

    return max(min(left, right), 0.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function.
    a,b = left foot & shoulder ; c,d = right shoulder & foot.
    Flat top (value 1) between b and c.

    Special case: when a == b, there is no rising slope -- this is a
    "left shoulder" shape, so membership is already 1 for any x >= a.
    Similarly, when c == d, there is no falling slope -- this is a
    "right shoulder" shape, so membership stays 1 for any x <= d.
    """
    if b == a:
        left = 1.0 if x >= a else 0.0
    else:
        left = (x - a) / (b - a)

    if d == c:
        right = 1.0 if x <= d else 0.0
    else:
        right = (d - x) / (d - c)

    return max(min(left, 1.0, right), 0.0)


# ---------------------------------------------------------------------------
# 2. FUZZY VARIABLE DEFINITIONS
# ---------------------------------------------------------------------------
# Each input/output variable has a range and a set of named membership
# functions ("linguistic terms") defined as simple trapezoid/triangle shapes.
# These numbers were chosen to sensibly cover the whole input range.

# ---- INPUT 1: Days Until Exam (0-30 days) ----
def days_membership(x):
    x = np.clip(x, 0, 30)
    return {
        "Near":   trapmf(x, 0, 0, 2, 6),
        "Medium": trimf(x, 3, 10, 17),
        "Far":    trapmf(x, 12, 20, 30, 30),
    }


# ---- INPUT 2: Study Hours Today (0-12 hours) ----
def hours_membership(x):
    x = np.clip(x, 0, 12)
    return {
        "Low":    trapmf(x, 0, 0, 1, 3),
        "Medium": trimf(x, 2, 4.5, 7),
        "High":   trapmf(x, 5, 8, 12, 12),
    }


# ---- INPUT 3: Preparation Level (0-100 %) ----
def prep_membership(x):
    x = np.clip(x, 0, 100)
    return {
        "Poor":    trapmf(x, 0, 0, 20, 45),
        "Average": trimf(x, 30, 50, 70),
        "Good":    trapmf(x, 55, 80, 100, 100),
    }


# ---- OUTPUT 1: Recommended Study Time (0-8 hours/day) ----
def study_time_membership(y):
    y = np.clip(y, 0, 8)
    return {
        "Low":    trapmf(y, 0, 0, 1.5, 3),
        "Medium": trimf(y, 2, 4, 6),
        "High":   trapmf(y, 5, 6.5, 8, 8),
    }


# ---- OUTPUT 2: Study Need (0-100 scale) ----
def study_need_membership(y):
    y = np.clip(y, 0, 100)
    return {
        "Low":    trapmf(y, 0, 0, 20, 45),
        "Medium": trimf(y, 30, 50, 70),
        "High":   trapmf(y, 55, 80, 100, 100),
    }


# Universe of discourse (sample points) used for aggregation & centroid.
STUDY_TIME_UNIVERSE = np.linspace(0, 8, 161)     # step of 0.05 hours
STUDY_NEED_UNIVERSE = np.linspace(0, 100, 201)   # step of 0.5


# ---------------------------------------------------------------------------
# 3. FUZZY RULE BASE (18 logical rules covering the valid input space)
# ---------------------------------------------------------------------------
# Antecedents are combined with the MIN operator (fuzzy AND).
# Consequents are clipped and aggregated using the MAX operator.
# Meaningfully integrates all 3 inputs: days_until_exam, study_hours, preparation_level.
RULES = [
    # --- Recommended Study Time rules (13 rules) ---
    {"if": {"days": "Near", "hours": "Low", "prep": "Poor"},         "then": ("time", "High")},   # R1: Near + Low hours + Poor prep -> High study time (all 3 inputs)
    {"if": {"days": "Near", "prep": "Poor"},                         "then": ("time", "High")},   # R2: Near exam + Poor prep -> High study time
    {"if": {"days": "Near", "hours": "Low"},                         "then": ("time", "High")},   # R3: Near exam + Low hours -> High study time
    {"if": {"days": "Near", "prep": "Average"},                      "then": ("time", "High")},   # R4: Near exam + Average prep -> High study time
    {"if": {"days": "Near", "prep": "Good"},                         "then": ("time", "Medium")}, # R5: Near exam + Good prep -> Medium study time
    {"if": {"days": "Medium", "hours": "Medium", "prep": "Average"}, "then": ("time", "Medium")}, # R6: Medium exam + Med hours + Avg prep -> Medium study time (all 3 inputs)
    {"if": {"days": "Medium", "prep": "Poor"},                       "then": ("time", "High")},   # R7: Medium exam + Poor prep -> High study time
    {"if": {"days": "Medium", "prep": "Average"},                    "then": ("time", "Medium")}, # R8: Medium exam + Average prep -> Medium study time
    {"if": {"days": "Medium", "prep": "Good"},                       "then": ("time", "Low")},    # R9: Medium exam + Good prep -> Low study time
    {"if": {"days": "Medium", "hours": "Low"},                       "then": ("time", "Medium")}, # R10: Medium exam + Low hours -> Medium study time
    {"if": {"days": "Far", "prep": "Poor"},                          "then": ("time", "Medium")}, # R11: Far exam + Poor prep -> Medium study time
    {"if": {"days": "Far", "prep": "Average"},                       "then": ("time", "Low")},    # R12: Far exam + Average prep -> Low study time
    {"if": {"days": "Far", "hours": "High", "prep": "Good"},         "then": ("time", "Low")},    # R13: Far exam + High hours + Good prep -> Low study time (all 3 inputs)

    # --- Study Need rules (5 rules) ---
    {"if": {"days": "Near", "prep": "Poor"},                         "then": ("need", "High")},   # R14: Near exam + Poor prep -> High need
    {"if": {"days": "Near", "hours": "Low"},                         "then": ("need", "High")},   # R15: Near exam + Low hours -> High need
    {"if": {"days": "Medium", "prep": "Poor"},                       "then": ("need", "High")},   # R16: Medium exam + Poor prep -> High need
    {"if": {"days": "Medium", "prep": "Average"},                    "then": ("need", "Medium")}, # R17: Medium exam + Average prep -> Medium need
    {"if": {"days": "Far", "prep": "Good"},                          "then": ("need", "Low")},    # R18: Far exam + Good prep -> Low need
]


# ---------------------------------------------------------------------------
# 4. MAMDANI FUZZY INFERENCE ENGINE
# ---------------------------------------------------------------------------
class FuzzyStudySystem:
    """
    A genuine Mamdani fuzzy inference system:
      Fuzzification -> Rule Firing (min) -> Implication (clip) ->
      Aggregation (max) -> Defuzzification (centroid)
    """

    def run(self, days_until_exam: float, study_hours: float, preparation_level: float) -> dict:
        # ---------- STEP 1: FUZZIFICATION ----------
        # Convert each crisp input into membership degrees for every term.
        days_mf = days_membership(days_until_exam)
        hours_mf = hours_membership(study_hours)
        prep_mf = prep_membership(preparation_level)

        fuzzified_inputs = {
            "days_until_exam": days_mf,
            "study_hours": hours_mf,
            "preparation_level": prep_mf,
        }

        # ---------- STEP 2: RULE EVALUATION (fuzzy AND = min) ----------
        rule_results = []
        for idx, rule in enumerate(RULES, start=1):
            memberships = []
            readable_conditions = []
            for var, term in rule["if"].items():
                if var == "days":
                    val = days_mf[term]
                elif var == "hours":
                    val = hours_mf[term]
                elif var == "prep":
                    val = prep_mf[term]
                else:
                    val = 0.0
                memberships.append(val)
                readable_conditions.append(f"{var} is {term}")

            # Fuzzy AND -> minimum of all antecedent memberships
            firing_strength = min(memberships) if memberships else 0.0

            output_var, output_term = rule["then"]
            rule_results.append({
                "rule_id": f"R{idx}",
                "conditions": " AND ".join(readable_conditions),
                "output": f"{output_var} is {output_term}",
                "firing_strength": round(firing_strength, 4),
                "output_var": output_var,
                "output_term": output_term,
            })

        # ---------- STEP 3 & 4: IMPLICATION (clip) + AGGREGATION (max) ----------
        # For each output variable, build an aggregated fuzzy set by taking
        # the MAX across all rules that fire for that output, after clipping
        # each rule's output membership curve at its own firing strength.
        time_agg = np.zeros_like(STUDY_TIME_UNIVERSE)
        need_agg = np.zeros_like(STUDY_NEED_UNIVERSE)

        for r in rule_results:
            strength = r["firing_strength"]
            if strength <= 0:
                continue  # a rule that didn't fire contributes nothing

            if r["output_var"] == "time":
                term = r["output_term"]
                # membership curve for this output term, clipped at firing strength
                curve = np.array([
                    min(study_time_membership(y)[term], strength)
                    for y in STUDY_TIME_UNIVERSE
                ])
                time_agg = np.maximum(time_agg, curve)  # max aggregation

            elif r["output_var"] == "need":
                term = r["output_term"]
                curve = np.array([
                    min(study_need_membership(y)[term], strength)
                    for y in STUDY_NEED_UNIVERSE
                ])
                need_agg = np.maximum(need_agg, curve)

        # ---------- STEP 5: DEFUZZIFICATION (centroid / center of gravity) ----------
        recommended_time = self._centroid(STUDY_TIME_UNIVERSE, time_agg, default=2.0)
        study_need_score = self._centroid(STUDY_NEED_UNIVERSE, need_agg, default=30.0)

        # Clip to guarantee outputs always stay within valid bounds
        recommended_time = float(np.clip(recommended_time, 0, 8))
        study_need_score = float(np.clip(study_need_score, 0, 100))

        need_label = self._label_from_score(study_need_score)

        return {
            "fuzzified_inputs": fuzzified_inputs,
            "rules": rule_results,
            "aggregated_time_curve": {"x": STUDY_TIME_UNIVERSE.tolist(), "y": time_agg.tolist()},
            "aggregated_need_curve": {"x": STUDY_NEED_UNIVERSE.tolist(), "y": need_agg.tolist()},
            "recommended_study_time": round(recommended_time, 2),
            "study_need_score": round(study_need_score, 2),
            "study_need_label": need_label,
        }

    @staticmethod
    def _centroid(universe: np.ndarray, membership: np.ndarray, default: float) -> float:
        """
        Centroid (center of gravity) defuzzification:
            centroid = sum(x_i * mu(x_i)) / sum(mu(x_i))

        If no rule fired at all (sum of memberships is 0), we avoid a
        division-by-zero error and fall back to a safe default value.
        """
        total_membership = np.sum(membership)
        if total_membership <= 1e-9:
            return default
        return float(np.sum(universe * membership) / total_membership)

    @staticmethod
    def _label_from_score(score: float) -> str:
        """Convert the numeric study_need score (0-100) into a simple label."""
        if score < 35:
            return "Low"
        elif score < 65:
            return "Medium"
        else:
            return "High"


if __name__ == "__main__":
    # Small manual demo when running this file directly
    system = FuzzyStudySystem()
    result = system.run(days_until_exam=5, study_hours=2, preparation_level=20)
    print("Recommended study time:", result["recommended_study_time"], "hours/day")
    print("Study need:", result["study_need_label"], f"({result['study_need_score']})")
