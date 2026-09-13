"""
visualization.py
=================
Helper functions that build clean, publication-quality Matplotlib charts
for the Streamlit UI:
    - Input membership functions with crisp input marked
    - Aggregated output fuzzy sets with reference curves and centroid marked
"""

import numpy as np
import matplotlib.pyplot as plt

from fuzzy_system import (
    days_membership, hours_membership, prep_membership,
    study_time_membership, study_need_membership,
)


def set_plot_styling(fig, ax):
    """Apply clean, modern styling to matplotlib axes."""
    ax.tick_params(colors="#333333", labelsize=8)
    ax.spines["top"].set_color("#dddddd")
    ax.spines["right"].set_color("#dddddd")
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(True, linestyle=":", color="#e0e0e0", alpha=0.7)


def plot_input_membership(variable_name: str, crisp_value: float, x_range, membership_fn, unit=""):
    """Plot the membership curves for an input variable, marking the crisp value."""
    xs = np.linspace(x_range[0], x_range[1], 200)
    terms = list(membership_fn(xs[0]).keys())

    fig, ax = plt.subplots(figsize=(4.2, 2.7), dpi=100)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, term in enumerate(terms):
        ys = [membership_fn(x)[term] for x in xs]
        ax.plot(xs, ys, label=term, color=colors[i % len(colors)], linewidth=1.5)

    ax.axvline(crisp_value, color="#222222", linestyle="--", linewidth=1.4,
               label=f"Input = {crisp_value}{unit}")
    ax.set_title(variable_name, fontsize=9.5, fontweight="bold", pad=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylabel("Membership degree", fontsize=8)
    set_plot_styling(fig, ax)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_days_membership(value):
    return plot_input_membership("Days Until Exam (days)", value, (0, 30), days_membership, unit=" d")


def plot_hours_membership(value):
    return plot_input_membership("Study Hours Today (hrs)", value, (0, 12), hours_membership, unit=" h")


def plot_prep_membership(value):
    return plot_input_membership("Preparation Level (%)", value, (0, 100), prep_membership, unit="%")


def plot_aggregated_output(title, x_vals, y_vals, centroid_value, x_label, reference_fn=None):
    """Plot the aggregated fuzzy output set with reference terms and centroid marked."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=100)

    if reference_fn:
        sample_terms = list(reference_fn(x_vals[0]).keys())
        ref_styles = {
            "Low": (":", "#1f77b4"),
            "Medium": (":", "#ff7f0e"),
            "High": (":", "#2ca02c"),
        }
        for term in sample_terms:
            linestyle, color = ref_styles.get(term, (":", "#888888"))
            term_ys = [reference_fn(x)[term] for x in x_vals]
            ax.plot(x_vals, term_ys, linestyle=linestyle, color=color, alpha=0.8,
                    linewidth=1.2, label=f"{term} (reference)")

    ax.fill_between(x_vals, y_vals, color="#6baed6", alpha=0.55, label="Aggregated output")
    ax.plot(x_vals, y_vals, color="#2171b5", linewidth=1.5)
    ax.axvline(centroid_value, color="#d95f02", linestyle="--", linewidth=1.8,
               label=f"Centroid = {centroid_value:.2f}")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_xlabel(x_label, fontsize=8.5)
    ax.set_ylabel("Membership degree", fontsize=8.5)
    set_plot_styling(fig, ax)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig
