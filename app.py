"""
app.py
======
Clean, simple, and intuitive Streamlit UI for the AI-Based Study Time Recommendation System.
Combines LangChain (natural language understanding) + Mamdani Fuzzy Logic.
Includes per-browser Settings for user API keys.
"""

import html
import os
import pandas as pd
import streamlit as st

from extractor import extract_student_data, ExtractionError
from explainer import explain_recommendation, ExplanationError
from fuzzy_system import FuzzyStudySystem, study_time_membership, study_need_membership
from llm_config import MissingAPIKeyError
from visualization import (
    plot_days_membership, plot_hours_membership, plot_prep_membership,
    plot_aggregated_output,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Study Time Recommendation System",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Minimal, clean CSS styling
st.markdown("""
<style>
    .input-quote {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        color: #93c5fd;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .status-pill {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-green {
        background-color: #14532d;
        color: #86efac;
    }
    .status-amber {
        background-color: #78350f;
        color: #fde68a;
    }
    .status-red {
        background-color: #7f1d1d;
        color: #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Browser-Specific Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ API Settings")
    st.caption("Your API key is kept in this browser session and is not saved to the project files or displayed to other users.")

    provider_choice = st.selectbox(
        "LLM Provider",
        options=["Groq (Free & Fast)", "OpenAI"],
        index=0,
    )
    provider_code = "groq" if "Groq" in provider_choice else "openai"
    st.session_state["user_provider"] = provider_code

    if provider_code == "groq":
        # Check if user already entered a key or if an env fallback exists
        current_val = st.session_state.get("user_groq_key", "")
        if not current_val and os.getenv("GROQ_API_KEY"):
            # Optional local default if on user's own machine
            current_val = os.getenv("GROQ_API_KEY")

        groq_key_input = st.text_input(
            "Groq API Key",
            value=current_val,
            type="password",
            placeholder="gsk_...",
            help="Your API key is kept in this browser session and is not saved to the project files or displayed to other users.",
        )
        st.session_state["user_groq_key"] = groq_key_input.strip() if groq_key_input else ""

        if st.session_state["user_groq_key"]:
            st.success("✅ Groq key active in this browser")
        else:
            st.info("💡 Get a free key at [console.groq.com/keys](https://console.groq.com/keys)")

    else:
        current_val = st.session_state.get("user_openai_key", "")
        if not current_val and os.getenv("OPENAI_API_KEY"):
            current_val = os.getenv("OPENAI_API_KEY")

        openai_key_input = st.text_input(
            "OpenAI API Key",
            value=current_val,
            type="password",
            placeholder="sk-...",
            help="Your API key is kept in this browser session and is not saved to the project files or displayed to other users.",
        )
        st.session_state["user_openai_key"] = openai_key_input.strip() if openai_key_input else ""

        if st.session_state["user_openai_key"]:
            st.success("✅ OpenAI key active in this browser")
        else:
            st.info("💡 Enter your OpenAI key to enable GPT models.")

    st.divider()
    st.caption("🔒 **Session Privacy**: When you close or refresh this tab, the browser-session key is cleared from memory.")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("🎓 AI-Based Study Time Recommendation System")
st.caption("A mini project combining Fuzzy Logic with LangChain")

with st.expander("ℹ️ How this works", expanded=False):
    st.write(
        "1. **Describe your situation**: Type how many days are left, how much you studied today, and how prepared you feel.\n"
        "2. **AI Extraction**: LangChain reads your text and extracts the numbers.\n"
        "3. **Fuzzy Logic**: A Mamdani inference system computes the recommended study hours and urgency score.\n"
        "4. **AI Coach Tips**: The model generates friendly, personalized study advice."
    )

# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------
st.subheader("Describe your current study situation:")
default_example = (
    "My exam is in 5 days. I studied 2 hours today and my preparation is poor. "
    "I still need some revision before the exam."
)

user_text = st.text_area(
    label="Your study situation:",
    value=default_example,
    height=100,
    label_visibility="collapsed",
)

analyze_clicked = st.button("Analyze Study Schedule", type="primary")

# -----------------------------------------------------------------------------
# Results (Only shown after clicking Analyze)
# -----------------------------------------------------------------------------
if analyze_clicked:
    if not user_text or not user_text.strip():
        st.warning("Please enter your study situation above before analyzing.")
        st.stop()

    # 1. Your Input
    st.header("1. Your Input")
    escaped_input = html.escape(user_text.strip())
    st.markdown(f'<div class="input-quote">&ldquo;{escaped_input}&rdquo;</div>', unsafe_allow_html=True)

    # 2. AI Extraction
    with st.spinner("Extracting information with AI..."):
        try:
            student_data = extract_student_data(user_text)
        except MissingAPIKeyError as e:
            st.error(f"Configuration error: {e}")
            st.info("👈 Please enter your API key in the **⚙️ Settings** sidebar on the left.")
            st.stop()
        except ExtractionError as e:
            st.warning(f"{e}")
            st.stop()
        except Exception as e:
            st.error("Could not contact the AI service. Please check your API key and connection.")
            st.caption(f"Error: {type(e).__name__}")
            st.stop()

    st.header("2. AI-Extracted Study Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Days Until Exam", f"{student_data.days_until_exam} days")
    col2.metric("Study Hours Today", f"{student_data.study_hours:.1f} hrs")
    col3.metric("Preparation Level", f"{student_data.preparation_level:.0f}%")

    # 3. Fuzzy Logic Inference
    fuzzy_system = FuzzyStudySystem()
    result = fuzzy_system.run(
        days_until_exam=student_data.days_until_exam,
        study_hours=student_data.study_hours,
        preparation_level=student_data.preparation_level,
    )

    st.header("3. Fuzzy Logic Analysis")
    st.markdown("**Fuzzification — membership degrees**")

    # Clean table
    fz = result["fuzzified_inputs"]
    membership_table = pd.DataFrame([
        {
            "Input Variable": "Days Until Exam",
            "Value": f"{student_data.days_until_exam} days",
            "Near": f"{fz['days_until_exam']['Near']:.2f}",
            "Medium": f"{fz['days_until_exam']['Medium']:.2f}",
            "Far": f"{fz['days_until_exam']['Far']:.2f}",
            "Low": "—",
            "High": "—",
            "Poor": "—",
            "Average": "—",
            "Good": "—",
        },
        {
            "Input Variable": "Study Hours Today",
            "Value": f"{student_data.study_hours:.1f} hrs",
            "Near": "—",
            "Medium": f"{fz['study_hours']['Medium']:.2f}",
            "Far": "—",
            "Low": f"{fz['study_hours']['Low']:.2f}",
            "High": f"{fz['study_hours']['High']:.2f}",
            "Poor": "—",
            "Average": "—",
            "Good": "—",
        },
        {
            "Input Variable": "Preparation Level",
            "Value": f"{student_data.preparation_level:.0f}%",
            "Near": "—",
            "Medium": "—",
            "Far": "—",
            "Low": "—",
            "High": "—",
            "Poor": f"{fz['preparation_level']['Poor']:.2f}",
            "Average": f"{fz['preparation_level']['Average']:.2f}",
            "Good": f"{fz['preparation_level']['Good']:.2f}",
        },
    ])
    st.dataframe(membership_table, hide_index=True)

    # 3 Membership Plots
    c1, c2, c3 = st.columns(3)
    c1.pyplot(plot_days_membership(student_data.days_until_exam))
    c2.pyplot(plot_hours_membership(student_data.study_hours))
    c3.pyplot(plot_prep_membership(student_data.preparation_level))

    # All Rules in a simple expander
    active_count = len([r for r in result["rules"] if r["firing_strength"] > 0])
    with st.expander(f"See every fuzzy rule and its firing strength ({active_count} firing)"):
        rules_df = pd.DataFrame([
            {
                "Rule": r["rule_id"],
                "IF": r["conditions"],
                "THEN": r["output"],
                "Firing Strength": f"{r['firing_strength']:.3f}",
            }
            for r in result["rules"]
        ])
        st.dataframe(rules_df, hide_index=True)

    # Defuzzification
    st.markdown("**Aggregation & Defuzzification (centroid method)**")
    p1, p2 = st.columns(2)
    p1.pyplot(plot_aggregated_output(
        "Output: Recommended Study Time",
        result["aggregated_time_curve"]["x"],
        result["aggregated_time_curve"]["y"],
        result["recommended_study_time"],
        "Hours / day",
        reference_fn=study_time_membership,
    ))
    p2.pyplot(plot_aggregated_output(
        "Output: Study Need Score",
        result["aggregated_need_curve"]["x"],
        result["aggregated_need_curve"]["y"],
        result["study_need_score"],
        "Study Need (0–100 scale)",
        reference_fn=study_need_membership,
    ))

    # 4. Final Result
    st.header("4. Final Result")
    res_col1, res_col2 = st.columns([1.5, 1])
    with res_col1:
        st.caption("Recommended Daily Study Time")
        st.subheader(f"{result['recommended_study_time']} hrs / day")
        st.caption(f"Study Need Score: {result['study_need_score']:.1f} / 100")
    with res_col2:
        need_label = result["study_need_label"]
        if need_label == "Low":
            pill_class = "status-green"
            pill_text = "Low Need — On Track"
        elif need_label == "Medium":
            pill_class = "status-amber"
            pill_text = "Moderate Need — Steady Pace"
        else:
            pill_class = "status-red"
            pill_text = "High Need — Urgent Action"
        st.markdown(f'<div style="margin-top: 20px;"><span class="status-pill {pill_class}">{pill_text}</span></div>', unsafe_allow_html=True)

    # 5. AI Explanation & Study Recommendations
    st.header("5. AI Explanation & Study Recommendations")
    with st.spinner("Writing personalized study tips..."):
        try:
            explanation = explain_recommendation(
                days_until_exam=student_data.days_until_exam,
                study_hours=student_data.study_hours,
                preparation_level=student_data.preparation_level,
                recommended_study_time=result["recommended_study_time"],
                study_need_label=result["study_need_label"],
            )

            st.markdown("#### What this means")
            st.write(explanation.explanation)

            st.markdown("#### Study Recommendations")
            for tip in explanation.tips:
                st.markdown(f"- {tip}")

        except MissingAPIKeyError as e:
            st.error(f"Configuration error: {e}")
            st.info("👈 Please enter your API key in the **⚙️ Settings** sidebar on the left.")
        except ExplanationError as e:
            st.warning(f"{e}")
        except Exception as e:
            st.error("Could not generate AI explanation right now.")

# Footer
st.divider()
st.caption(
    "Mini project demo — Fuzzy Logic (membership functions, fuzzification, rule evaluation, defuzzification) "
    "+ LangChain for natural-language understanding and explanation."
)
