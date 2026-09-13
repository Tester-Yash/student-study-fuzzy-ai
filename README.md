# 🎓 AI-Based Study Time Recommendation System Using Fuzzy Logic

A beginner-friendly college mini-project that combines **Fuzzy Logic**
(for reasoning under uncertainty) with **LangChain + an LLM**
(for understanding natural language and explaining results).

---

## 1. Problem Statement

Students often struggle to decide how much time they should spend
studying today. The right amount depends on multiple, imprecise
factors at once — how many days are left until the exam, how much they
already studied, and how prepared they feel. These factors are
**fuzzy** (not crisp): "a few days," "poor preparation," and "studied
a little" don't have exact numeric boundaries.

## 2. Objective

To build a system where:
1. A student describes their situation in plain English.
2. An **LLM (via LangChain)** reads that sentence and extracts three
   numeric facts.
3. A **fuzzy logic engine** (built from scratch, Mamdani-style)
   reasons about those numbers the way a human expert would, and
   recommends a daily study time and a study-need level.
4. The **LLM** explains the result and gives personalised study tips.

## 3. Technologies Used

| Component            | Technology                          |
|-----------------------|--------------------------------------|
| Web UI                | Streamlit                            |
| Language understanding | LangChain + Groq LLM (or OpenAI)   |
| Structured data        | Pydantic                            |
| Fuzzy logic engine     | Pure Python + NumPy (no fuzzy library) |
| Charts                 | Matplotlib                          |
| Testing                | Pytest                              |

---

## 4. How LangChain Is Used

LangChain is used for **two real chains**, each doing genuine language
work — not just a "for-show" API call.

### Chain 1 — Extraction (`extractor.py`)
```
Natural-language student input
        ↓
ONE LangChain structured-output LLM call
        ↓
ExtractedStudentData (Pydantic)
        ↓
validation/conversion to StudentStudyData
        ↓
Mamdani fuzzy system
```
The LLM reads a natural-language description such as:

> "My exam is in 5 days. I studied 2 hours today and my preparation is poor. I still need some revision before the exam."

...and in **ONE single structured LLM call** (`with_structured_output(ExtractedStudentData)`), extracts structured information into our `ExtractedStudentData` Pydantic model:
- `is_study_related` (bool): Structured boolean indicating whether the input genuinely describes an exam/study situation (`False` for greetings, chit-chat, or off-topic questions).
- `days_until_exam` (int, 0–30): Quantitative days remaining until the exam.
- `study_hours` (float, 0–12): Hours already studied today (defaults to 0.0 if not specified).
- `preparation_level` (float, 0–100): Self-rated preparation level as a percentage (e.g. poor ≈ 20%, average ≈ 50%, good ≈ 80%).
- `rejection_reason` (optional str): Contextual explanation when input is not study-related.

Once validated, `extractor.py` converts the data into `StudentStudyData`, which enforces strict range bounds (`0 <= days <= 30`, `0 <= hours <= 12`, `0 <= prep <= 100`) before passing it directly to the Mamdani fuzzy logic engine.

No regex or keyword-based parsing is used — the model uses genuine natural language understanding with `.with_structured_output(ExtractedStudentData)`.

The chain also detects irrelevant or casual input (e.g. "hello", off-topic questions) using structured classification (`is_study_related = False`) and prompts the user to describe their study situation instead of guessing meaningless numbers.

### Chain 2 — Explanation (`explainer.py`)
```
Fuzzy system's numeric output -> LangChain prompt -> LLM
                                -> explanation + 3-5 study tips
```
The fuzzy engine's numeric recommendation (e.g. "5.15 hours/day, High need") is computed mathematically by our Mamdani engine. The second chain asks the LLM via `with_structured_output(ExplanationResult)` to turn that into a short, friendly explanation and 3 to 5 concrete, actionable study tips — genuine natural language generation grounded in the numbers our fuzzy logic computed.

---

## 5. How Fuzzy Logic Is Used

The fuzzy logic engine (`fuzzy_system.py`) is a genuine **Mamdani Fuzzy
Inference System** implemented from scratch in Python/NumPy (no
`scikit-fuzzy` or similar library), so every step can be explained
line-by-line.

### Inputs
| Input                | Range     | Terms                 |
|-----------------------|-----------|------------------------|
| Days Until Exam       | 0–30 days | Near, Medium, Far     |
| Study Hours Today     | 0–12 hrs  | Low, Medium, High     |
| Preparation Level     | 0–100 %   | Poor, Average, Good   |

### Outputs
| Output                    | Range        | Terms               |
|-----------------------------|--------------|----------------------|
| Recommended Study Time      | 0–8 hrs/day  | Low, Medium, High   |
| Study Need                  | 0–100 score  | Low, Medium, High   |

### Membership Functions
Simple **triangular** and **trapezoidal** shapes are used (see
`trimf()` and `trapmf()` in `fuzzy_system.py`). For example, "Near"
(days until exam) is a trapezoid that is fully true (1.0) from 0 to 2
days, then fades out to 0 by day 6.

### Fuzzification
Each crisp input value (e.g. `days_until_exam = 5`) is converted into
membership degrees for every term, e.g. `Near = 0.25, Medium = 0.29,
Far = 0.0`. This is done by simply evaluating each term's membership
function at that input value.

### Fuzzy Rules (18 rules)
Rules combine input terms using the **minimum** operator for fuzzy
AND. Example:
```
R1: IF days_until_exam is Near AND preparation is Poor
    THEN recommended_study_time is High
```
The full rule base (in `RULES` inside `fuzzy_system.py`) covers the
input space sensibly — near exam + poor prep pushes study time up;
far exam + good prep pulls it down; medium situations land in between.

### Rule Evaluation & Aggregation
- **Firing strength** of each rule = `min()` of its antecedent
  membership degrees (fuzzy AND).
- **Implication**: each rule's output membership curve is clipped at
  its own firing strength.
- **Aggregation**: all rules pointing to the same output are combined
  using the **maximum** operator, producing one aggregated fuzzy set
  per output.

### Defuzzification — Centroid Method
The aggregated fuzzy set is converted back into one crisp number using
the **centroid (center of gravity)** formula:

```
centroid = Σ(x_i * μ(x_i)) / Σ(μ(x_i))
```

Centroid is used (instead of, say, "mean of maximum") because it
considers the *whole shape* of the aggregated fuzzy set, giving a
smooth, balanced result that doesn't jump abruptly as inputs change
slightly — which matches how a human would weigh several rules of
advice together.

If no rule fires at all (an edge case), the code safely falls back to
a default value instead of dividing by zero.

### Why Not Plain if/else?
`if/else` logic requires hard boundaries ("if days < 3, near exam").
A student with `days = 2.9` and one with `days = 3.1` would get
completely different, contradictory advice, even though their
situations are nearly identical. Fuzzy logic allows **partial
membership** in multiple categories at once (e.g. "70% Near, 30%
Medium"), so the output changes smoothly — just like real human
judgment.

---

## 6. System Architecture

```
                ┌─────────────────────────┐
   User input   │      Streamlit UI       │
  ───────────►  │        (app.py)         │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   Chain 1: Extractor    │   ONE LangChain call
                │     (extractor.py)      │   -> ExtractedStudentData (Pydantic)
                └────────────┬────────────┘
                             │ validation & bounds clipping
                             ▼
                ┌─────────────────────────┐
                │    StudentStudyData     │   Validated schema
                │       (models.py)       │   (days, hours, prep)
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │      Fuzzy Engine       │   Mamdani Inference
                │    (fuzzy_system.py)    │   (fuzzification -> rules ->
                └────────────┬────────────┘    aggregation -> centroid)
                             │ (study_time, study_need)
                             ▼
                ┌─────────────────────────┐
                │   Chain 2: Explainer    │   LangChain + LLM
                │     (explainer.py)      │   -> ExplanationResult (tips)
                └────────────┬────────────┘
                             │
                             ▼
                  Results shown in Streamlit UI
```

### 🔒 API Key Privacy
- In the web app, your API key is kept in this browser session and is not saved to the project files or displayed to other users.
- When you close or refresh this tab, the browser-session key is cleared from memory.
- For local development, keys are placed in a `.env` file (which is git-ignored and never committed).
- For cloud deployment, keys are configured via secure Streamlit Cloud Secrets.

---

## 7. Project Structure

```
student-study-fuzzy-ai/
│
├── app.py                # Streamlit UI (main entry point)
├── fuzzy_system.py        # Mamdani fuzzy inference engine (from scratch)
├── extractor.py           # LangChain chain #1: text -> structured data
├── explainer.py           # LangChain chain #2: numbers -> explanation
├── models.py               # Pydantic data models
├── llm_config.py            # Groq/OpenAI LLM setup (reads API keys safely)
├── visualization.py          # Matplotlib charts for the UI
├── requirements.txt
├── .env.example
├── .gitignore
│
├── docs/
│   └── viva_notes.md       # Simple Q&A for viva preparation
│
└── tests/
    └── test_fuzzy.py       # Automated tests
```

---

## 8. How to Run Locally

1. **Clone/download this project** and open a terminal inside the folder.

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste your Groq API key:
   ```
   GROQ_API_KEY=your_real_key_here
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 9. How to Get a Free Groq API Key

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up / log in (free).
3. Create a new API key.
4. Paste it into your `.env` file as `GROQ_API_KEY=...`

**Never commit your real `.env` file or share your API key publicly.**

---

## 10. How to Deploy on Streamlit Community Cloud

1. Push this project to a **public or private GitHub repository**
   (make sure `.env` is **not** included — it's already in
   `.gitignore`).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and
   sign in with GitHub.
3. Click **"New app"**, select your repository, branch, and set the
   main file path to `app.py`.
4. Before deploying, open **"Advanced settings" → "Secrets"** and add:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your_real_key_here"
   GROQ_MODEL = "qwen/qwen3.8-27b"
   ```
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   and launch `app.py` automatically.

The app reads secrets from `st.secrets` on Streamlit Cloud and from
`.env` locally — no code changes are needed between environments.

---

## 11. Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers normal inputs, edge cases (exam very near /
very far, low/high study hours, boundary values 0 and max),
defuzzification correctness, division-by-zero safety, invalid/casual
input handling, and output-range guarantees.

---

## 12. Viva Preparation

See [`docs/viva_notes.md`](docs/viva_notes.md) for short, beginner-friendly
answers to common viva questions about this project.
