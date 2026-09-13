# Viva Notes — AI-Based Study Time Recommendation System Using Fuzzy Logic

Short, simple answers you can memorize and explain confidently.

---

**1. What is your project?**
It's a web app where a student types a sentence about their exam
situation (days left, hours studied, preparation level), and the
system recommends how many hours they should study today — using
fuzzy logic — and explains it using AI.

**2. Why did you choose this topic?**
Because deciding "how much to study" is not a black-and-white
decision — it depends on several vague factors together, which makes
it a perfect real-world example for fuzzy logic, and combining it with
an LLM makes the input/output natural and easy to use.

**3. Where is AI used?**
AI (an LLM through LangChain) is used twice: first, to read the
student's sentence and pull out three numbers (days, hours,
preparation); second, to turn the fuzzy system's numeric result into a
friendly explanation and study tips.

**4. Why are you using LangChain?**
LangChain makes it easy to build a "prompt → LLM → structured output"
pipeline. It lets me force the AI's answer into a clean data format
(using Pydantic) instead of parsing messy text myself.

**5. What does the LLM extract?**
Three values: `days_until_exam` (0–30), `study_hours` (0–12), and
`preparation_level` (0–100%). It understands phrases like "poor
preparation" or "next week" and converts them into these numbers.

**6. What is fuzzy logic?**
Fuzzy logic is a way of reasoning with degrees of truth instead of
strict true/false. For example, "3 days left" can be partly "Near"
and partly "Medium" at the same time, instead of just one category.

**7. What is fuzzification?**
Fuzzification is the process of converting a crisp number (like
`days = 5`) into membership degrees for fuzzy categories, e.g. `Near =
0.25, Medium = 0.29, Far = 0.0`.

**8. What are membership functions?**
They are mathematical shapes (triangles or trapezoids in our project)
that define how strongly a value belongs to a fuzzy category. For
example, the "Near" category is 1.0 (fully true) for 0–2 days and
gradually fades to 0 by day 6.

**9. What are fuzzy rules?**
They are IF-THEN statements written in fuzzy terms, like: "IF days is
Near AND preparation is Poor THEN study_time is High." My project has
18 such rules covering different situations.

**10. What is Mamdani inference?**
It's a type of fuzzy inference method where: inputs are fuzzified →
each rule's strength is found using the minimum (AND) operator → all
rules for one output are combined using the maximum (aggregation) →
finally, one crisp number is calculated using defuzzification.

**11. What is defuzzification?**
It's the last step that converts the combined fuzzy output back into
one single crisp number — for example, turning a fuzzy "study time is
mostly High" into an actual number like `6.6 hours`.

**12. Why use centroid defuzzification?**
Centroid (center of gravity) looks at the entire shape of the
combined fuzzy output, not just its peak, so the result changes
smoothly as inputs change slightly. This avoids sudden, unrealistic
jumps in the recommendation.

**13. Why not use normal if/else?**
Plain if/else needs hard cut-offs (e.g. "days < 3 = near"), so two
very similar situations (2.9 days vs 3.1 days) could get completely
different advice. Fuzzy logic allows partial membership in multiple
categories, giving smoother and more realistic recommendations.

**14. What is the input?**
A free-text sentence describing the student's situation — for
example: "My exam is in 5 days, I studied 2 hours today, and my
preparation is poor."

**15. What is the output?**
Two things: a recommended daily study time (in hours, 0–8) and a
study-need level (Low/Medium/High), plus an AI-written explanation and
3–5 study tips.
