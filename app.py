"""AI Resume and Job Matcher

Provide a resume and a job description (paste text or upload PDF/DOCX/TXT).
The app highlights matching skills, missing requirements, and suggests
wording improvements. Powered by the Groq API.

Local run:  streamlit run app.py
Deploy:     push to GitHub, then add GROQ_API_KEY in Streamlit Cloud > Secrets
"""

import io
import json
import os
import re

import streamlit as st
from groq import Groq, GroqError

# Default model. If Groq retires it, change this or set GROQ_MODEL in secrets.
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_CHARS = 12000  # per document, keeps requests fast and within limits

SYSTEM_PROMPT = """You are an expert recruiter and resume coach.

You will receive a RESUME and a JOB DESCRIPTION. Compare them and respond with
ONLY a valid JSON object (no markdown fences, no extra text) with exactly this
structure:

{
  "match_score": 0-100 integer estimating how well the resume fits the job,
  "summary": "2-3 sentence overall assessment",
  "matching_skills": [
    {"skill": "skill or requirement from the job", "evidence": "where/how the resume shows it"}
  ],
  "missing_requirements": [
    {
      "requirement": "requirement in the job description not found in the resume",
      "importance": "High | Medium | Low",
      "suggestion": "honest advice: how to address it (e.g. relevant experience to mention if truly held, a course, or a project)"
    }
  ],
  "wording_improvements": [
    {
      "original": "an exact line or phrase from the resume",
      "improved": "a stronger rewrite that uses action verbs, measurable impact and job-relevant keywords",
      "reason": "why this is better for this job"
    }
  ],
  "keywords_to_add": ["important job keywords missing from the resume"]
}

Rules:
- Base everything only on the text provided. Never invent experience, employers, degrees or numbers.
- If a rewrite needs a metric the resume does not give, use a placeholder like [X%] rather than making one up.
- Give 3-8 wording improvements, prioritizing the most impactful.
- Keep each item concise."""


def get_secret(name: str) -> str | None:
    """Read a value from Streamlit secrets, falling back to environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def read_upload(uploaded) -> str:
    """Extract text from an uploaded PDF, DOCX or TXT file."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    return data.decode("utf-8", errors="ignore").strip()


def get_text_input(label: str, key: str, placeholder: str) -> str:
    """Show a paste box plus an optional file uploader; return the resulting text."""
    text = st.text_area(label, height=300, key=f"{key}_text", placeholder=placeholder)
    uploaded = st.file_uploader(
        f"…or upload {label.lower()} (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        key=f"{key}_file",
    )
    if uploaded is not None:
        try:
            extracted = read_upload(uploaded)
            if extracted:
                return extracted
            st.warning(f"Couldn't read any text from {uploaded.name}. Try pasting the text instead.")
        except Exception as e:
            st.warning(f"Couldn't read {uploaded.name} ({e}). Try pasting the text instead.")
    return text.strip()


def parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating accidental code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def analyze(client: Groq, resume: str, job: str) -> dict:
    user_message = f"RESUME:\n{resume[:MAX_CHARS]}\n\n---\n\nJOB DESCRIPTION:\n{job[:MAX_CHARS]}"
    response = client.chat.completions.create(
        model=get_secret("GROQ_MODEL") or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )
    return parse_json(response.choices[0].message.content)


def score_label(score: int) -> str:
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Partial match"
    return "Weak match"


def main() -> None:
    st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="📄", layout="wide")
    st.title("📄 AI Resume & Job Matcher")
    st.caption("See how your resume fits a job, what's missing, and how to word it better.")

    api_key = get_secret("GROQ_API_KEY")

    with st.sidebar:
        st.header("Settings")
        if not api_key:
            api_key = st.text_input("Groq API key", type="password") or None
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "1. Add your resume\n"
            "2. Add the job description\n"
            "3. Click *Match my resume*\n\n"
            "Your text is sent to Groq for analysis and is not stored by this app."
        )

    col_resume, col_job = st.columns(2)
    with col_resume:
        resume = get_text_input("Resume", "resume", "Paste your resume text here…")
    with col_job:
        job = get_text_input("Job description", "job", "Paste the job description here…")

    if st.button("Match my resume", type="primary"):
        if not resume or not job:
            st.warning("Please provide both a resume and a job description.")
            return
        if not api_key:
            st.error("Please add your Groq API key (sidebar, or GROQ_API_KEY in secrets).")
            return
        if len(resume) > MAX_CHARS or len(job) > MAX_CHARS:
            st.info(f"Long text detected: only the first {MAX_CHARS:,} characters of each are analyzed.")

        with st.spinner("Comparing your resume with the job..."):
            try:
                st.session_state["match"] = analyze(Groq(api_key=api_key), resume, job)
            except json.JSONDecodeError:
                st.error("The response couldn't be parsed. Please try again.")
                return
            except GroqError as e:
                st.error(f"API error: {e}")
                return

    result = st.session_state.get("match")
    if not result:
        return

    st.markdown("---")
    try:
        score = max(0, min(100, int(result.get("match_score", 0))))
    except (TypeError, ValueError):
        score = 0

    m1, m2 = st.columns([1, 3])
    with m1:
        st.metric("Match score", f"{score}%", score_label(score))
    with m2:
        st.progress(score / 100)
        st.write(result.get("summary", ""))

    tab_match, tab_missing, tab_words, tab_keys = st.tabs(
        ["✅ Matching skills", "⚠️ Missing requirements", "✏️ Wording improvements", "🔑 Keywords to add"]
    )

    with tab_match:
        items = result.get("matching_skills", [])
        if not items:
            st.info("No clear matches found.")
        for item in items:
            st.markdown(f"✅ **{item.get('skill', '')}**  \n{item.get('evidence', '')}")

    with tab_missing:
        items = result.get("missing_requirements", [])
        if not items:
            st.success("No major gaps found. Nice!")
        icons = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
        for item in items:
            level = item.get("importance", "Medium")
            with st.expander(f"{icons.get(level, '🟠')} {item.get('requirement', '')}  ·  {level}", expanded=True):
                st.markdown(f"💡 {item.get('suggestion', '')}")

    with tab_words:
        items = result.get("wording_improvements", [])
        if not items:
            st.info("No wording suggestions.")
        for i, item in enumerate(items, 1):
            with st.expander(f"Suggestion {i}", expanded=True):
                st.markdown(f"❌ **Before:** {item.get('original', '')}")
                st.markdown(f"✅ **After:** {item.get('improved', '')}")
                st.caption(item.get("reason", ""))
        if items:
            st.caption("Replace any [X%]-style placeholders with your real numbers, and only keep claims that are true.")

    with tab_keys:
        keywords = result.get("keywords_to_add", [])
        if keywords:
            st.write("Work these in naturally where they truthfully apply to your experience:")
            st.markdown(" ".join(f"`{k}`" for k in keywords))
        else:
            st.info("No missing keywords found.")


if __name__ == "__main__":
    main()
