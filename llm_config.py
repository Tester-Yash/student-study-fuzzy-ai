"""
llm_config.py
=============
Central place that creates the LangChain chat model object.

Supports two providers:
    - "groq"   (default, free tier available)
    - "openai" (optional)

API keys are checked in order of priority:
    1. Browser session state (st.session_state) -> entered by the user in their own browser tab
    2. Streamlit cloud secrets (st.secrets)     -> used when deployed on Streamlit Cloud
    3. Environment variables (.env)            -> used for local development

This ensures that individual users can provide their own API key in their
own browser without burning the repository owner's key.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class MissingAPIKeyError(Exception):
    """Raised when no valid API key can be found for the selected provider."""
    pass


def _get_secret(key_name: str) -> str | None:
    """
    Look up a configuration secret. Checks the current browser's
    Streamlit session state first, then cloud secrets, then .env.
    """
    # 1. Try Streamlit session state (private to this browser tab)
    try:
        import streamlit as st
        # Direct key match
        val = st.session_state.get(key_name)
        if val and str(val).strip():
            return str(val).strip()

        # Dedicated browser-session state keys
        if key_name == "GROQ_API_KEY":
            browser_key = st.session_state.get("user_groq_key")
            if browser_key and str(browser_key).strip():
                return str(browser_key).strip()

        if key_name == "OPENAI_API_KEY":
            browser_key = st.session_state.get("user_openai_key")
            if browser_key and str(browser_key).strip():
                return str(browser_key).strip()

        if key_name == "LLM_PROVIDER":
            browser_provider = st.session_state.get("user_provider")
            if browser_provider and str(browser_provider).strip():
                return str(browser_provider).strip()

        # 2. Try Streamlit secrets store
        if key_name in st.secrets and st.secrets[key_name]:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    # 3. Fall back to local environment variable (.env)
    return os.getenv(key_name)


def get_provider() -> str:
    """Return the configured LLM provider, defaulting to 'groq'."""
    return (_get_secret("LLM_PROVIDER") or "groq").lower().strip()


def get_llm(temperature: float = 0.2, max_tokens: int | None = None):
    """
    Build and return a LangChain chat model instance based on the
    configured provider. Raises MissingAPIKeyError with a friendly
    message if the required key is missing.
    """
    provider = get_provider()

    if provider == "groq":
        api_key = _get_secret("GROQ_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is missing. Please enter your Groq API key in the "
                "⚙️ Settings sidebar on the left. It will stay private to your browser session."
            )
        model_name = _get_secret("GROQ_MODEL") or "qwen/qwen3.8-27b"
        from langchain_groq import ChatGroq
        # Groq free tier enforces an Output Tokens Per Minute (OTPM) limit of 1000.
        # Specifying max_tokens prevents Groq from over-estimating token consumption.
        token_limit = max_tokens if max_tokens is not None else 400
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            max_tokens=token_limit,
            max_retries=2,
        )

    elif provider == "openai":
        api_key = _get_secret("OPENAI_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is missing. Please enter your OpenAI API key in the "
                "⚙️ Settings sidebar on the left. It will stay private to your browser session."
            )
        model_name = _get_secret("OPENAI_MODEL") or "gpt-4o-mini"
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model_name, "temperature": temperature, "api_key": api_key}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'openai'."
        )
