"""
SentimentScribe — Supabase authentication layer.

Supports:
  - Email / password sign in and sign up
  - Google OAuth (SSO) — no email verification required

Usage in any Streamlit page:
    from dashboard.auth import require_auth, get_current_user, logout

    require_auth()               # gate: shows login UI and calls st.stop() if not logged in
    user = get_current_user()    # returns Supabase user dict after login
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
from supabase import create_client, Client


# ── Supabase client (cached per session) ──────────────────────────────────────

@st.cache_resource
def _get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)


# ── Public helpers ─────────────────────────────────────────────────────────────

def require_auth() -> None:
    """Gate function. If no active session, render login UI and halt the page."""
    _handle_oauth_callback()
    if st.session_state.get("user") is None:
        _render_auth_ui()
        st.stop()


def get_current_user() -> dict | None:
    """Return the current Supabase user dict, or None if not logged in."""
    return st.session_state.get("user")


def get_supabase() -> Client:
    """Return the shared Supabase client for data operations."""
    return _get_supabase()


def logout() -> None:
    """Sign the user out and clear session state."""
    try:
        _get_supabase().auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("user", None)
    st.session_state.pop("access_token", None)
    st.rerun()


# ── OAuth callback handler ─────────────────────────────────────────────────────

def _handle_oauth_callback() -> None:
    """Detect a returning OAuth redirect and exchange the code for a session."""
    code = st.query_params.get("code")
    if not code or st.session_state.get("user"):
        return
    try:
        resp = _get_supabase().auth.exchange_code_for_session({"auth_code": code})
        if resp.user:
            st.session_state["user"] = resp.user
            st.session_state["access_token"] = resp.session.access_token
            st.query_params.clear()
            st.rerun()
    except Exception as exc:
        st.query_params.clear()
        st.error(f"Google sign-in failed: {exc}")


# ── Internal UI ───────────────────────────────────────────────────────────────

def _render_auth_ui() -> None:
    """Render centered login / signup tabs."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 📚 SentimentScribe")
        st.caption("Book review sentiment analysis · Sign in to continue")
        st.divider()

        # Google SSO button
        _google_oauth_button()

        st.divider()
        st.caption("or continue with email")

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            _login_form()

        with tab_signup:
            _signup_form()


def _google_oauth_button() -> None:
    """Render the Google sign-in button and trigger the OAuth redirect."""
    if st.button("Continue with Google", use_container_width=True, icon="🔵"):
        app_url = st.secrets.get("APP_URL", "http://localhost:8501")
        try:
            resp = _get_supabase().auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": app_url,
                    "query_params": {"access_type": "offline", "prompt": "consent"},
                },
            })
            # Navigate the parent window to Google's auth page
            components.html(
                f"<script>window.parent.location.href = '{resp.url}';</script>",
                height=0,
            )
        except Exception as exc:
            st.error(f"Could not start Google sign-in: {exc}")


def _login_form() -> None:
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

    if submitted:
        if not email or not password:
            st.error("Please enter both email and password.")
            return
        try:
            resp = _get_supabase().auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state["user"] = resp.user
            st.session_state["access_token"] = resp.session.access_token
            st.rerun()
        except Exception as exc:
            st.error(f"Sign in failed: {exc}")


def _signup_form() -> None:
    with st.form("signup_form"):
        email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pw",
                                  help="At least 6 characters")
        password2 = st.text_input("Confirm password", type="password", key="signup_pw2")
        submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submitted:
        if not email or not password:
            st.error("Please fill in all fields.")
            return
        if password != password2:
            st.error("Passwords do not match.")
            return
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return
        try:
            resp = _get_supabase().auth.sign_up({"email": email, "password": password})
            if resp.user:
                st.success("Account created! Check your email to confirm, then sign in.")
            else:
                st.warning("Sign up submitted. Check your email for a confirmation link.")
        except Exception as exc:
            st.error(f"Sign up failed: {exc}")
