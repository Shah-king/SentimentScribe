"""
SentimentScribe — Supabase authentication layer.

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
from supabase import create_client, Client


# ── Supabase client ────────────────────────────────────────────────────────────

@st.cache_resource
def _get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)


# ── Public helpers ─────────────────────────────────────────────────────────────

def require_auth() -> None:
    """Gate function. If no active session, render login UI and halt the page."""
    if st.session_state.get("user") is None and not st.session_state.get("demo_mode"):
        _inject_global_styles()
        _render_auth_ui()
        st.stop()


def get_current_user() -> dict | None:
    return st.session_state.get("user")


def get_supabase() -> Client:
    return _get_supabase()


def logout() -> None:
    try:
        _get_supabase().auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("user", None)
    st.session_state.pop("access_token", None)
    st.session_state.pop("demo_mode", None)
    st.rerun()


# ── CSS ───────────────────────────────────────────────────────────────────────

def _inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── 1. Strip ALL default Streamlit padding so columns fill the viewport ── */
        [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
            padding: 0 !important;
        }
        [data-testid="stMain"] > div:first-child {
            padding: 0 !important;
            max-width: 100% !important;
        }
        /* Remove gap injected by stBlock wrapper */
        [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }

        /* ── 2. Page background ────────────────────────────────────────────────── */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f0f0ff 0%, #e8e8ff 50%, #f5f0ff 100%);
        }
        [data-testid="stHeader"] { background: transparent !important; }
        #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
        [data-testid="stSidebar"] { display: none; }

        /* ── 3. Column containers: make them fill height and sit flush ─────────── */
        /*
         * Streamlit wraps each col in [data-testid="stColumn"].
         * We target them by position to apply the correct panel background.
         */
        [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            align-items: stretch !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        /* LEFT column — indigo gradient branding panel */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
            background: linear-gradient(160deg, #6366f1 0%, #4f46e5 45%, #7c3aed 100%);
            padding: 0 !important;
            min-height: 100vh;
        }
        /* RIGHT column — white auth card */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            background: #ffffff;
            padding: 0 !important;
            min-height: 100vh;
        }
        /* Inner div that Streamlit inserts inside each column */
        [data-testid="stColumn"] > div:first-child {
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
        }

        /* ── 4. Brand panel (pure HTML, lives inside left col) ─────────────────── */
        .brand-panel {
            width: 100%;
            max-width: 100%;
            min-height: 100vh;
            padding: 64px 48px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        .brand-panel::before {
            content: "";
            position: absolute;
            top: -80px; right: -80px;
            width: 320px; height: 320px;
            border-radius: 50%;
            background: rgba(255,255,255,0.06);
            pointer-events: none;
        }
        .brand-panel::after {
            content: "";
            position: absolute;
            bottom: -100px; left: -60px;
            width: 400px; height: 400px;
            border-radius: 50%;
            background: rgba(255,255,255,0.04);
            pointer-events: none;
        }
        .brand-logo {
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
            margin-bottom: 6px;
        }
        .brand-logo span { color: #c7d2fe; }
        .brand-tagline {
            font-size: 1.05rem;
            color: #c7d2fe;
            font-weight: 500;
            margin-bottom: 32px;
        }
        .brand-desc {
            color: rgba(255,255,255,0.8);
            font-size: 0.95rem;
            line-height: 1.7;
            margin-bottom: 36px;
            max-width: 340px;
        }
        .brand-feature {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
            color: #e0e7ff;
            font-size: 0.9rem;
        }
        .brand-feature-dot {
            width: 28px; height: 28px;
            border-radius: 8px;
            background: rgba(255,255,255,0.15);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.85rem;
            flex-shrink: 0;
        }
        .brand-badge {
            margin-top: 48px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 100px;
            padding: 8px 16px;
            color: #e0e7ff;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .brand-badge-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* ── 5. Auth panel wrapper (Streamlit widgets inside right col) ────────── */
        .auth-header {
            padding: 64px 52px 0 52px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
        }
        .auth-footer {
            padding: 0 52px 40px 52px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
        }
        .auth-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: #1e1b4b;
            margin-bottom: 6px;
            letter-spacing: -0.3px;
        }
        .auth-subtitle {
            font-size: 0.95rem;
            color: #6b7280;
            margin-bottom: 24px;
        }

        /* ── 6. Streamlit widgets inside right column: add side padding ────────── */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child
            [data-testid="stVerticalBlock"] > * {
            padding-left: 52px !important;
            padding-right: 52px !important;
        }
        /* But the top-level vertical block itself needs no extra padding */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child
            > div > [data-testid="stVerticalBlock"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        /* ── 7. Tab pills ──────────────────────────────────────────────────────── */
        [data-baseweb="tab-list"] {
            background: #f3f4f6 !important;
            border-radius: 12px !important;
            padding: 4px !important;
            gap: 2px !important;
            border: none !important;
            margin-top: 8px !important;
        }
        [data-baseweb="tab"] {
            border-radius: 10px !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            color: #6b7280 !important;
            padding: 8px 20px !important;
            border: none !important;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: #ffffff !important;
            color: #4f46e5 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }

        /* ── 8. Input fields ───────────────────────────────────────────────────── */
        .stTextInput > div > div > input {
            border: 1.5px solid #e5e7eb !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            font-size: 0.9rem !important;
            color: #111827 !important;
            background: #fafafa !important;
            transition: border-color 0.2s, box-shadow 0.2s !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
            background: #ffffff !important;
        }
        .stTextInput label {
            font-size: 0.825rem !important;
            font-weight: 600 !important;
            color: #374151 !important;
            margin-bottom: 4px !important;
        }

        /* ── 9. Primary button ─────────────────────────────────────────────────── */
        .stFormSubmitButton > button[kind="primaryFormSubmit"],
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            width: 100% !important;
            transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
            box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
        }
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover,
        .stButton > button[kind="primary"]:hover {
            opacity: 0.92 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
        }

        /* ── 10. Secondary / demo button ───────────────────────────────────────── */
        .stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 1.5px solid #e5e7eb !important;
            color: #374151 !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            width: 100% !important;
            transition: border-color 0.2s, background 0.2s, transform 0.15s !important;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: #6366f1 !important;
            color: #6366f1 !important;
            background: #f0f0ff !important;
            transform: translateY(-1px) !important;
        }

        /* ── 11. OR divider ────────────────────────────────────────────────────── */
        .or-divider {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 16px 0;
            color: #9ca3af;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .or-divider::before, .or-divider::after {
            content: "";
            flex: 1;
            height: 1px;
            background: #e5e7eb;
        }

        /* ── 12. Forgot password ───────────────────────────────────────────────── */
        .forgot-link {
            text-align: right;
            margin-top: -8px;
            margin-bottom: 12px;
        }
        .forgot-link a {
            color: #6366f1;
            font-size: 0.82rem;
            text-decoration: none;
            font-weight: 500;
        }
        .forgot-link a:hover { text-decoration: underline; }

        /* ── 13. Alert overrides ───────────────────────────────────────────────── */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            font-size: 0.875rem !important;
        }

        /* ── 14. Hide Streamlit form border ────────────────────────────────────── */
        [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }

        /* ── 15. Responsive: stack on mobile ───────────────────────────────────── */
        @media (max-width: 768px) {
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
                display: none !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
                min-height: 100vh;
            }
            .auth-header { padding: 40px 28px 0 28px; }
            .auth-footer  { padding: 0 28px 32px 28px; }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child
                [data-testid="stVerticalBlock"] > * {
                padding-left: 28px !important;
                padding-right: 28px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Internal UI ───────────────────────────────────────────────────────────────

def _render_auth_ui() -> None:
    # ── Single st.columns() call — both panels side by side ──────────────────
    # gap="small" keeps panels flush; CSS zeroes out remaining spacing.
    left, right = st.columns([1, 1], gap="small")

    # ── LEFT: pure HTML branding panel (no Streamlit widgets needed) ─────────
    with left:
        st.markdown(
            """
            <div class="brand-panel">
                <div class="brand-logo">📚 Sentiment<span>Scribe</span></div>
                <div class="brand-tagline">AI-Powered Sentiment Intelligence</div>
                <div class="brand-desc">
                    Analyze book reviews in real-time and uncover insights
                    using state-of-the-art NLP models trained on thousands of reviews.
                </div>
                <div class="brand-feature">
                    <div class="brand-feature-dot">⚡</div>
                    Real-time sentiment prediction
                </div>
                <div class="brand-feature">
                    <div class="brand-feature-dot">🔍</div>
                    Explainable AI (SHAP insights)
                </div>
                <div class="brand-feature">
                    <div class="brand-feature-dot">📈</div>
                    Trend monitoring &amp; analytics
                </div>
                <div class="brand-feature">
                    <div class="brand-feature-dot">📦</div>
                    Bulk CSV analysis &amp; export
                </div>
                <div class="brand-badge">
                    <div class="brand-badge-dot"></div>
                    ML model online · FastAPI + HuggingFace
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── RIGHT: static header HTML + Streamlit widgets + static footer HTML ───
    # KEY FIX: do NOT open an HTML div and try to close it around Streamlit
    # widgets — Streamlit renders each st.markdown as a self-contained iframe
    # fragment, so unclosed tags never wrap subsequent widgets.
    # Instead, the right column's white background is applied via CSS targeting
    # [data-testid="stColumn"]:last-child, and we use thin HTML blocks only for
    # the static title/subtitle/footer text.
    with right:
        # Static header (safe: no Streamlit widgets inside)
        st.markdown(
            """
            <div class="auth-header">
                <div class="auth-title">Welcome back 👋</div>
                <div class="auth-subtitle">Sign in to your account to continue</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Streamlit widgets — rendered directly in the column, no wrapper div
        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])
        with tab_login:
            _login_form()
        with tab_signup:
            _signup_form()

        # OR divider + demo button
        st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)
        if st.button("🚀  Try Demo — No account needed", key="demo_btn", use_container_width=True):
            st.session_state["demo_mode"] = True
            st.session_state["user"] = None
            st.rerun()

        # Static footer (safe: no Streamlit widgets inside)
        st.markdown(
            """
            <div class="auth-footer">
                <p style="text-align:center;color:#9ca3af;font-size:0.78rem;margin-top:16px;">
                    By signing in you agree to our Terms of Service
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _login_form() -> None:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("✉️  Email address", placeholder="you@example.com", key="login_email")
        password = st.text_input("🔒  Password", type="password", key="login_pw", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

    # Forgot password link (outside form so it doesn't submit)
    st.markdown(
        '<div class="forgot-link"><a href="#" onclick="window.location.search=\'?reset=1\'">Forgot password?</a></div>',
        unsafe_allow_html=True,
    )

    if submitted:
        if not email or not password:
            st.error("Please enter both email and password.")
            return
        with st.spinner("Signing in…"):
            try:
                resp = _get_supabase().auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
                st.session_state["user"] = resp.user
                st.session_state["access_token"] = resp.session.access_token
                st.session_state.pop("show_reset", None)
                st.rerun()
            except Exception:
                st.error("Incorrect email or password. Try demo mode to explore the app.")
                st.session_state["reset_email_prefill"] = email
                st.session_state["show_reset"] = True

    if st.session_state.get("show_reset"):
        st.divider()
        st.warning("Enter your email below to receive a password reset link.")
        prefill = st.session_state.get("reset_email_prefill", "")
        reset_email = st.text_input("Reset email", value=prefill, key="reset_email_input")
        if st.button("Send Reset Link", use_container_width=True):
            _send_password_reset(reset_email)


def _send_password_reset(email: str) -> None:
    if not email:
        st.error("Please enter your email address.")
        return
    app_url = st.secrets.get("APP_URL", "http://localhost:8501")
    try:
        _get_supabase().auth.reset_password_email(
            email,
            options={"redirect_to": app_url},
        )
        st.success(f"Reset link sent to **{email}**. Check your inbox.")
        st.session_state.pop("show_reset", None)
        st.session_state.pop("reset_email_prefill", None)
    except Exception as exc:
        st.error(f"Could not send reset email: {exc}")


def _signup_form() -> None:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.form("signup_form", clear_on_submit=False):
        email = st.text_input("✉️  Email address", placeholder="you@example.com", key="signup_email")
        password = st.text_input("🔒  Password", type="password", key="signup_pw",
                                  placeholder="Min. 6 characters")
        password2 = st.text_input("🔒  Confirm password", type="password", key="signup_pw2",
                                   placeholder="Repeat password")
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
        with st.spinner("Creating account…"):
            try:
                resp = _get_supabase().auth.sign_up({"email": email, "password": password})
                if resp.user:
                    st.success("Account created! Check your email to confirm, then sign in.")
                else:
                    st.warning("Sign up submitted. Check your email for a confirmation link.")
            except Exception as exc:
                st.error(f"Sign up failed: {exc}")
