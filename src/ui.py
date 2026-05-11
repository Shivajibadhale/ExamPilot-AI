import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #080b12;
            --panel: #111827;
            --panel-2: #162033;
            --text: #eef4ff;
            --muted: #9caec8;
            --line: rgba(148, 163, 184, 0.24);
            --cyan: #24d3ee;
            --green: #58d68d;
            --pink: #f472b6;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(36, 211, 238, 0.12), transparent 32rem),
                linear-gradient(135deg, #080b12 0%, #0d1320 42%, #10151e 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: rgba(8, 11, 18, 0.95);
            border-right: 1px solid var(--line);
        }

        .hero {
            min-height: 260px;
            display: flex;
            align-items: center;
            padding: 42px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                linear-gradient(120deg, rgba(36, 211, 238, 0.16), rgba(244, 114, 182, 0.08)),
                linear-gradient(180deg, rgba(17, 24, 39, 0.88), rgba(12, 18, 30, 0.92));
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            margin-bottom: 18px;
        }

        .hero h1 {
            font-size: 3.3rem;
            line-height: 1.03;
            margin: 0 0 14px 0;
            letter-spacing: 0;
        }

        .hero p {
            color: var(--muted);
            max-width: 920px;
            font-size: 1.05rem;
        }

        .eyebrow {
            color: var(--cyan) !important;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.78rem !important;
            letter-spacing: 0 !important;
            margin-bottom: 10px !important;
        }

        .agent-card {
            min-height: 138px;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
            background: rgba(17, 24, 39, 0.72);
        }

        .agent-card h3 {
            font-size: 1rem;
            margin: 0 0 8px 0;
        }

        .agent-card p {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0;
        }

        .status-pill {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 14px;
            margin: 16px 0;
            background: rgba(22, 32, 51, 0.72);
            color: var(--text);
        }

        .status-pill.muted {
            color: var(--muted);
        }

        .chat-row {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 10px;
            background: rgba(17, 24, 39, 0.82);
        }

        .chat-role {
            color: var(--cyan);
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 6px;
            text-transform: uppercase;
        }

        div.stButton > button,
        div[data-testid="stChatInput"] textarea {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .hero {
                padding: 26px;
            }
            .hero h1 {
                font-size: 2.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def agent_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="agent-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(role: str, content: str) -> None:
    label = "Student" if role == "user" else "AI Copilot"
    st.markdown(
        f"""
        <div class="chat-row">
            <div class="chat-role">{label}</div>
            <div>{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
