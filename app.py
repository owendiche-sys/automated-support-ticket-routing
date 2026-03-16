from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Support Ticket Routing",
    page_icon="🎫",
    layout="wide"
)


# =========================
# HELPERS
# =========================
def softmax(x, axis=1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


@st.cache_resource
def load_pipeline(model_path: Path):
    return joblib.load(model_path)


def get_prediction_details(fitted_pipeline, text_input: str):
    """
    Return:
    - predicted class
    - confidence score
    - scoring method
    - ranked class probabilities / pseudo-probabilities
    """
    text_series = pd.Series([text_input])

    model = fitted_pipeline.named_steps["model"]
    vectorizer = fitted_pipeline.named_steps["tfidf"]

    if hasattr(model, "predict_proba"):
        X_vec = vectorizer.transform(text_series)
        probs = model.predict_proba(X_vec)
        classes = model.classes_
        pred_idx = np.argmax(probs, axis=1)[0]
        predicted_label = classes[pred_idx]
        confidence = float(probs[0, pred_idx])

        ranked = pd.DataFrame({
            "routed_team": classes,
            "score": probs[0]
        }).sort_values("score", ascending=False).reset_index(drop=True)

        return predicted_label, confidence, "predict_proba", ranked

    if hasattr(model, "decision_function"):
        X_vec = vectorizer.transform(text_series)
        decision = model.decision_function(X_vec)

        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])

        pseudo_probs = softmax(decision, axis=1)
        classes = model.classes_
        pred_idx = np.argmax(pseudo_probs, axis=1)[0]
        predicted_label = classes[pred_idx]
        confidence = float(pseudo_probs[0, pred_idx])

        ranked = pd.DataFrame({
            "routed_team": classes,
            "score": pseudo_probs[0]
        }).sort_values("score", ascending=False).reset_index(drop=True)

        return predicted_label, confidence, "decision_function_softmax", ranked

    predicted_label = fitted_pipeline.predict(text_series)[0]
    ranked = pd.DataFrame({
        "routed_team": [predicted_label],
        "score": [np.nan]
    })

    return predicted_label, np.nan, "not_available", ranked


def build_ticket_text(subject: str, description: str) -> str:
    subject = subject.strip()
    description = description.strip()
    combined = f"{subject} {description}".strip()
    return " ".join(combined.split())


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_ticket_routing_pipeline.joblib"

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

pipeline = load_pipeline(MODEL_PATH)


# =========================
# STYLES
# =========================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .sub-text {
            color: #4b5563;
            margin-bottom: 1.25rem;
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 16px;
            border-radius: 14px;
        }
        .result-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 0.2rem;
        }
        .result-value {
            font-size: 1.2rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">Automated Support Ticket Routing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Predict the most appropriate support team for an incoming ticket using the trained NLP routing pipeline.</div>',
    unsafe_allow_html=True
)


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Routing Settings")
review_threshold = st.sidebar.slider(
    "Human review threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.01
)

show_top_classes = st.sidebar.checkbox("Show class ranking", value=True)
show_example_tickets = st.sidebar.checkbox("Show example tickets", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Loaded model: `{MODEL_PATH.name}`")


# =========================
# INPUTS
# =========================
col1, col2 = st.columns([1.25, 1])

with col1:
    st.subheader("Ticket Input")

    subject = st.text_input(
        "Subject",
        placeholder="Example: Need help with my account"
    )

    description = st.text_area(
        "Description",
        height=220,
        placeholder=(
            "Example: I cannot reset my password and I have been locked out of the portal. "
            "I tried several times but it still does not work."
        )
    )

    submitted = st.button("Route Ticket", use_container_width=True)

with col2:
    st.subheader("Example Use Cases")

    if show_example_tickets:
        example_options = {
            "Account access issue":
                ("Need help logging in",
                 "I cannot sign in to my account and the password reset process is not working."),
            "Billing issue":
                ("Charge looks wrong",
                 "I noticed an unexpected charge on my invoice and need someone to review it."),
            "Technical issue":
                ("App keeps failing",
                 "The dashboard keeps crashing when I try to export my report."),
            "Delivery issue":
                ("Order has not arrived",
                 "Tracking has not updated and I still have not received the package."),
            "Refund / return issue":
                ("Need refund update",
                 "The item arrived damaged and I want to know the status of my refund."),
            "Product inquiry":
                ("Question about plans",
                 "Can you explain the difference between the available pricing plans?")
        }

        selected_example = st.selectbox(
            "Load an example",
            options=["None"] + list(example_options.keys())
        )

        if selected_example != "None":
            ex_subject, ex_description = example_options[selected_example]
            st.info(
                f"**Subject:** {ex_subject}\n\n"
                f"**Description:** {ex_description}"
            )


# =========================
# PREDICTION
# =========================
if submitted:
    ticket_text = build_ticket_text(subject, description)

    if not ticket_text:
        st.warning("Please enter a subject, description, or both.")
        st.stop()

    predicted_team, confidence_score, score_method, ranked_df = get_prediction_details(
        pipeline,
        ticket_text
    )

    if np.isnan(confidence_score):
        routing_decision = "Auto-route"
    else:
        routing_decision = "Auto-route" if confidence_score >= review_threshold else "Human review"

    st.markdown("---")
    st.subheader("Routing Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="result-label">Predicted Team</div>
                <div class="result-value">{predicted_team}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:
        conf_text = "Not available" if np.isnan(confidence_score) else f"{confidence_score:.3f}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="result-label">Confidence Score</div>
                <div class="result-value">{conf_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="result-label">Routing Decision</div>
                <div class="result-value">{routing_decision}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(f"Scoring method: {score_method}")

    if routing_decision == "Human review":
        st.warning(
            "This ticket falls below the review threshold and should be checked by a human before final routing."
        )
    else:
        st.success(
            "This ticket is above the review threshold and can be auto-routed."
        )

    if show_top_classes and not ranked_df.empty:
        st.subheader("Class Ranking")

        ranked_display = ranked_df.copy()
        ranked_display["score"] = ranked_display["score"].apply(
            lambda x: round(float(x), 4) if pd.notna(x) else x
        )

        st.dataframe(
            ranked_display,
            use_container_width=True,
            hide_index=True
        )

        chart_df = ranked_df.head(6).copy()
        chart_df = chart_df.sort_values("score", ascending=True)

        st.subheader("Top Class Scores")
        st.bar_chart(
            data=chart_df.set_index("routed_team")["score"],
            use_container_width=True
        )

    st.subheader("Combined Ticket Text")
    st.code(ticket_text, language="text")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "This app uses the saved TF-IDF + classifier pipeline from the support ticket routing project."
)