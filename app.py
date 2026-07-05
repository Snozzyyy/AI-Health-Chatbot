import streamlit as st
import numpy as np
import json
import pickle
import os

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Chatbot",
    page_icon="\U0001f3e5",
    layout="wide"
)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_PRIMARY = os.path.join(BASE_DIR, "models", "optimised_model.keras")
MODEL_PATH_FALLBACK = os.path.join(BASE_DIR, "models", "best_model.keras")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "artifacts", "feature_names.json")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "artifacts", "label_encoder.pkl")
LABEL_MAPPING_PATH = os.path.join(BASE_DIR, "artifacts", "label_mapping.json")


# ─── Model & Artifact Loading ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load TF model, label encoder, and feature names."""
    import tensorflow as tf

    # Load model
    model = None
    if os.path.exists(MODEL_PATH_PRIMARY):
        model = tf.keras.models.load_model(MODEL_PATH_PRIMARY)
    elif os.path.exists(MODEL_PATH_FALLBACK):
        model = tf.keras.models.load_model(MODEL_PATH_FALLBACK)
    else:
        return None, None, None

    # Load feature names
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    # Load label encoder
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder, feature_names


def get_gemini_client(api_key):
    """Configure and return Gemini client instance."""
    from google import genai

    client = genai.Client(api_key=api_key)
    return client


# ─── Core Functions ───────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"


def extract_symptoms(patient_text, client, feature_names):
    """Use Gemini to extract symptoms from patient text and map to feature vocabulary."""
    from google.genai import types

    symptom_list = ", ".join(feature_names)

    prompt = f"""You are a medical symptom extraction assistant. Your job is to identify symptoms
from a patient's natural language description and map them ONLY to symptoms from the provided list.

VALID SYMPTOMS LIST:
{symptom_list}

PATIENT DESCRIPTION:
"{patient_text}"

INSTRUCTIONS:
- Identify all symptoms mentioned or implied in the patient's description.
- Map them ONLY to symptoms from the valid list above. Do not invent new symptoms.
- Handle informal language, misspellings, and abbreviations (e.g., "can't breathe" -> "shortness of breath").
- If a described symptom doesn't closely match anything in the list, skip it.
- If no valid symptoms are detected, return an empty list.

Respond in JSON format:
{{"matched_symptoms": ["symptom1", "symptom2", ...], "reasoning": "brief explanation of mapping"}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        result = json.loads(response.text)
        matched = [s for s in result.get("matched_symptoms", []) if s in feature_names]
        return matched, result.get("reasoning", "")
    except Exception as e:
        st.error(f"Symptom extraction failed: {str(e)}")
        return [], ""


def create_feature_vector(symptoms, feature_names):
    """Convert list of symptom names to binary numpy array of shape (1, 252)."""
    vector = np.zeros((1, len(feature_names)), dtype=np.float32)
    for symptom in symptoms:
        if symptom in feature_names:
            idx = feature_names.index(symptom)
            vector[0, idx] = 1.0
    return vector


def predict_diseases(feature_vector, model, label_encoder, top_k=5):
    """Run model prediction, return top-k diseases with confidence scores."""
    probabilities = model.predict(feature_vector, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    predictions = []
    for idx in top_indices:
        disease = label_encoder.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        predictions.append({"disease": disease, "confidence": confidence})

    return predictions


def assess_urgency_and_advise(predictions, symptoms, client):
    """Use Gemini to assess urgency level and provide health advice."""
    from google.genai import types

    pred_text = "\n".join(
        [f"- {p['disease']} ({p['confidence']*100:.1f}%)" for p in predictions]
    )
    symptom_text = ", ".join(symptoms)

    prompt = f"""You are a medical triage assistant. Based on the following AI-predicted conditions
and reported symptoms, assess the urgency and provide advice.

PREDICTED CONDITIONS (with confidence):
{pred_text}

REPORTED SYMPTOMS:
{symptom_text}

Provide your assessment in JSON format:
{{
    "urgency_level": "Low" or "Medium" or "High" or "Emergency",
    "explanation": "Brief 1-2 sentence explanation of urgency level",
    "advice": ["advice point 1", "advice point 2", "advice point 3"]
}}

RULES:
- Be conservative: if uncertain, lean toward higher urgency.
- Always include "Consult a healthcare professional for proper diagnosis" as one advice point.
- Keep advice actionable and concise.
- "Emergency" means seek immediate medical attention (e.g., heart attack symptoms, stroke signs).
- "High" means see a doctor within 24 hours.
- "Medium" means schedule an appointment soon.
- "Low" means monitor and practice self-care."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "urgency_level": "Medium",
            "explanation": "Unable to assess urgency automatically. Please use your judgment.",
            "advice": [
                "Consult a healthcare professional for proper diagnosis.",
                f"Error details: {str(e)}"
            ]
        }


def format_response(symptoms_detected, predictions, urgency_info):
    """Format the complete bot response as structured content."""
    return {
        "symptoms": symptoms_detected,
        "predictions": predictions,
        "urgency": urgency_info
    }


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚕️ AI Health Chatbot")
    st.markdown("---")
    st.markdown(
        """**How it works:**
1. Describe your symptoms in plain language
2. AI extracts and maps symptoms to medical terms
3. ML model predicts possible conditions"""
    )
    st.markdown("---")

    secrets_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    default_key = os.environ.get("GEMINI_API_KEY", "") or secrets_key
    api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Enter your Gemini API key to enable AI features",
        key="api_key_input"
    )
    if api_key:
        st.session_state["api_key"] = api_key

    st.markdown("---")
    st.error(
        "**Disclaimer:** This tool is for educational purposes only. "
        "It does not provide medical diagnosis. Always consult a healthcare professional."
    )
    st.markdown("---")

    st.markdown("**Model Statistics:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Diseases", "175")
    col2.metric("Symptoms", "252")
    col3.metric("Accuracy", "~90%")


# ─── Session State Initialization ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm your AI Health Pre-Triage Assistant. "
                "Describe your symptoms in detail, and I'll help identify possible conditions.\n\n"
                "For example: *\"I've been having a headache for 3 days, "
                "with some nausea and sensitivity to light.\"*"
            )
        }
    ]

if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False


# ─── Main Area ────────────────────────────────────────────────────────────────
st.title("\U0001f3e5 AI Health Chatbot — Pre-Triage Assistant")

# Load model
model, label_encoder, feature_names = load_model()
if model is not None:
    st.session_state["model_loaded"] = True
else:
    st.warning("Model files not found. Please ensure model files are in the `models/` directory.")

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and isinstance(msg.get("data"), dict):
            data = msg["data"]
            with st.expander(f"Detected Symptoms ({len(data['symptoms'])})", expanded=False):
                for s in data["symptoms"]:
                    st.markdown(f"- {s}")
            for i, pred in enumerate(data["predictions"]):
                col_name, col_bar = st.columns([2, 3])
                with col_name:
                    st.markdown(f"**{i+1}. {pred['disease'].replace('_', ' ').title()}**")
                with col_bar:
                    st.progress(
                        min(pred["confidence"], 1.0),
                        text=f"{pred['confidence']*100:.1f}%"
                    )

# Chat input
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Validate prerequisites
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        with st.chat_message("assistant"):
            st.info("Please enter your Gemini API key in the sidebar to enable AI-powered analysis.")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "Please enter your Gemini API key in the sidebar to enable AI-powered analysis."
            })
    elif not st.session_state["model_loaded"]:
        with st.chat_message("assistant"):
            st.error("The prediction model is not loaded. Cannot process symptoms.")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "The prediction model is not loaded. Cannot process symptoms."
            })
    else:
        # Process symptoms
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your symptoms..."):
                gemini_client = get_gemini_client(api_key)

                # Step 1: Extract symptoms
                symptoms_detected, reasoning = extract_symptoms(user_input, gemini_client, feature_names)

                if not symptoms_detected:
                    response_text = (
                        "I couldn't identify any specific symptoms from your description. "
                        "Could you try describing your symptoms more specifically? "
                        "For example, mention pain location, duration, severity, or associated issues."
                    )
                    st.markdown(response_text)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response_text
                    })
                else:
                    # Step 2: Create feature vector and predict
                    feature_vector = create_feature_vector(symptoms_detected, feature_names)
                    predictions = predict_diseases(feature_vector, model, label_encoder)

                    # Step 3: Assess urgency
                    urgency_info = assess_urgency_and_advise(predictions, symptoms_detected, gemini_client)

                    # ─── Render Response ──────────────────────────────────────
                    # Detected symptoms
                    with st.expander(f"Detected Symptoms ({len(symptoms_detected)})", expanded=False):
                        for s in symptoms_detected:
                            st.markdown(f"- {s}")
                        if reasoning:
                            st.caption(f"Reasoning: {reasoning}")

                    # Predictions
                    st.markdown("### Top Predictions")
                    for i, pred in enumerate(predictions):
                        col_name, col_bar = st.columns([2, 3])
                        with col_name:
                            st.markdown(f"**{i+1}. {pred['disease'].replace('_', ' ').title()}**")
                        with col_bar:
                            st.progress(
                                min(pred["confidence"], 1.0),
                                text=f"{pred['confidence']*100:.1f}%"
                            )

                    # Urgency
                    st.markdown("### Urgency Assessment")
                    urgency_level = urgency_info.get("urgency_level", "Medium")
                    urgency_colors = {
                        "Low": "✅",
                        "Medium": "⚠️",
                        "High": "\U0001f536",
                        "Emergency": "\U0001f6a8"
                    }
                    icon = urgency_colors.get(urgency_level, "⚠️")
                    st.metric(
                        label="Urgency Level",
                        value=f"{icon} {urgency_level}"
                    )
                    st.markdown(urgency_info.get("explanation", ""))

                    # Advice
                    st.markdown("### Recommendations")
                    for advice_point in urgency_info.get("advice", []):
                        st.markdown(f"- {advice_point}")

                    # Store in session state
                    response_data = format_response(symptoms_detected, predictions, urgency_info)
                    summary = (
                        f"Detected {len(symptoms_detected)} symptom(s). "
                        f"Top prediction: {predictions[0]['disease'].replace('_', ' ').title()} "
                        f"({predictions[0]['confidence']*100:.1f}%). "
                        f"Urgency: {urgency_level}."
                    )
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": summary,
                        "data": response_data
                    })
