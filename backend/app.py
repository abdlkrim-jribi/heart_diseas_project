"""
HeartGuard AI - Complete Streamlit Application
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Page config first
st.set_page_config(page_title="HeartGuard AI", page_icon="❤️", layout="wide")

# Add backend to path
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

# Try to change directory (safe if fails)
try:
    os.chdir(project_root)
except Exception:
    pass

# Import after path is set - with proper error handling
try:
    from model_service import get_predictor, HeartDiseasePredictor
    import logging
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Make sure all backend files are present in the models/ directory")
    st.stop()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def init_session_state():
    """Initialize session state variables"""
    if 'collected_features' not in st.session_state:
        st.session_state.collected_features = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None


def render_header():
    """Render app header"""
    st.set_page_config(page_title="HeartGuard AI", page_icon="❤️", layout="wide")

    st.title("🫀 HeartGuard AI")
    st.markdown("### Intelligent Cardiac Risk Assessment System")
    st.markdown("---")


def render_sidebar_debug():
    """Render debug information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🔧 Debug Info")

    model_p = project_root / 'models' / 'heart_model_gnb.pkl'
    scaler_p = project_root / 'models' / 'scaler.pkl'

    st.sidebar.caption(f"**Working Dir:** `{os.getcwd()}`")
    st.sidebar.caption(f"**Project Root:** `{project_root}`")
    st.sidebar.caption(f"**Model exists:** {model_p.exists()}")
    st.sidebar.caption(f"**Scaler exists:** {scaler_p.exists()}")


def render_sidebar_progress(predictor):
    """Render feature collection progress in sidebar"""
    st.sidebar.header("📊 Collection Progress")

    collected = st.session_state.collected_features
    total_features = 13
    collected_count = len(collected)
    progress = collected_count / total_features

    st.sidebar.progress(progress)
    st.sidebar.metric("Features Collected", f"{collected_count}/{total_features}")

    if collected:
        st.sidebar.markdown("#### ✅ Collected Features:")
        for feature, value in list(collected.items())[:5]:
            if feature in predictor.FEATURE_CONFIG:
                label = predictor.FEATURE_CONFIG[feature]['label']
                st.sidebar.text(f"• {label}: {value}")

    missing = set(predictor.FEATURE_NAMES) - set(collected.keys())
    if missing:
        st.sidebar.markdown(f"#### ⏳ Remaining: {len(missing)}")


def form_based_mode():
    """Form-based feature input"""
    st.header("📋 Form-Based Mode")
    st.markdown("Manually input all medical features using the form below.")

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Make sure your model files are in the models/ directory!")
        return

    with st.form("feature_form"):
        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 29, 77, 50, help="Patient age in years")
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox(
                "Chest Pain Type",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina",
                                       3: "Non-anginal Pain", 4: "Asymptomatic"}[x]
            )
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
            chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 250)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox(
                "Resting ECG",
                options=[0, 1, 2],
                format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality",
                                       2: "Left Ventricular Hypertrophy"}[x]
            )

        with col2:
            thalach = st.slider("Maximum Heart Rate", 71, 202, 150)
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1)
            slope = st.selectbox(
                "Slope of Peak Exercise ST",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x]
            )
            ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
            thal = st.selectbox(
                "Thalassemia",
                options=[3, 6, 7],
                format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x]
            )

        submitted = st.form_submit_button("🔍 Get Prediction", type="primary", use_container_width=True)

        if submitted:
            features = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }

            try:
                with st.spinner("Making prediction..."):
                    result = predictor.predict(features)
                    st.session_state.prediction_result = result
                    st.session_state.collected_features = features

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_prediction_result():
    """Render prediction results"""
    if st.session_state.prediction_result is None:
        return

    result = st.session_state.prediction_result

    st.markdown("---")
    st.header("🎯 Prediction Results")

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Diagnosis",
            result['diagnosis'],
            delta=None
        )

    with col2:
        st.metric(
            "Probability",
            f"{result['probability'] * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Risk Level",
            f"{result['risk_emoji']} {result['risk_level']}"
        )

    # Progress bar
    st.progress(result['probability'])

    # Disclaimer
    st.warning(result['medical_disclaimer'])

    # Action buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.collected_features = {}
            st.session_state.conversation_history = []
            st.rerun()

    with col_b:
        report_text = f"""HeartGuard AI Assessment Report

Diagnosis: {result['diagnosis']}
Probability: {result['probability'] * 100:.1f}%
Risk Level: {result['risk_level']}

{result['medical_disclaimer']}

Features Used:
{str(st.session_state.collected_features)}
"""
        st.download_button(
            "📥 Download Report",
            data=report_text,
            file_name="heart_assessment.txt",
            use_container_width=True
        )


def conversational_mode():
    """Conversational chatbot mode with Cerebras LLM"""
    st.header("💬 Conversational Mode")
    st.markdown("Chat naturally with AI to collect your health information.")

    try:
        from chatbot_service import get_chatbot
        chatbot = get_chatbot()
        predictor = get_predictor()
    except Exception as e:
        st.error(f"❌ Chatbot initialization failed: {e}")
        st.info("💡 Make sure CEREBRAS_API_KEY is set in .env file or Streamlit secrets")
        return

    # Display chat history
    for msg in st.session_state.conversation_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Feature collection progress (sidebar)
    collected_count = len(st.session_state.collected_features)
    if collected_count > 0:
        st.sidebar.success(f"📋 Features: {collected_count}/13 collected")

        missing = set(predictor.FEATURE_NAMES) - set(st.session_state.collected_features.keys())
        if missing and collected_count < 13:
            st.sidebar.info(f"Still need: {', '.join(list(missing)[:3])}")

    # Check if ready for prediction
    if collected_count == 13 and st.session_state.prediction_result is None:
        st.success("🎉 All 13 features collected! Ready for prediction.")
        if st.button("🔍 Get Prediction Now", type="primary", use_container_width=True):
            try:
                result = predictor.predict(st.session_state.collected_features)
                st.session_state.prediction_result = result
                st.rerun()
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # Display prediction if available
    if st.session_state.prediction_result:
        render_prediction_result()
        return

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, new_features = chatbot.chat(
                    prompt,
                    st.session_state.conversation_history,
                    st.session_state.collected_features
                )

                # Update collected features
                st.session_state.collected_features.update(new_features)

                # Add response to history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                st.markdown(response)

        st.rerun()

    # Quick start buttons
    if len(st.session_state.conversation_history) == 0:
        st.info("👋 Start by saying hello or sharing your age and medical history!")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🩺 Start Assessment"):
                starter = "Hello, I'd like to assess my heart disease risk."
                st.session_state.conversation_history.append({"role": "user", "content": starter})
                st.rerun()
        with col2:
            if st.button("❓ Learn More"):
                starter = "What information do you need from me?"
                st.session_state.conversation_history.append({"role": "user", "content": starter})
                st.rerun()
        with col3:
            if st.button("🔄 Reset Chat"):
                st.session_state.conversation_history = []
                st.session_state.collected_features = {}
                st.session_state.prediction_result = None
                st.rerun()


def main():
    """Main application entry point"""
    render_header()
    init_session_state()

    # Try to load predictor
    try:
        predictor = get_predictor()
        st.sidebar.success("✅ Model loaded successfully!")
        render_sidebar_progress(predictor)
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed!")
        st.error(f"""
        **Error loading model:** {e}

        **Required files:**
        - `models/heart_model_gnb.pkl`
        - `models/scaler.pkl`

        **Solution:**
        1. Run: python train_sklearn.py
        2. Restart the app
        """)
        render_sidebar_debug()
        return

    # Sidebar navigation
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Mode:",
        ["📋 Form-Based", "💬 Conversational"],
        index=0
    )

    # Route to appropriate mode
    if page == "📋 Form-Based":
        form_based_mode()
        render_prediction_result()

    elif page == "💬 Conversational":
        conversational_mode()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    **HeartGuard AI** uses machine learning to assess cardiac risk based on clinical features.

    **⚠️ Medical Disclaimer:**
    This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)

    # Debug info
    render_sidebar_debug()


if __name__ == "__main__":
    main()
