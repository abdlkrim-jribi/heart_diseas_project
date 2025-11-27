# 🫀 HeartGuard AI - Cardiac Risk Assessment System

AI-powered heart disease prediction using Machine Learning and Conversational AI.

## Features
- 📋 **Form-Based Mode**: Manual input of 13 medical features
- 💬 **Conversational Mode**: Natural language chatbot powered by Cerebras LLM
- 🎯 **Prediction**: Gaussian Naive Bayes classifier with risk assessment

## Tech Stack
- **Frontend**: Streamlit
- **ML Model**: Scikit-learn (GaussianNB)
- **LLM**: Cerebras Llama 3.3 70B
- **Python**: 3.10+

## Local Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with `CEREBRAS_API_KEY=your_key_here`
4. Run: `streamlit run backend/app.py`

## Medical Disclaimer
⚠️ This tool is for educational purposes only. Not a substitute for professional medical advice.
