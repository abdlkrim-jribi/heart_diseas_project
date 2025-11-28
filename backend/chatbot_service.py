"""
Cerebras-powered Medical Chatbot for Feature Extraction
Uses Llama 3.3 70B for intelligent conversation + feature extraction
"""

import os
import json
import streamlit as st
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Try to load .env for local development
try:
    load_dotenv()
except:
    pass



class MedicalChatbot:
    """Conversational AI for heart disease feature collection"""

    FEATURE_PROMPTS = {
        'age': "How old are you?",
        'sex': "What is your biological sex? (Male/Female)",
        'cp': "Do you experience chest pain? If yes, what type?",
        'trestbps': "What is your resting blood pressure? (in mm Hg)",
        'chol': "What is your serum cholesterol level? (in mg/dl)",
        'fbs': "Is your fasting blood sugar greater than 120 mg/dl? (Yes/No)",
        'restecg': "What are your resting ECG results?",
        'thalach': "What is your maximum heart rate achieved during exercise?",
        'exang': "Do you experience chest pain during exercise? (Yes/No)",
        'oldpeak': "What is your ST depression induced by exercise? (0.0 to 6.2)",
        'slope': "What is the slope of your peak exercise ST segment?",
        'ca': "How many major vessels are colored by fluoroscopy? (0-3)",
        'thal': "What is your thalassemia status?"
    }

    def __init__(self):
        """Initialize the chatbot with Cerebras API"""

        # Try multiple sources for API key
        api_key = None

        # Source 1: Try Streamlit secrets (Cloud deployment)
        try:
            api_key = st.secrets.get("CEREBRAS_API_KEY")
        except Exception as e:
            print(f"Note: Could not access st.secrets: {e}")
            pass

        # Source 2: Try environment variable (.env file for local development)
        if not api_key:
            api_key = os.getenv('CEREBRAS_API_KEY')

        # Source 3: Fallback error if nothing found
        if not api_key:
            error_msg = (
                "❌ CEREBRAS_API_KEY not found!\n\n"
                "For Streamlit Cloud deployment:\n"
                "  1. Go to app Settings → Secrets\n"
                "  2. Add: CEREBRAS_API_KEY = your_key_here\n\n"
                "For local development:\n"
                "  1. Create .env file in project root\n"
                "  2. Add: CEREBRAS_API_KEY=your_key_here\n"
            )
            raise ValueError(error_msg)

        print(f"✅ CEREBRAS_API_KEY loaded successfully")

        # Initialize Cerebras client
        try:
            from cerebras.cloud.sdk import Cerebras
            print("🔄 Initializing Cerebras client...")
            self.client = Cerebras(api_key=api_key)
            print("✅ Cerebras client initialized")

        except TypeError as e:
            # Handle case where SDK has issues with proxies or kwargs
            if 'proxies' in str(e) or 'unexpected keyword argument' in str(e):
                print("⚠️ SDK TypeError detected, using alternative initialization...")
                import httpx
                from cerebras.cloud.sdk import Cerebras

                # Create clean httpx client without proxy settings
                http_client = httpx.Client(timeout=60.0)
                self.client = Cerebras(api_key=api_key, http_client=http_client)
                print("✅ Cerebras client initialized (alternative method)")
            else:
                raise e

        except Exception as e:
            error_msg = f"❌ Failed to initialize Cerebras client: {e}"
            print(error_msg)
            raise Exception(error_msg)

        # Set model
        self.model = "llama-3.3-70b"
        print(f"✅ Model set to: {self.model}")

    def create_system_prompt(self, collected_features: Dict) -> str:
        """Dynamic system prompt based on collected features"""
        missing = [f for f in self.FEATURE_PROMPTS.keys() if f not in collected_features]
        collected_list = list(collected_features.keys())
        collected_count = len(collected_features)
        missing_preview = missing[:3] if missing else ['None - ready!']

        prompt = "You are a compassionate medical AI assistant helping patients assess their cardiac health.\n\n"
        prompt += "Your Dual Role:\n"
        prompt += "1. Feature Extraction: Naturally extract these 13 medical features through conversation:\n"
        prompt += "   - age, sex, cp (chest pain), trestbps (BP), chol (cholesterol)\n"
        prompt += "   - fbs (blood sugar), restecg (ECG), thalach (max heart rate)\n"
        prompt += "   - exang (exercise angina), oldpeak (ST depression), slope (ST slope)\n"
        prompt += "   - ca (# vessels), thal (thalassemia)\n\n"
        prompt += "2. General Conversation: Answer medical questions, provide health education.\n\n"
        prompt += "Current Status:\n"
        prompt += f"- Collected: {collected_list} ({collected_count}/13)\n"
        prompt += f"- Still needed: {missing_preview}\n\n"
        prompt += "Guidelines:\n"
        prompt += "- Ask 1-2 questions at a time\n"
        prompt += "- Use simple language, explain medical terms\n"
        prompt += "- If user shares info, acknowledge it\n"
        prompt += "- When all 13 collected, congratulate and suggest prediction\n\n"
        prompt += "Be warm, professional, and medically accurate."

        return prompt

    def extract_features_from_text(self, user_message: str, collected: Dict) -> Dict:
        """Use LLM to extract features from user message"""

        collected_list = list(collected.keys())

        extraction_prompt = "Extract medical features from this user message and return ONLY valid JSON.\n\n"
        extraction_prompt += f'User message: "{user_message}"\n\n'
        extraction_prompt += f"Already collected: {collected_list}\n\n"
        extraction_prompt += "Expected features (only if explicitly mentioned):\n"
        extraction_prompt += "- age: integer 29-77\n"
        extraction_prompt += "- sex: 0 (female) or 1 (male)\n"
        extraction_prompt += "- cp: 1-4 (chest pain type)\n"
        extraction_prompt += "- trestbps: integer 94-200 (blood pressure)\n"
        extraction_prompt += "- chol: integer 126-564 (cholesterol)\n"
        extraction_prompt += "- fbs: 0 or 1 (fasting blood sugar > 120)\n"
        extraction_prompt += "- restecg: 0-2 (resting ECG)\n"
        extraction_prompt += "- thalach: integer 71-202 (max heart rate)\n"
        extraction_prompt += "- exang: 0 or 1 (exercise angina)\n"
        extraction_prompt += "- oldpeak: float 0.0-6.2 (ST depression)\n"
        extraction_prompt += "- slope: 1-3 (ST slope)\n"
        extraction_prompt += "- ca: 0-3 (major vessels)\n"
        extraction_prompt += "- thal: 3, 6, or 7 (thalassemia)\n\n"
        extraction_prompt += "Examples:\n"
        extraction_prompt += 'User: "I am 58 and male" -> {"age": 58, "sex": 1}\n'
        extraction_prompt += 'User: "cholesterol 240" -> {"chol": 240}\n'
        extraction_prompt += 'User: "no chest pain" -> {"cp": 4}\n'
        extraction_prompt += 'User: "just chatting" -> {}\n\n'
        extraction_prompt += "Return ONLY the JSON object, nothing else."

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()

            # Clean markdown code blocks
            if "```json" in content:
                parts = content.split("```json")
                if len(parts) > 1:
                    content = parts[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) > 1:
                    content = parts[1].split("```")[0].strip()

            # Extract JSON
            if "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                content = content[start:end]

            extracted = json.loads(content)

            # Validate
            valid_features = {}
            ranges = {
                'age': (29, 77), 'trestbps': (94, 200), 'chol': (126, 564),
                'thalach': (71, 202), 'oldpeak': (0.0, 6.2), 'ca': (0, 3),
                'sex': (0, 1), 'cp': (1, 4), 'fbs': (0, 1), 'restecg': (0, 2),
                'exang': (0, 1), 'slope': (1, 3), 'thal': (3, 7)
            }

            for key, value in extracted.items():
                if key not in self.FEATURE_PROMPTS or key in collected:
                    continue

                try:
                    if key in ['age', 'trestbps', 'chol', 'thalach', 'ca', 'sex',
                               'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
                        value = int(float(value))
                    elif key == 'oldpeak':
                        value = float(value)

                    if key in ranges:
                        min_val, max_val = ranges[key]
                        if min_val <= value <= max_val:
                            valid_features[key] = value
                        else:
                            print(f"Warning: {key}={value} out of range [{min_val}, {max_val}]")

                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {key}: {value} ({e})")
                    continue

            return valid_features

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {}
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}

    def chat(self, message: str, history: List[Dict], collected_features: Dict) -> Tuple[str, Dict]:
        """Main chat function"""

        new_features = self.extract_features_from_text(message, collected_features)

        messages = [{"role": "system", "content": self.create_system_prompt(collected_features)}]

        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=400
            )

            assistant_message = response.choices[0].message.content

            if new_features:
                feature_names = {
                    'age': 'age', 'sex': 'sex', 'cp': 'chest pain type',
                    'trestbps': 'blood pressure', 'chol': 'cholesterol',
                    'fbs': 'fasting blood sugar', 'restecg': 'resting ECG',
                    'thalach': 'max heart rate', 'exang': 'exercise angina',
                    'oldpeak': 'ST depression', 'slope': 'ST slope',
                    'ca': 'major vessels', 'thal': 'thalassemia'
                }

                feature_list = [feature_names.get(k, k) for k in new_features.keys()]
                acknowledgment = "\n\nNoted: " + ", ".join(feature_list)
                assistant_message += acknowledgment

            return assistant_message, new_features

        except Exception as e:
            error_msg = f"I apologize, I'm having technical difficulties: {str(e)}"
            print(f"Chat error: {e}")
            return error_msg, {}


_chatbot = None


def get_chatbot():
    """Get or create chatbot instance"""
    global _chatbot
    if _chatbot is None:
        print("Initializing Cerebras chatbot...")
        _chatbot = MedicalChatbot()
        print("Chatbot ready!")
    return _chatbot
