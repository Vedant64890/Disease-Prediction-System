"""
Advanced Disease Prediction Chatbot System
With Professional UI/UX, Advanced AI Features, and Confidence Labels
"""

import os
import json
import pickle
import hashlib
import hmac
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class ChatbotConfig:
    """Centralized configuration for the chatbot system"""
    
    APP_TITLE = "🏥 Advanced Disease Prediction AI Chatbot"
    APP_SUBTITLE = "Professional Healthcare Screening & Symptom Analysis"
    
    # Files
    USERS_FILE = "users.json"
    APPOINTMENTS_FILE = "appointments.json"
    MODEL_FILE = "advanced_model.pkl"
    PREPROCESSING_FILE = "preprocessing_data.pkl"
    CHAT_HISTORY_FILE = "chat_histories.json"
    
    # Chatbot Settings
    MAX_CHAT_HISTORY = 20
    CONFIDENCE_THRESHOLD = 0.65
    WARNING_THRESHOLD = 0.40
    
    # Color Schemes for UI
    COLORS = {
        "primary": "#0066CC",
        "success": "#00CC66",
        "warning": "#FF6600",
        "danger": "#FF3333",
        "light": "#F0F2F6",
        "dark": "#1F1F1F",
        "info": "#0099FF"
    }
    
    # Symptom Categories
    SYMPTOM_CATEGORIES = {
        "Respiratory": ["cough", "shortness_of_breath", "continuous_sneezing", "throat_irritation"],
        "Digestive": ["stomach_pain", "nausea", "vomiting", "diarrhoea", "constipation", "acidity"],
        "Neurological": ["headache", "dizziness", "memory_loss", "tremor"],
        "Joint & Muscle": ["joint_pain", "muscle_pain", "body_ache"],
        "Systemic": ["high_fever", "chills", "fatigue", "weakness"],
        "Skin": ["skin_rash", "itching", "yellowish_skin"],
        "Other": []
    }
    
    # Priority Symptoms (Urgent Indicators)
    PRIORITY_SYMPTOMS = [
        "chest_pain", "shortness_of_breath", "severe_headache",
        "loss_of_consciousness", "severe_bleeding", "difficulty_breathing"
    ]


# ============================================================================
# SECURITY & AUTHENTICATION
# ============================================================================

class AuthenticationManager:
    """Handles user authentication and password management"""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """Hash password using PBKDF2-SHA256"""
        if salt is None:
            salt_bytes = os.urandom(16)
        elif isinstance(salt, str):
            salt_bytes = bytes.fromhex(salt)
        else:
            salt_bytes = salt
        
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120000)
        return f"pbkdf2_sha256${salt_bytes.hex()}${digest.hex()}"
    
    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            if stored_hash.startswith("pbkdf2_sha256$"):
                _, salt_hex, expected_hex = stored_hash.split("$", 2)
                candidate = AuthenticationManager.hash_password(password, salt=salt_hex).split("$", 2)[2]
                return hmac.compare_digest(candidate, expected_hex)
        except Exception:
            return False
        return False
    
    @staticmethod
    def load_users() -> Dict:
        """Load user database"""
        if not os.path.exists(ChatbotConfig.USERS_FILE):
            return {}
        try:
            with open(ChatbotConfig.USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    @staticmethod
    def save_users(users: Dict) -> None:
        """Save user database"""
        with open(ChatbotConfig.USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)


# ============================================================================
# ADVANCED AI MODEL MANAGER
# ============================================================================

class AdvancedModelManager:
    """Manages advanced ML models for disease prediction"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.model_metadata = {}
        self.feature_importance = None
    
    def load_or_create_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                            feature_names: List[str], label_encoder: LabelEncoder) -> Dict:
        """Load existing model or create new one"""
        if os.path.exists(ChatbotConfig.MODEL_FILE):
            try:
                with open(ChatbotConfig.MODEL_FILE, "rb") as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get("model")
                    self.label_encoder = model_data.get("label_encoder")
                    self.feature_names = model_data.get("feature_names")
                    self.model_metadata = model_data
                    return model_data
            except Exception as e:
                st.warning(f"Could not load existing model: {e}. Creating new one...")
        
        return self.train_advanced_model(X_train, y_train, feature_names, label_encoder)
    
    def train_advanced_model(self, X_train: np.ndarray, y_train: np.ndarray,
                            feature_names: List[str], label_encoder: LabelEncoder) -> Dict:
        """Train advanced ensemble models"""
        
        st.info("🤖 Training advanced AI models...")
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_train)
        
        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            verbose=0
        )
        gb_model.fit(X_train, y_train)
        gb_predictions = gb_model.predict(X_train)
        
        # Store feature importance from Random Forest
        self.feature_importance = {
            name: float(importance)
            for name, importance in zip(feature_names, rf_model.feature_importances_)
        }
        
        # Create ensemble (weighted average)
        self.model = {
            "random_forest": rf_model,
            "gradient_boosting": gb_model,
            "weights": [0.6, 0.4]  # RF gets higher weight
        }
        
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        
        model_data = {
            "model": self.model,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "feature_importance": self.feature_importance,
            "created_at": datetime.now().isoformat()
        }
        
        # Save model
        with open(ChatbotConfig.MODEL_FILE, "wb") as f:
            pickle.dump(model_data, f)
        
        st.success("✅ Advanced AI models trained and saved!")
        return model_data
    
    def predict_with_confidence(self, symptoms: List[str]) -> Dict:
        """Make prediction with confidence scores and feature importance"""
        if not self.model or not self.feature_names:
            raise ValueError("Model not loaded")
        
        # Create feature vector
        feature_vector = {symptom: 0 for symptom in self.feature_names}
        for symptom in symptoms:
            if symptom in feature_vector:
                feature_vector[symptom] = 1
        
        X = np.array([[feature_vector[f] for f in self.feature_names]])
        
        # Get predictions from both models
        rf_pred = self.model["random_forest"].predict_proba(X)[0]
        gb_pred = self.model["gradient_boosting"].predict_proba(X)[0]
        
        # Weighted ensemble
        ensemble_proba = (
            np.array(rf_pred) * self.model["weights"][0] +
            np.array(gb_pred) * self.model["weights"][1]
        )
        
        predicted_idx = np.argmax(ensemble_proba)
        predicted_disease = self.label_encoder.classes_[predicted_idx]
        confidence = float(ensemble_proba[predicted_idx] * 100)
        
        # Top alternatives
        top_indices = np.argsort(ensemble_proba)[::-1][:3]
        alternatives = [
            {
                "disease": self.label_encoder.classes_[idx],
                "confidence": float(ensemble_proba[idx] * 100)
            }
            for idx in top_indices
        ]
        
        # Get feature importance for selected symptoms
        symptom_importance = {
            symptom: self.feature_importance.get(symptom, 0.0)
            for symptom in symptoms if symptom in self.feature_importance
        }
        
        return {
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "alternatives": alternatives,
            "all_probabilities": {
                disease: float(prob * 100)
                for disease, prob in zip(self.label_encoder.classes_, ensemble_proba)
            },
            "symptom_importance": symptom_importance,
            "symptom_count": len(symptoms),
            "requires_urgent_care": confidence > 80 and len(symptoms) >= 3
        }


# ============================================================================
# ADVANCED CHATBOT CONVERSATION ENGINE
# ============================================================================

class ConversationEngine:
    """Advanced conversation management with NLP-like capabilities"""
    
    # Intent patterns
    INTENT_PATTERNS = {
        "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
        "help": ["help", "what can you do", "assistance", "guide", "how to use"],
        "predict": ["predict", "predict now", "run prediction", "what disease", "diagnosis", "result"],
        "symptoms": ["symptom", "symptoms", "i have", "i'm experiencing"],
        "appointment": ["doctor", "appointment", "book", "consultation", "hospital"],
        "tests": ["test", "tests", "lab", "blood report", "blood test", "reports"],
        "remedy": ["remedy", "home remedy", "care", "treatment", "medicine", "what should i do"],
        "urgent": ["urgent", "emergency", "critical", "severe", "danger", "hospital"],
        "clarify": ["why", "explain", "how", "what do you mean", "clarify"],
        "thanks": ["thank you", "thanks", "appreciate", "helped me", "great"]
    }
    
    def __init__(self):
        self.conversation_history = []
        self.detected_symptoms = []
        self.conversation_context = {}
    
    def detect_intent(self, user_message: str) -> str:
        """Detect user intent from message"""
        message_lower = user_message.lower()
        
        for intent, keywords in self.INTENT_PATTERNS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return intent
        return "general"
    
    def extract_symptoms(self, user_message: str, all_symptoms: List[str]) -> List[str]:
        """Extract symptoms mentioned in user message"""
        message_lower = user_message.lower().replace("_", " ")
        detected = []
        aliases = {
            "high_fever": ["fever", "high temperature"],
            "muscle_pain": ["body pain", "body ache", "muscle ache"],
            "body_ache": ["body pain", "body ache", "muscle ache"],
            "stomach_pain": ["abdominal pain", "belly pain", "stomach ache"],
            "shortness_of_breath": ["breathlessness", "short of breath", "difficulty breathing"],
            "chest_pain": ["pain in chest", "chest tightness"],
            "vomiting": ["throwing up", "puking"],
            "diarrhoea": ["diarrhea", "loose motion", "loose stools"],
            "fatigue": ["tired", "tiredness", "exhausted"],
            "cough": ["coughing", "dry cough"],
            "headache": ["head pain"],
        }
        
        for symptom in all_symptoms:
            symptom_display = symptom.replace("_", " ")
            symptom_terms = [symptom_display] + aliases.get(symptom, [])
            if any(term in message_lower for term in symptom_terms):
                detected.append(symptom)
        
        return detected

    @staticmethod
    def _format_symptoms(symptoms: List[str]) -> str:
        if not symptoms:
            return "none yet"
        return ", ".join(symptom.replace("_", " ").title() for symptom in symptoms)
    
    def generate_contextual_response(self, intent: str, detected_symptoms: List[str],
                                    prediction: Optional[Dict] = None) -> str:
        """Generate contextual response based on intent and context"""
        symptom_text = self._format_symptoms(detected_symptoms)
        symptom_count = len(detected_symptoms)
        
        responses = {
            "greeting": (
                "Hi, I can help you screen symptoms and prepare them for prediction. "
                "Tell me what you are feeling in one sentence, for example: fever, cough, headache, and body pain."
            ),
            
            "help": (
                "I can help with:\n"
                "- Detecting symptoms from your message.\n"
                "- Preparing symptoms for disease prediction.\n"
                "- Explaining confidence scores and next steps.\n"
                "- Suggesting doctor consultation, home-care topics, and warning signs.\n"
                "- Pointing you to urgent care when symptoms look dangerous.\n\n"
                f"Current saved symptoms: {symptom_text}."
            ),
            
            "symptoms": (
                f"I have saved these symptoms: {symptom_text}. "
                + (
                    "Add one or two more symptoms if you have them, or run the predictor when ready."
                    if symptom_count < 3
                    else "You can run a prediction now, ask about possible tests, or book a doctor review."
                )
            ),
            
            "predict": (
                f"Current saved symptoms: {symptom_text}. Use the Symptoms or Analyzer page to run the prediction when ready. "
                "The result is screening support only, not a confirmed diagnosis."
                if not prediction
                else f"""
**Prediction Result**
Primary Disease: **{prediction['predicted_disease']}**
Confidence Level: **{prediction['confidence']:.1f}%**

Alternative Possibilities:
{chr(10).join(f"- {alt['disease']}: {alt['confidence']:.1f}%" for alt in prediction['alternatives'])}

This is a screening tool only and should not replace professional medical advice.
"""
            ),
            
            "appointment": (
                "For booking, first run a prediction so the app can suggest the right specialist. "
                "Then use the appointment section to select doctor, hospital or clinic, date, time, and consultation mode."
            ),

            "tests": (
                "Lab tests depend on the predicted condition and a doctor's review. "
                f"Current saved symptoms: {symptom_text}. Run a prediction first, then ask about reports or tests."
            ),

            "remedy": (
                "For home care, focus on rest, fluids, light food if tolerated, and monitoring symptoms. "
                "Avoid self-medicating for serious or worsening symptoms. Run a prediction or consult a doctor for condition-specific advice."
            ),

            "clarify": (
                "This chatbot detects symptom words from your message, saves them, and uses the trained model to support disease screening. "
                "It cannot confirm a diagnosis; it helps you decide what to review next with a clinician."
            ),

            "urgent": (
                "**Urgent alert:** Please seek immediate medical attention or contact emergency services, especially for chest pain, "
                "breathing trouble, fainting, confusion, severe weakness, repeated vomiting, or rapidly worsening symptoms."
            ),
            
            "thanks": "You're welcome. You can keep asking about symptoms, prediction, tests, home care, warning signs, or doctor booking.",
            
            "general": (
                "I can help best when you describe symptoms directly, such as: fever and cough, headache and body pain, "
                "stomach pain with vomiting, or chest pain with breathlessness."
            )
        }
        
        return responses.get(intent, responses["general"])
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self, limit: int = ChatbotConfig.MAX_CHAT_HISTORY) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history[-limit:]


# ============================================================================
# SESSION STATE MANAGER
# ============================================================================

class SessionStateManager:
    """Manages Streamlit session state"""
    
    @staticmethod
    def init_session():
        """Initialize session state"""
        defaults = {
            "logged_in": False,
            "current_user": None,
            "conversation_engine": ConversationEngine(),
            "model_manager": AdvancedModelManager(),
            "current_prediction": None,
            "selected_symptoms": [],
            "show_advanced_analytics": False,
            "chat_input_key": 0,
            "app_page": "Dashboard"
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def save_chat_history(username: str) -> None:
        """Save chat history to file"""
        try:
            histories = {}
            if os.path.exists(ChatbotConfig.CHAT_HISTORY_FILE):
                with open(ChatbotConfig.CHAT_HISTORY_FILE, "r") as f:
                    histories = json.load(f)
            
            histories[username] = st.session_state.conversation_engine.get_history()
            
            with open(ChatbotConfig.CHAT_HISTORY_FILE, "w") as f:
                json.dump(histories, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Error saving chat history: {e}")
    
    @staticmethod
    def load_chat_history(username: str) -> List[Dict]:
        """Load chat history from file"""
        try:
            if os.path.exists(ChatbotConfig.CHAT_HISTORY_FILE):
                with open(ChatbotConfig.CHAT_HISTORY_FILE, "r") as f:
                    histories = json.load(f)
                    return histories.get(username, [])
        except Exception:
            pass
        return []


# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Advanced UI/UX components"""
    
    @staticmethod
    def render_confidence_gauge(confidence: float, disease: str) -> None:
        """Render confidence gauge with Plotly"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"AI Confidence for {disease}"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "#FF6B6B"},
                    {'range': [40, 65], 'color': "#FFA500"},
                    {'range': [65, 80], 'color': "#4ECDC4"},
                    {'range': [80, 100], 'color': "#45B7D1"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=400, margin=dict(l=50, r=50, t=80, b=50))
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_prediction_breakdown(prediction: Dict) -> None:
        """Render detailed prediction breakdown"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Primary Disease", prediction["predicted_disease"], 
                     f"{prediction['confidence']:.1f}%")
        
        with col2:
            st.metric("Confidence Level",
                     "HIGH" if prediction["confidence"] > 75 else "MEDIUM" if prediction["confidence"] > 50 else "LOW",
                     f"{prediction['confidence']:.1f}%")
        
        with col3:
            st.metric("Symptoms Analyzed", prediction["symptom_count"])
    
    @staticmethod
    def render_alternatives_chart(alternatives: List[Dict]) -> None:
        """Render alternatives as horizontal bar chart"""
        df_alt = pd.DataFrame(alternatives)
        fig = px.bar(
            df_alt,
            x="confidence",
            y="disease",
            orientation='h',
            title="Top Alternative Diagnoses",
            labels={"confidence": "AI Confidence (%)", "disease": "Disease"},
            color="confidence",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_symptom_importance(symptom_importance: Dict) -> None:
        """Render symptom importance as pie chart"""
        if not symptom_importance:
            st.info("No symptom importance data available")
            return
        
        df_importance = pd.DataFrame(
            list(symptom_importance.items()),
            columns=["Symptom", "Importance"]
        )
        
        fig = px.pie(
            df_importance,
            values="Importance",
            names="Symptom",
            title="Symptom Importance in Prediction"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_symptom_categories(all_symptoms: List[str]) -> Dict[str, List[str]]:
        """Render organized symptom categories"""
        categorized = {}
        for category, symptoms in ChatbotConfig.SYMPTOM_CATEGORIES.items():
            categorized[category] = [s for s in symptoms if s in all_symptoms]
        
        # Add uncategorized
        categorized_flat = [s for symptoms in categorized.values() for s in symptoms]
        uncategorized = [s for s in all_symptoms if s not in categorized_flat]
        if uncategorized:
            categorized["Other"] = uncategorized
        
        return categorized


# ============================================================================
# UTILITIES
# ============================================================================

def format_symptom_name(symptom: str) -> str:
    """Format symptom name for display"""
    return symptom.replace("_", " ").title()


def check_urgent_symptoms(symptoms: List[str]) -> bool:
    """Check if any urgent symptoms are present"""
    for symptom in symptoms:
        if any(urgent in symptom.lower() for urgent in ChatbotConfig.PRIORITY_SYMPTOMS):
            return True
    return False


if __name__ == "__main__":
    # Test imports
    print("✅ Advanced Chatbot System Loaded Successfully")
