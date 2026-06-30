import base64
import calendar
import hashlib
import hmac
import html
import json
import os
import pickle
import re
from datetime import date, datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
import warnings
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

ASSISTANT_IMAGE_PATH = os.path.join("assets", "chatbot_launcher.png")
PATIENT_AVATAR_PATH = os.path.join("assets", "patient_avatar.png")
THEME_OPTIONS = ("Light", "Dark")
ASSISTANT_WELCOME_MESSAGE = ""
ASSISTANT_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ASSISTANT_DEFAULT_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{ASSISTANT_DEFAULT_MODEL}:generateContent"
ASSISTANT_MAX_HISTORY = 12
ASSISTANT_MIN_PREDICTION_CONFIDENCE = 45.0
EXTERNAL_MEDICAL_EVIDENCE_ENABLED = os.getenv("EXTERNAL_MEDICAL_EVIDENCE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"
NLM_CONDITIONS_SEARCH_URL = "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search"
NHS_CONTENT_API_BASE_URL = os.getenv("NHS_CONTENT_API_BASE_URL", "https://api.service.nhs.uk/nhs-website-content").rstrip("/")
EXTERNAL_EVIDENCE_USER_AGENT = "DiseasePredictionDashboard/1.0"
MAPBOX_GEOCODING_URL = "https://api.mapbox.com/search/geocode/v6/forward"
HERE_GEOCODE_URL = "https://geocode.search.hereapi.com/v1/geocode"
HERE_DISCOVER_URL = "https://discover.search.hereapi.com/v1/discover"
ASSISTANT_SUPPRESSED_LIVE_ERROR_PATTERNS = (
    r"(?:this|the)\s+model\s+is\s+currently\s+experiencing\s+high\s+demand\.?\s*(?:please\s+try\s+again\s+later\.?)?",
    r"(?:i\s+am|i'm|im)\s+currently\s+experiencing\s+high\s+demand\.?\s*(?:please\s+try\s+again\s+later\.?)?",
    r"(?:gemini|model|service|api)\s+(?:is\s+)?(?:currently\s+)?(?:overloaded|unavailable|busy)\.?\s*(?:please\s+try\s+again\s+later\.?)?",
)
ASSISTANT_GREETING_TERMS = ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")
ASSISTANT_HELP_TERMS = ("help", "what can you do", "assist", "chatbot", "assistant")
ASSISTANT_THANKS_TERMS = ("thank you", "thanks", "thx")
ASSISTANT_BOOKING_TERMS = ("doctor", "appointment", "book", "consultation", "hospital")
ASSISTANT_TEST_TERMS = ("test", "tests", "lab", "labs", "blood report", "blood test", "diagnostic", "reports")
ASSISTANT_AI_TERMS = ("api key", "ai key", "gemini", "google ai", "model", "live ai")
ASSISTANT_PREDICT_TERMS = ("prediction", "predict", "screening", "result")
ASSISTANT_REMEDY_TERMS = ("remedy", "home remedy", "care", "treatment", "what should i do")
ASSISTANT_MEDICINE_ADVICE_TERMS = (
    "medicine",
    "medicines",
    "tablet",
    "tablets",
    "drug",
    "drugs",
    "dose",
    "dosage",
    "continue medicine",
    "continue medicines",
    "continue tablet",
    "continue tablets",
    "should i continue",
    "discontinue",
    "stop taking",
    "stop medicine",
    "prescribe",
    "prescription",
    "paracetamol",
    "dolo",
    "crocin",
    "ibuprofen",
    "aspirin",
    "cetirizine",
    "antibiotic",
    "cough syrup",
    "nasal spray",
    "antacid",
    "ors",
    "inhaler",
    "steroid",
)
ASSISTANT_WARNING_TERMS = ("warning", "warnings", "danger", "emergency", "urgent", "red flag", "red flags")
ASSISTANT_DISEASE_TERMS = ("which disease", "what disease", "likely disease", "what do i have")
ASSISTANT_SPECIALIST_TERMS = ("specialist", "which doctor", "what doctor", "consult", "who should i see")
HEALTHCARE_SEARCH_RADII_M = (15000,)
HEALTHCARE_MAX_RESULTS = 80
ASSISTANT_STARTER_PROMPTS = (
    "I have blocked nose",
    "I have fever since yesterday",
    "I have cough with throat pain",
    "I have headache and body pain",
)
ASSISTANT_FOLLOWUP_PROMPTS = (
    "2 days, no fever, sneezing yes",
    "I took paracetamol",
    "No breathing problem, cough yes",
    "Predict when ready",
)
ASSISTANT_LOAD_TRIGGERS = (
    "use these symptoms",
    "use symptoms",
    "load symptoms",
    "send symptoms",
    "select these symptoms",
)
ASSISTANT_PREDICT_TRIGGERS = (
    "predict now",
    "run prediction",
    "show prediction",
    "screen now",
    "check prediction now",
    "predict from this",
)
CITY_OPTIONS = tuple(
    sorted(
        {
            "Agartala, Tripura",
            "Agra, Uttar Pradesh",
            "Ahmedabad, Gujarat",
            "Ajmer, Rajasthan",
            "Aligarh, Uttar Pradesh",
            "Allahabad, Uttar Pradesh",
            "Amravati, Maharashtra",
            "Amritsar, Punjab",
            "Anand, Gujarat",
            "Asansol, West Bengal",
            "Aurangabad, Maharashtra",
            "Bareilly, Uttar Pradesh",
            "Belagavi, Karnataka",
            "Bengaluru, Karnataka",
            "Bhavnagar, Gujarat",
            "Bhilai, Chhattisgarh",
            "Bhopal, Madhya Pradesh",
            "Bhubaneswar, Odisha",
            "Bikaner, Rajasthan",
            "Bilaspur, Chhattisgarh",
            "Chandigarh, Chandigarh",
            "Chennai, Tamil Nadu",
            "Coimbatore, Tamil Nadu",
            "Cuttack, Odisha",
            "Dehradun, Uttarakhand",
            "Delhi, Delhi",
            "Dhanbad, Jharkhand",
            "Durgapur, West Bengal",
            "Faridabad, Haryana",
            "Gandhinagar, Gujarat",
            "Ghaziabad, Uttar Pradesh",
            "Bopal, Ahmedabad, Gujarat",
            "Chandkheda, Ahmedabad, Gujarat",
            "Ghatlodia, Ahmedabad, Gujarat",
            "Gota, Ahmedabad, Gujarat",
            "Gorakhpur, Uttar Pradesh",
            "Gurugram, Haryana",
            "Guwahati, Assam",
            "Gwalior, Madhya Pradesh",
            "Hisar, Haryana",
            "Hubballi, Karnataka",
            "Hyderabad, Telangana",
            "Indore, Madhya Pradesh",
            "Jabalpur, Madhya Pradesh",
            "Jaipur, Rajasthan",
            "Jalandhar, Punjab",
            "Jamnagar, Gujarat",
            "Jamshedpur, Jharkhand",
            "Jhansi, Uttar Pradesh",
            "Jodhpur, Rajasthan",
            "Junagadh, Gujarat",
            "Kanpur, Uttar Pradesh",
            "Kochi, Kerala",
            "Kolhapur, Maharashtra",
            "Kolkata, West Bengal",
            "Kota, Rajasthan",
            "Kozhikode, Kerala",
            "Lucknow, Uttar Pradesh",
            "Ludhiana, Punjab",
            "Madurai, Tamil Nadu",
            "Mangaluru, Karnataka",
            "Meerut, Uttar Pradesh",
            "Mohali, Punjab",
            "Moradabad, Uttar Pradesh",
            "Mumbai, Maharashtra",
            "Mysuru, Karnataka",
            "Nagpur, Maharashtra",
            "Nashik, Maharashtra",
            "Navi Mumbai, Maharashtra",
            "Maninagar, Ahmedabad, Gujarat",
            "Naranpura, Ahmedabad, Gujarat",
            "Naroda, Ahmedabad, Gujarat",
            "Navrangpura, Ahmedabad, Gujarat",
            "New Delhi, Delhi",
            "Nikol, Ahmedabad, Gujarat",
            "Noida, Uttar Pradesh",
            "Panaji, Goa",
            "Patiala, Punjab",
            "Patna, Bihar",
            "Pimpri-Chinchwad, Maharashtra",
            "Pune, Maharashtra",
            "Raipur, Chhattisgarh",
            "Rajkot, Gujarat",
            "Ranchi, Jharkhand",
            "Satellite, Ahmedabad, Gujarat",
            "Rohtak, Haryana",
            "Saharanpur, Uttar Pradesh",
            "Salem, Tamil Nadu",
            "Shillong, Meghalaya",
            "Shimla, Himachal Pradesh",
            "Siliguri, West Bengal",
            "Solapur, Maharashtra",
            "Srinagar, Jammu and Kashmir",
            "Surat, Gujarat",
            "Thane, Maharashtra",
            "Thaltej, Ahmedabad, Gujarat",
            "Thiruvananthapuram, Kerala",
            "Tiruchirappalli, Tamil Nadu",
            "Tirupati, Andhra Pradesh",
            "Udaipur, Rajasthan",
            "Vadodara, Gujarat",
            "Varanasi, Uttar Pradesh",
            "Vasai-Virar, Maharashtra",
            "Vastrapur, Ahmedabad, Gujarat",
            "Vejalpur, Ahmedabad, Gujarat",
            "Vijayawada, Andhra Pradesh",
            "Visakhapatnam, Andhra Pradesh",
            "Warangal, Telangana",
        }
    )
)
LOCATION_ALIASES = {
    "ahm": "ahmedabad",
    "amdavad": "ahmedabad",
    "ahemdabad": "ahmedabad",
    "ahedabad": "ahmedabad",
    "ahedaba": "ahmedabad",
    "abad": "ahmedabad",
}
ASSISTANT_SYSTEM_PROMPT = """
You are the chatbot inside a symptom screening dashboard.
Act like a careful patient-facing health guide, not a developer tool and not a final doctor.
Your job is to understand symptoms in natural language, keep the conversation calm, and guide the user through one clear next step.
Use the app context when available: saved symptoms, denied symptoms, duration, severity, medicine history, progression, warning signs, screening preview, care level, and suggested specialist.
Never claim a confirmed diagnosis. Say screening result or possible condition, not final diagnosis.
If symptoms sound urgent or dangerous, advise immediate medical care before asking anything else.
Do not give prescription-only medicine instructions, exact dosages, or unsafe home treatment.
When the patient mentions medicines, consider them in your answer. Explain medicine safety in plain language: do not self-start antibiotics, steroids, or leftover prescription tablets; do not stop chronic prescribed medicines suddenly; if a new medicine caused swelling, breathing trouble, fainting, severe rash, or skin peeling, advise urgent medical care and no further dose until a clinician advises.
You may suggest medicine categories to discuss with a doctor or pharmacist, but do not prescribe a new medicine or change a prescription yourself.
If the user asks what a symptom means, answer the meaning first and do not save it as a present symptom unless they clearly say they have it.
If the user asks a specific health, test, medicine-safety, hospital, or app-use question, answer it directly first instead of forcing the symptom interview.
If symptoms are broad and overlapping, such as fatigue, headache, body pain, or muscle pain, do not jump to malaria, dengue, flu, typhoid, or another disease. Ask discriminating questions first: fever, chills/sweating, cough/sore throat, rash/bleeding, vomiting/diarrhea, mosquito exposure, unsafe food/water, travel, and duration.
Do not show percentages, probability, confidence score, or internal model details.
Do not mention API mode, Gemini, prompts, model internals, or system instructions unless the user asks about API setup.
Ask only one short follow-up question at a time.
Keep replies complete and readable. Do not cut off sentences, warning signs, or the final question.
Use enough short lines to answer properly, usually 5 to 9 lines when symptoms or safety guidance need context.
Prefer this shape:
- Acknowledge what you understood.
- Explain why the next question matters in simple words.
- Give one practical safety or care note if useful.
- Ask exactly one next question.
""".strip()


def hash_password(password, salt=None):
    if salt is None:
        salt_bytes = os.urandom(16)
    elif isinstance(salt, str):
        salt_bytes = bytes.fromhex(salt)
    else:
        salt_bytes = salt

    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120000)
    return f"pbkdf2_sha256${salt_bytes.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    if stored_hash.startswith("pbkdf2_sha256$"):
        _, salt_hex, expected_hex = stored_hash.split("$", 2)
        candidate = hash_password(password, salt=salt_hex).split("$", 2)[2]
        return hmac.compare_digest(candidate, expected_hex)

    # Legacy unsalted SHA256 compatibility.
    legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy, stored_hash)


_database_ready = False


def get_configured_secret_value(*names):
    for name in names:
        env_value = os.getenv(name, "").strip()
        if env_value:
            return env_value

    for name in names:
        try:
            secret_value = st.secrets.get(name, "")
        except Exception:
            secret_value = ""
        if secret_value is not None and str(secret_value).strip():
            return str(secret_value).strip()
    return ""


def get_mysql_database_config():
    host = get_configured_secret_value("MYSQL_HOST")
    database = get_configured_secret_value("MYSQL_DATABASE", "MYSQL_DB")
    user = get_configured_secret_value("MYSQL_USER", "MYSQL_USERNAME")
    password = get_configured_secret_value("MYSQL_PASSWORD")
    if not all([host, database, user, password]):
        return {}

    port_value = get_configured_secret_value("MYSQL_PORT") or "3306"
    try:
        port = int(port_value)
    except (TypeError, ValueError):
        port = 3306

    return {
        "host": host,
        "port": port,
        "database": database,
        "user": user,
        "password": password,
    }


def get_mysql_user_database_config():
    return get_mysql_database_config()


def mysql_storage_enabled():
    return bool(get_mysql_database_config())


def mysql_user_storage_enabled():
    return mysql_storage_enabled()


def get_mysql_connection():
    config = get_mysql_database_config()
    if not config:
        raise RuntimeError("MySQL storage is not configured.")

    try:
        import mysql.connector
    except ImportError as exc:
        raise RuntimeError(
            "MySQL storage is configured, but mysql-connector-python is not installed. "
            "Run: pip install mysql-connector-python"
        ) from exc

    try:
        return mysql.connector.connect(**config)
    except Exception as exc:
        raise RuntimeError(f"Could not connect to the MySQL database: {exc}") from exc


def get_mysql_user_connection():
    return get_mysql_connection()


def initialize_mysql_storage():
    if not mysql_storage_enabled():
        return

    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(120) PRIMARY KEY,
                first_name VARCHAR(120) NOT NULL,
                last_name VARCHAR(120) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id VARCHAR(80) PRIMARY KEY,
                username VARCHAR(120) NOT NULL,
                patient_name VARCHAR(160) NOT NULL,
                predicted_disease VARCHAR(160) NOT NULL,
                specialist VARCHAR(160) NOT NULL,
                doctor_name VARCHAR(180) NOT NULL,
                hospital VARCHAR(220) NOT NULL,
                consultation_mode VARCHAR(80) NOT NULL,
                appointment_date VARCHAR(20) NOT NULL,
                appointment_slot VARCHAR(80) NOT NULL,
                status VARCHAR(40) NOT NULL DEFAULT 'Booked',
                booked_at VARCHAR(40) NOT NULL,
                symptoms_json LONGTEXT NOT NULL,
                reason TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_appointments_username (username),
                INDEX idx_appointments_slot (doctor_name, appointment_date, appointment_slot, status)
            ) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lab_appointments (
                lab_appointment_id VARCHAR(80) PRIMARY KEY,
                username VARCHAR(120) NOT NULL,
                patient_name VARCHAR(160) NOT NULL,
                predicted_disease VARCHAR(160) NOT NULL,
                lab_name VARCHAR(220) NOT NULL,
                lab_tests_json LONGTEXT NOT NULL,
                total_amount DOUBLE NOT NULL DEFAULT 0,
                payment_method VARCHAR(80) NOT NULL,
                payment_status VARCHAR(80) NOT NULL,
                appointment_date VARCHAR(20) NOT NULL,
                appointment_slot VARCHAR(80) NOT NULL,
                status VARCHAR(40) NOT NULL DEFAULT 'Booked',
                booked_at VARCHAR(40) NOT NULL,
                symptoms_json LONGTEXT NOT NULL,
                payment_reference VARCHAR(180) NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_lab_appointments_username (username),
                INDEX idx_lab_appointments_slot (lab_name, appointment_date, appointment_slot, status)
            ) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def initialize_mysql_user_storage():
    initialize_mysql_storage()


def load_users_from_mysql():
    conn = get_mysql_user_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT username, first_name, last_name, password_hash
            FROM users
            ORDER BY username
            """
        )
        rows = cursor.fetchall()
        return {
            row["username"]: {
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "username": row["username"],
                "password_hash": row["password_hash"],
            }
            for row in rows
        }
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def save_users_to_mysql(users):
    conn = get_mysql_user_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        for username, user in users.items():
            username_key = str(user.get("username") or username).strip().lower()
            if not username_key:
                continue
            cursor.execute(
                """
                INSERT INTO users (
                    username, first_name, last_name, password_hash, updated_at
                ) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON DUPLICATE KEY UPDATE
                    first_name = VALUES(first_name),
                    last_name = VALUES(last_name),
                    password_hash = VALUES(password_hash),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    username_key,
                    str(user.get("first_name", "")).strip(),
                    str(user.get("last_name", "")).strip(),
                    str(user.get("password_hash", "")).strip(),
                ),
            )
        conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def ensure_database_ready():
    global _database_ready
    if _database_ready:
        return
    if not mysql_storage_enabled():
        raise RuntimeError(
            "MySQL storage is required. Configure MYSQL_HOST, MYSQL_PORT, "
            "MYSQL_DATABASE, MYSQL_USER, and MYSQL_PASSWORD in .streamlit/secrets.toml."
        )
    initialize_mysql_storage()
    _database_ready = True


def load_users():
    ensure_database_ready()
    return load_users_from_mysql()


def save_users(users):
    ensure_database_ready()
    save_users_to_mysql(users)


def get_model_artifact_signature():
    artifact_paths = ("best_model.pkl", "preprocessing_data.pkl")
    return tuple(
        (path, os.path.getmtime(path), os.path.getsize(path))
        for path in artifact_paths
    )


@st.cache_resource
def load_model_and_data(artifact_signature):
    # artifact_signature is used by Streamlit cache to reload after retraining.
    with open("best_model.pkl", "rb") as f:
        model_metadata = pickle.load(f)
    with open("preprocessing_data.pkl", "rb") as f:
        preprocessing_data = pickle.load(f)
    return model_metadata, preprocessing_data


def get_model_metric(model_metadata, preprocessing_data, metric_name):
    stored_value = model_metadata.get(metric_name)
    if stored_value is not None:
        return float(stored_value)

    validation_metrics = compute_validation_metrics(model_metadata, preprocessing_data)
    return float(validation_metrics.get(metric_name, 0.0))


@st.cache_data(show_spinner=False)
def compute_validation_metrics(_model_metadata, _preprocessing_data):
    model = _model_metadata["model"]
    X_val = _preprocessing_data.get("X_val")
    y_val = _preprocessing_data.get("y_val")
    if X_val is None or y_val is None or len(y_val) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    y_pred = np.asarray(model.predict(X_val))
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.rint(y_pred).astype(int)
    class_count = len(_model_metadata["label_encoder"].classes_)
    y_pred = np.clip(y_pred.astype(int), 0, class_count - 1)

    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
    }


@st.cache_data
def load_symptom_remedies(path="csv_files/home_remedies_training.csv"):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["symptom", "home_remedies"])
    df = pd.read_csv(path)
    required_cols = {"symptom", "home_remedies"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame(columns=["symptom", "home_remedies"])
    df["symptom"] = df["symptom"].astype(str).str.strip().str.lower()
    return df


def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "ui_theme" not in st.session_state or st.session_state.ui_theme not in THEME_OPTIONS:
        st.session_state.ui_theme = "Light"
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "prediction_context" not in st.session_state:
        st.session_state.prediction_context = None
    if "prediction_symptoms" not in st.session_state:
        st.session_state.prediction_symptoms = []
    if "symptom_chat_history" not in st.session_state:
        st.session_state.symptom_chat_history = []
    if "assistant_detected_symptoms" not in st.session_state:
        st.session_state.assistant_detected_symptoms = []
    if "assistant_clinical_notes" not in st.session_state:
        st.session_state.assistant_clinical_notes = {}
    if "assistant_pending_question" not in st.session_state:
        st.session_state.assistant_pending_question = {}
    if "assistant_asked_question_keys" not in st.session_state:
        st.session_state.assistant_asked_question_keys = []
    if "assistant_trigger_predict" not in st.session_state:
        st.session_state.assistant_trigger_predict = False
    if "assistant_panel_open" not in st.session_state:
        st.session_state.assistant_panel_open = False
    if "provider_search_results" not in st.session_state:
        st.session_state.provider_search_results = None
    if "provider_search_context" not in st.session_state:
        st.session_state.provider_search_context = ""
    if "selected_booking_provider" not in st.session_state:
        st.session_state.selected_booking_provider = None
    if "selected_booking_provider_context" not in st.session_state:
        st.session_state.selected_booking_provider_context = ""
    if "provider_autofill_notice" not in st.session_state:
        st.session_state.provider_autofill_notice = ""
    if "provider_location_fingerprint" not in st.session_state:
        st.session_state.provider_location_fingerprint = ""
    if "lab_provider_search_results" not in st.session_state:
        st.session_state.lab_provider_search_results = None
    if "lab_provider_search_context" not in st.session_state:
        st.session_state.lab_provider_search_context = ""
    if "selected_lab_provider" not in st.session_state:
        st.session_state.selected_lab_provider = None
    if "selected_lab_provider_context" not in st.session_state:
        st.session_state.selected_lab_provider_context = ""
    if "lab_provider_autofill_notice" not in st.session_state:
        st.session_state.lab_provider_autofill_notice = ""
    if "lab_provider_location_fingerprint" not in st.session_state:
        st.session_state.lab_provider_location_fingerprint = ""
    if "lab_test_cart" not in st.session_state:
        st.session_state.lab_test_cart = []
    if "lab_booking_notice" not in st.session_state:
        st.session_state.lab_booking_notice = ""
    if "assistant_live_mode" in st.session_state:
        del st.session_state["assistant_live_mode"]
    if "assistant_api_key" in st.session_state:
        del st.session_state["assistant_api_key"]
    if "assistant_model_name" in st.session_state:
        del st.session_state["assistant_model_name"]


def get_active_ui_theme():
    theme = st.session_state.get("ui_theme", THEME_OPTIONS[0])
    if theme not in THEME_OPTIONS:
        return THEME_OPTIONS[0]
    return theme


def toggle_ui_theme():
    st.session_state.ui_theme = "Dark" if get_active_ui_theme() == "Light" else "Light"


def render_theme_toggle():
    current_theme = get_active_ui_theme()
    label = "Dark" if current_theme == "Light" else "Light"
    icon = ":material/dark_mode:" if current_theme == "Light" else ":material/lightbulb:"
    st.button(
        label,
        icon=icon,
        icon_position="right",
        key="ui_theme_toggle",
        help=f"Switch to {label} theme",
        on_click=toggle_ui_theme,
        width="content",
    )
    inject_theme_toggle_runtime_styles(current_theme)


def inject_theme_toggle_runtime_styles(active_theme):
    if active_theme == "Dark":
        button_background = """
            linear-gradient(145deg, rgba(27, 38, 52, 0.98), rgba(12, 18, 27, 0.96))
        """
        border_color = "rgba(148, 163, 184, 0.30)"
        text_color = "#dffcf5"
        shadow = (
            "0 10px 24px rgba(0, 0, 0, 0.34), "
            "inset 0 1px 0 rgba(255, 255, 255, 0.10)"
        )
        sheen = "rgba(255, 255, 255, 0.18)"
        sidebar_background = "linear-gradient(145deg, #162231, #0b1118)"
    else:
        button_background = "linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(240, 245, 255, 0.9))"
        border_color = "rgba(90, 99, 216, 0.18)"
        text_color = "var(--patient-blue-dark)"
        shadow = "0 12px 26px rgba(64, 74, 130, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.78)"
        sheen = "rgba(255, 255, 255, 0.74)"
        sidebar_background = "linear-gradient(145deg, #ffffff, #eef4ff)"

    st.markdown(
        f"""
        <style>
        html body #dp-theme-floating-toggle#dp-theme-floating-toggle {{
            display: none !important;
            pointer-events: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }}

        html body div[class*="st-key-ui_theme_toggle"],
        html body div[data-testid="stElementContainer"]:has(button[title="Switch to Light theme"]),
        html body div[data-testid="stElementContainer"]:has(button[aria-label="Switch to Light theme"]),
        html body div[data-testid="stElementContainer"]:has(button[title="Switch to Dark theme"]),
        html body div[data-testid="stElementContainer"]:has(button[aria-label="Switch to Dark theme"]) {{
            display: flex !important;
            justify-content: flex-end !important;
            width: 100% !important;
            max-width: none !important;
            margin: 0 0 0.45rem auto !important;
            pointer-events: auto !important;
            z-index: 20 !important;
            isolation: isolate !important;
        }}

        html body div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"],
        html body div[data-testid="stElementContainer"]:has(button[title="Switch to Light theme"]) div[data-testid="stButton"],
        html body div[data-testid="stElementContainer"]:has(button[aria-label="Switch to Light theme"]) div[data-testid="stButton"],
        html body div[data-testid="stElementContainer"]:has(button[title="Switch to Dark theme"]) div[data-testid="stButton"],
        html body div[data-testid="stElementContainer"]:has(button[aria-label="Switch to Dark theme"]) div[data-testid="stButton"] {{
            display: flex !important;
            justify-content: flex-end !important;
            width: auto !important;
            min-width: 0 !important;
            max-width: none !important;
        }}

        html body div[class*="st-key-ui_theme_toggle"] button,
        html body button[title="Switch to Light theme"],
        html body button[aria-label="Switch to Light theme"],
        html body button[title="Switch to Dark theme"],
        html body button[aria-label="Switch to Dark theme"] {{
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.22rem !important;
            width: 78px !important;
            min-width: 78px !important;
            max-width: 78px !important;
            height: 30px !important;
            min-height: 30px !important;
            max-height: 30px !important;
            padding: 0 0.48rem !important;
            border-radius: 999px !important;
            overflow: hidden !important;
            background: {button_background} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
            box-shadow: {shadow} !important;
            pointer-events: auto !important;
            cursor: pointer !important;
        }}

        html body div[class*="st-key-ui_theme_toggle"] button::before,
        html body button[title="Switch to Light theme"]::before,
        html body button[aria-label="Switch to Light theme"]::before,
        html body button[title="Switch to Dark theme"]::before,
        html body button[aria-label="Switch to Dark theme"]::before {{
            content: "";
            position: absolute;
            inset: -32% 54% -32% -35%;
            background: linear-gradient(115deg, transparent, {sheen}, transparent);
            transform: translateX(-130%);
            animation: themeButtonSheen 2.8s var(--theme-transition-ease) infinite;
            pointer-events: none;
        }}

        html body div[class*="st-key-ui_theme_toggle"] button p,
        html body div[class*="st-key-ui_theme_toggle"] button span,
        html body button[title="Switch to Light theme"] *,
        html body button[aria-label="Switch to Light theme"] *,
        html body button[title="Switch to Dark theme"] *,
        html body button[aria-label="Switch to Dark theme"] * {{
            position: relative !important;
            z-index: 1 !important;
            color: inherit !important;
            fill: currentColor !important;
            font-size: 0.72rem !important;
            font-weight: 900 !important;
            line-height: 1 !important;
            pointer-events: none !important;
        }}

        html body [data-testid="stSidebarCollapsedControl"],
        html body button[data-testid="stSidebarCollapsedControl"],
        html body [data-testid="stSidebarCollapseButton"],
        html body button[data-testid="stSidebarCollapseButton"],
        html body [data-testid="stExpandSidebarButton"],
        html body button[data-testid="stExpandSidebarButton"],
        html body [aria-label="Open sidebar"],
        html body [aria-label="Close sidebar"],
        html body [aria-label="Collapse sidebar"] {{
            background: {sidebar_background} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
            box-shadow: {shadow} !important;
        }}

        @media (max-width: 720px) {{
            html body div[class*="st-key-ui_theme_toggle"] button,
            html body button[title="Switch to Light theme"],
            html body button[aria-label="Switch to Light theme"],
            html body button[title="Switch to Dark theme"],
            html body button[aria-label="Switch to Dark theme"] {{
                width: 74px !important;
                min-width: 74px !important;
                max-width: 74px !important;
                height: 30px !important;
                min-height: 30px !important;
                max-height: 30px !important;
                padding: 0 0.46rem !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_theme_control_styles():
    st.markdown(
        """
        <style>
        :root {
            --theme-transition-time: 420ms;
            --theme-transition-fast: 180ms;
            --theme-transition-ease: cubic-bezier(0.2, 0.8, 0.2, 1);
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .block-container {
            transition:
                background var(--theme-transition-time) var(--theme-transition-ease),
                background-color var(--theme-transition-time) var(--theme-transition-ease),
                color var(--theme-transition-time) var(--theme-transition-ease),
                border-color var(--theme-transition-time) var(--theme-transition-ease),
                box-shadow var(--theme-transition-time) var(--theme-transition-ease),
                filter var(--theme-transition-time) var(--theme-transition-ease) !important;
        }

        .stApp {
            animation: themeCanvasSettle 520ms var(--theme-transition-ease) both;
        }

        .block-container {
            animation: themeContentSettle 560ms var(--theme-transition-ease) both;
        }

        .topbar-shell,
        .page-hero,
        .status-card,
        .info-card,
        .theme-control-shell,
        .sidebar-brand,
        .sidebar-user-chip,
        .patient-hero-card,
        .patient-profile-panel,
        .patient-chart-panel,
        .patient-mini-panel,
        .patient-stat-card,
        .patient-appointment-item,
        .patient-profile-chip,
        .patient-calendar-day,
        .remedy-summary-card,
        .remedy-table,
        .remedy-purpose,
        .remedy-safety,
        .provider-card,
        .provider-empty,
        .doctor-action-card,
        .doctor-action-link,
        .assistant-panel-header,
        .assistant-launcher-card,
        .assistant-summary-card,
        .assistant-live-card,
        .assistant-prompt-bar,
        .assistant-symptom-rack,
        .assistant-chat-history,
        .assistant-chat-empty,
        div[data-testid="stMetric"],
        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stAlert"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stChatMessage"],
        section[data-testid="stSidebar"] > div {
            transition:
                background var(--theme-transition-time) var(--theme-transition-ease),
                background-color var(--theme-transition-time) var(--theme-transition-ease),
                border-color var(--theme-transition-time) var(--theme-transition-ease),
                color var(--theme-transition-time) var(--theme-transition-ease),
                box-shadow var(--theme-transition-time) var(--theme-transition-ease),
                transform var(--theme-transition-fast) var(--theme-transition-ease),
                opacity var(--theme-transition-time) var(--theme-transition-ease),
                backdrop-filter var(--theme-transition-time) var(--theme-transition-ease) !important;
        }

        .topbar-shell,
        .page-hero,
        .patient-hero-card,
        .patient-profile-panel,
        .patient-stat-card,
        .info-card,
        .status-card,
        .provider-card,
        .assistant-launcher-card,
        .assistant-panel-header,
        .remedy-summary-card,
        div[data-testid="stMetric"],
        div[data-testid="stForm"],
        div[data-testid="stPlotlyChart"] {
            animation: themeSurfaceSettle 520ms var(--theme-transition-ease) both;
        }

        .patient-stat-card:nth-child(2),
        .info-card:nth-child(2),
        .assistant-summary-card:nth-child(2),
        .provider-card:nth-child(2) {
            animation-delay: 45ms;
        }

        .patient-stat-card:nth-child(3),
        .info-card:nth-child(3),
        .assistant-summary-card:nth-child(3),
        .provider-card:nth-child(3) {
            animation-delay: 90ms;
        }

        .patient-stat-card:nth-child(4),
        .info-card:nth-child(4),
        .assistant-summary-card:nth-child(4),
        .provider-card:nth-child(4) {
            animation-delay: 135ms;
        }

        h1, h2, h3, h4,
        p,
        span,
        label,
        input,
        textarea,
        button,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] * {
            transition:
                color var(--theme-transition-time) var(--theme-transition-ease),
                -webkit-text-fill-color var(--theme-transition-time) var(--theme-transition-ease),
                background-color var(--theme-transition-time) var(--theme-transition-ease),
                border-color var(--theme-transition-time) var(--theme-transition-ease) !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        textarea,
        div[data-baseweb="tag"],
        .stMultiSelect [data-baseweb="tag"],
        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton a,
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            transition:
                background var(--theme-transition-time) var(--theme-transition-ease),
                background-color var(--theme-transition-time) var(--theme-transition-ease),
                border-color var(--theme-transition-time) var(--theme-transition-ease),
                color var(--theme-transition-time) var(--theme-transition-ease),
                box-shadow var(--theme-transition-time) var(--theme-transition-ease),
                transform var(--theme-transition-fast) var(--theme-transition-ease) !important;
        }

        .theme-control-shell {
            position: relative;
            overflow: hidden;
        }

        .theme-control-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(115deg, transparent 0%, rgba(255, 255, 255, 0.52) 42%, transparent 70%);
            transform: translateX(-120%);
            animation: themeControlSweep 850ms var(--theme-transition-ease) 120ms both;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] button:hover,
        .theme-control-shell:hover,
        .provider-card-link:hover .provider-card--clickable,
        .assistant-launcher-card--interactive:hover,
        .doctor-action-link:hover,
        .doctor-action-card:hover,
        .patient-stat-card:hover,
        .info-card:hover,
        .status-card:hover {
            transform: translateY(-2px);
        }

        @keyframes themeCanvasSettle {
            0% {
                opacity: 0.72;
                filter: saturate(0.88) brightness(0.96);
            }
            100% {
                opacity: 1;
                filter: saturate(1) brightness(1);
            }
        }

        @keyframes themeContentSettle {
            0% {
                opacity: 0.82;
                transform: translateY(8px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes themeSurfaceSettle {
            0% {
                opacity: 0.78;
                transform: translateY(10px) scale(0.992);
            }
            100% {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        @keyframes themeControlSweep {
            0% {
                opacity: 0;
                transform: translateX(-120%);
            }
            24% {
                opacity: 0.72;
            }
            100% {
                opacity: 0;
                transform: translateX(120%);
            }
        }

        .theme-control-shell {
            margin: 0.75rem 0 0.55rem 0;
            padding: 0.88rem;
            border-radius: 18px;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.98), rgba(244, 246, 255, 0.94)),
                linear-gradient(120deg, rgba(90, 99, 216, 0.10), rgba(15, 159, 122, 0.08));
            border: 1px solid rgba(90, 99, 216, 0.14);
            box-shadow: 0 10px 22px rgba(64, 74, 130, 0.08);
        }

        .theme-control-kicker {
            font-family: var(--font-data);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--patient-muted);
        }

        .theme-control-title {
            margin-top: 0.24rem;
            color: var(--patient-blue-dark);
            font-family: var(--font-display);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .theme-control-shell p {
            margin: 0.24rem 0 0 0;
            color: var(--patient-muted);
            font-size: 0.78rem;
            line-height: 1.42;
        }

        .theme-logo-preview {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 76px;
            margin-top: 0.78rem;
            border-radius: 18px;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(90, 99, 216, 0.12), rgba(15, 159, 122, 0.10)),
                linear-gradient(145deg, rgba(255, 255, 255, 0.82), rgba(238, 242, 255, 0.76));
            border: 1px solid rgba(90, 99, 216, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
        }

        .theme-logo-preview::before,
        .theme-logo-preview::after {
            content: "";
            position: absolute;
            pointer-events: none;
        }

        .theme-logo-preview::before {
            inset: -45% 24%;
            background: linear-gradient(115deg, transparent, rgba(255, 255, 255, 0.62), transparent);
            transform: rotate(14deg) translateX(-80%);
            animation: themeLogoCrystalSweep 2.6s var(--theme-transition-ease) infinite;
        }

        .theme-logo-preview::after {
            inset: 10px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.46);
            opacity: 0.7;
        }

        .theme-logo-ring {
            position: absolute;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            border: 2px solid rgba(90, 99, 216, 0.34);
            box-shadow: 0 0 0 8px rgba(90, 99, 216, 0.08), 0 14px 30px rgba(64, 74, 130, 0.16);
            animation: themeLogoPulse 2.2s ease-in-out infinite;
        }

        .theme-logo-ring::before,
        .theme-logo-ring::after {
            content: "";
            position: absolute;
            border-radius: 999px;
        }

        .theme-logo-preview--light .theme-logo-ring::before {
            width: 24px;
            height: 24px;
            inset: 11px 0 0 14px;
            background: #3840a0;
            box-shadow: 8px -4px 0 0 #f9fbff;
        }

        .theme-logo-preview--dark .theme-logo-ring::before {
            width: 22px;
            height: 22px;
            inset: 12px 0 0 12px;
            background: #ffb86b;
            box-shadow: 0 0 18px rgba(255, 184, 107, 0.55);
        }

        .theme-logo-preview--dark .theme-logo-ring::after {
            inset: 6px;
            border: 1px dashed rgba(255, 184, 107, 0.58);
            animation: themeLogoSpin 12s linear infinite;
        }

        .theme-logo-mark {
            position: relative;
            z-index: 1;
            margin-top: 54px;
            color: var(--patient-blue-dark);
            font-family: var(--font-data);
            font-size: 0.66rem;
            font-weight: 900;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        .theme-logo-button-marker {
            height: 0;
            overflow: hidden;
        }

        html body #dp-theme-floating-toggle#dp-theme-floating-toggle {
            display: none !important;
            pointer-events: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
        }

        #dp-theme-floating-toggle-style {
            display: none !important;
            pointer-events: none !important;
            visibility: hidden !important;
        }

        div[class*="st-key-ui_theme_toggle"] {
            position: relative !important;
            z-index: 5 !important;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: 100% !important;
            min-width: 0 !important;
            max-width: none !important;
            margin: 0 0 0.45rem auto !important;
            pointer-events: auto !important;
            opacity: 1 !important;
            visibility: visible !important;
            isolation: isolate !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] {
            display: flex !important;
            justify-content: flex-end !important;
            width: auto !important;
            min-width: 0 !important;
            pointer-events: auto !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button {
            position: relative;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.22rem !important;
            width: 78px !important;
            min-width: 78px !important;
            max-width: 78px !important;
            min-height: 30px !important;
            height: 30px !important;
            padding: 0 0.48rem !important;
            border-radius: 999px !important;
            overflow: hidden !important;
            color: var(--patient-blue-dark) !important;
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(240, 245, 255, 0.9)) !important;
            border: 1px solid rgba(90, 99, 216, 0.18) !important;
            box-shadow: 0 12px 26px rgba(64, 74, 130, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.78) !important;
            pointer-events: auto !important;
            cursor: pointer !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button::before {
            content: "";
            position: absolute;
            inset: -32% 54% -32% -35%;
            background: linear-gradient(115deg, transparent, rgba(255, 255, 255, 0.74), transparent);
            transform: translateX(-130%);
            animation: themeButtonSheen 2.8s var(--theme-transition-ease) infinite;
            pointer-events: none;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button p,
        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button span {
            position: relative;
            z-index: 1;
            color: inherit !important;
            font-size: 0.72rem !important;
            font-weight: 900 !important;
            line-height: 1 !important;
            pointer-events: none !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button [data-testid="stIconMaterial"] {
            font-size: 0.88rem !important;
            animation: themeLogoPulse 2.2s ease-in-out infinite;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] {
            margin-bottom: 0.95rem;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button {
            position: relative;
            min-height: 50px !important;
            border-radius: 999px !important;
            overflow: hidden !important;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(239, 243, 255, 0.92)),
                linear-gradient(120deg, rgba(90, 99, 216, 0.18), rgba(15, 159, 122, 0.14)) !important;
            border: 1px solid rgba(90, 99, 216, 0.20) !important;
            color: var(--patient-blue-dark) !important;
            box-shadow: 0 14px 28px rgba(64, 74, 130, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.82) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button::before {
            content: "";
            position: absolute;
            inset: -30% 55% -30% -30%;
            background: linear-gradient(115deg, transparent, rgba(255, 255, 255, 0.74), transparent);
            transform: translateX(-130%);
            animation: themeButtonSheen 2.8s var(--theme-transition-ease) infinite;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button p,
        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button span {
            position: relative;
            z-index: 1;
            color: inherit !important;
            font-weight: 900 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button [data-testid="stIconMaterial"] {
            font-size: 1.28rem !important;
            animation: themeLogoPulse 2.2s ease-in-out infinite;
        }

        @keyframes themeLogoPulse {
            0%, 100% {
                transform: scale(1);
                opacity: 0.92;
            }
            50% {
                transform: scale(1.08);
                opacity: 1;
            }
        }

        @keyframes themeLogoCrystalSweep {
            0% {
                opacity: 0;
                transform: rotate(14deg) translateX(-86%);
            }
            35% {
                opacity: 0.78;
            }
            100% {
                opacity: 0;
                transform: rotate(14deg) translateX(86%);
            }
        }

        @keyframes themeLogoSpin {
            to {
                transform: rotate(360deg);
            }
        }

        @keyframes themeButtonSheen {
            0%, 46% {
                opacity: 0;
                transform: translateX(-130%);
            }
            58% {
                opacity: 0.88;
            }
            100% {
                opacity: 0;
                transform: translateX(230%);
            }
        }

        section[data-testid="stSidebar"] {
            will-change: transform, opacity;
            transform-origin: left center;
            transition:
                transform 460ms var(--theme-transition-ease),
                opacity 360ms var(--theme-transition-ease),
                box-shadow 460ms var(--theme-transition-ease) !important;
        }

        section[data-testid="stSidebar"] > div {
            will-change: transform, opacity;
            transition:
                transform 500ms var(--theme-transition-ease),
                opacity 420ms var(--theme-transition-ease),
                background var(--theme-transition-time) var(--theme-transition-ease),
                border-color var(--theme-transition-time) var(--theme-transition-ease),
                box-shadow var(--theme-transition-time) var(--theme-transition-ease) !important;
        }

        section[data-testid="stSidebar"] > div {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }

        section[data-testid="stSidebar"] > div::-webkit-scrollbar {
            width: 0 !important;
            height: 0 !important;
            display: none !important;
        }

        html.dp-sidebar-opening section[data-testid="stSidebar"] {
            animation: sidebarDrawerOpen 520ms var(--theme-transition-ease) both;
        }

        html.dp-sidebar-opening section[data-testid="stSidebar"] > div {
            animation: sidebarPanelOpen 560ms var(--theme-transition-ease) both;
        }

        html.dp-sidebar-closing section[data-testid="stSidebar"] {
            opacity: 0.08 !important;
            pointer-events: none !important;
            transform: translateX(-104%) scale(0.985) !important;
        }

        html.dp-sidebar-closing section[data-testid="stSidebar"] > div {
            transform: translateX(-18px) scale(0.982) !important;
            opacity: 0.35 !important;
        }

        html.dp-sidebar-closing [data-testid="stSidebarCollapseButton"],
        html.dp-sidebar-closing button[data-testid="stSidebarCollapseButton"] {
            transform: translateX(-12px) scale(0.92) !important;
            opacity: 0.2 !important;
        }

        html.dp-sidebar-opening [data-testid="stSidebarCollapsedControl"],
        html.dp-sidebar-opening button[data-testid="stSidebarCollapsedControl"],
        html.dp-sidebar-opening [data-testid="stExpandSidebarButton"],
        html.dp-sidebar-opening button[data-testid="stExpandSidebarButton"] {
            animation: sidebarTogglePop 420ms var(--theme-transition-ease) both;
        }

        @keyframes sidebarDrawerOpen {
            0% {
                opacity: 0;
                transform: translateX(-104%) scale(0.985);
            }
            62% {
                opacity: 1;
                transform: translateX(8px) scale(1.002);
            }
            100% {
                opacity: 1;
                transform: translateX(0) scale(1);
            }
        }

        @keyframes sidebarPanelOpen {
            0% {
                opacity: 0.35;
                transform: translateX(-24px) scale(0.982);
            }
            68% {
                opacity: 1;
                transform: translateX(4px) scale(1.004);
            }
            100% {
                opacity: 1;
                transform: translateX(0) scale(1);
            }
        }

        @keyframes sidebarTogglePop {
            0% {
                transform: scale(0.88);
            }
            70% {
                transform: scale(1.08);
            }
            100% {
                transform: scale(1);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 1ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 1ms !important;
            }
        }

        @media (max-width: 720px) {
            div[class*="st-key-ui_theme_toggle"] {
                width: 100% !important;
                max-width: none !important;
                margin-bottom: 0.38rem !important;
            }

            div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button {
                width: 74px !important;
                min-width: 74px !important;
                max-width: 74px !important;
                min-height: 30px !important;
                height: 30px !important;
                max-height: 30px !important;
                padding: 0 0.46rem !important;
            }
        }

        .assistant-panel-header,
        .assistant-launcher-card,
        .assistant-chat-history,
        .assistant-chat-bubble,
        div[data-testid="stChatMessage"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) {
            filter: none !important;
            backdrop-filter: none !important;
            animation: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_dark_theme_styles():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --bg-1: #090d12;
            --bg-2: #111820;
            --ink: #edf5ff;
            --muted: #a7b6c8;
            --line: rgba(136, 160, 181, 0.22);
            --panel: rgba(18, 25, 34, 0.84);
            --panel-strong: rgba(22, 31, 42, 0.96);
            --teal: #35d3ad;
            --teal-soft: rgba(53, 211, 173, 0.18);
            --orange: #ffb86b;
            --shadow: 0 28px 70px rgba(0, 0, 0, 0.48);
            --shadow-soft: 0 16px 38px rgba(0, 0, 0, 0.34);
            --patient-bg: #090d12;
            --patient-surface: #121923;
            --patient-card: #161f2a;
            --patient-blue: #35d3ad;
            --patient-blue-dark: #dffcf5;
            --patient-lilac: rgba(139, 92, 246, 0.14);
            --patient-cyan: rgba(45, 212, 191, 0.15);
            --patient-yellow: rgba(251, 191, 36, 0.16);
            --patient-red: rgba(251, 113, 133, 0.16);
            --patient-text: #eef6ff;
            --patient-muted: #9fb0c3;
            --patient-line: rgba(136, 160, 181, 0.22);
            --patient-shadow: 0 26px 70px rgba(0, 0, 0, 0.46);
            --patient-soft-shadow: 0 16px 36px rgba(0, 0, 0, 0.32);
        }

        html {
            scrollbar-color: #35d3ad #0b1118;
        }

        body,
        .stApp,
        [data-testid="stAppViewContainer"] {
            color: var(--ink) !important;
            background:
                linear-gradient(118deg, rgba(53, 211, 173, 0.12), transparent 28%),
                linear-gradient(244deg, rgba(109, 93, 252, 0.14), transparent 34%),
                linear-gradient(18deg, rgba(255, 184, 107, 0.10), transparent 36%),
                linear-gradient(160deg, #05080d 0%, #0b121b 42%, #061615 68%, #140f1e 100%) !important;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
                linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
                linear-gradient(180deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px);
            background-size: 54px 54px;
            mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.72), transparent 92%);
            animation: darkGridDrift 28s linear infinite, darkGridReveal 720ms var(--theme-transition-ease) both;
        }

        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background:
                repeating-linear-gradient(132deg, transparent 0 72px, rgba(159, 246, 223, 0.055) 73px, transparent 76px),
                repeating-linear-gradient(48deg, transparent 0 94px, rgba(174, 184, 255, 0.048) 95px, transparent 99px),
                linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.035), transparent);
            mix-blend-mode: screen;
            opacity: 0.9;
            animation: crystalFacetShift 24s linear infinite, darkGridReveal 760ms var(--theme-transition-ease) both;
        }

        .block-container {
            position: relative;
            z-index: 1;
        }

        ::-webkit-scrollbar-track {
            background: #0b1118;
            border-color: #070a0f;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #35d3ad, #8b5cf6);
            border-color: #070a0f;
            box-shadow: none;
        }

        h1, h2, h3, h4,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        .topbar-title,
        .page-title,
        .section-title,
        .status-title,
        .info-card h4,
        .patient-greeting,
        .patient-profile-name,
        .patient-stat-value,
        .patient-chart-heading,
        .patient-plot-title,
        .provider-title,
        .provider-card h4,
        .assistant-shell-title,
        .assistant-launcher-title,
        .assistant-summary-value,
        .remedy-summary-value {
            color: var(--ink) !important;
            -webkit-text-fill-color: currentColor !important;
        }

        p,
        .topbar-subtitle,
        .page-description,
        .section-copy,
        .status-copy,
        .info-card p,
        .patient-hero-text,
        .patient-stat-sub,
        .patient-profile-role,
        .patient-profile-chip span,
        .patient-plot-caption,
        .provider-address,
        .provider-meta,
        .provider-contact,
        .assistant-shell-copy,
        .assistant-launcher-copy,
        .assistant-summary-copy,
        .remedy-table-note,
        .stCaption {
            color: var(--muted) !important;
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: transparent !important;
        }

        [data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"],
        button[data-testid="stExpandSidebarButton"],
        [aria-label="Open sidebar"],
        [aria-label="Close sidebar"],
        [aria-label="Collapse sidebar"] {
            background: linear-gradient(145deg, #162231, #0b1118) !important;
            border-color: rgba(53, 211, 173, 0.35) !important;
            color: #dffcf5 !important;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.44), 0 0 0 5px rgba(53, 211, 173, 0.12) !important;
        }

        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stExpandSidebarButton"] *,
        [data-testid="stSidebarCollapsedControl"] *,
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="stSidebarCollapsedControl"] svg {
            color: #dffcf5 !important;
            fill: currentColor !important;
        }

        section[data-testid="stSidebar"] {
            background: transparent !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] > div {
            background:
                linear-gradient(160deg, rgba(29, 43, 58, 0.92), rgba(8, 13, 20, 0.94) 62%),
                linear-gradient(45deg, rgba(53, 211, 173, 0.11), rgba(109, 93, 252, 0.10)) !important;
            border: 1px solid rgba(159, 246, 223, 0.22) !important;
            box-shadow: 0 28px 82px rgba(0, 0, 0, 0.54), inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
            backdrop-filter: blur(22px) saturate(1.18);
        }

        .sidebar-brand,
        .theme-control-shell,
        .sidebar-user-chip {
            background:
                linear-gradient(145deg, rgba(30, 45, 60, 0.88), rgba(9, 15, 23, 0.92)),
                linear-gradient(120deg, rgba(53, 211, 173, 0.13), rgba(109, 93, 252, 0.12)) !important;
            border: 1px solid rgba(159, 246, 223, 0.24) !important;
            color: var(--ink) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 18px 42px rgba(0, 0, 0, 0.38) !important;
        }

        .sidebar-brand-title,
        .theme-control-title {
            color: #dffcf5 !important;
        }

        .sidebar-brand-copy,
        .theme-control-kicker,
        .theme-control-shell p {
            color: var(--muted) !important;
        }

        .theme-control-shell::after {
            background: linear-gradient(115deg, transparent 0%, rgba(53, 211, 173, 0.34) 42%, transparent 70%) !important;
        }

        .theme-logo-preview {
            background:
                linear-gradient(135deg, rgba(53, 211, 173, 0.18), rgba(109, 93, 252, 0.16)),
                linear-gradient(145deg, rgba(23, 34, 47, 0.86), rgba(9, 14, 21, 0.90)) !important;
            border: 1px solid rgba(159, 246, 223, 0.24) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10), 0 18px 38px rgba(0, 0, 0, 0.35) !important;
        }

        .theme-logo-preview::after {
            border-color: rgba(159, 246, 223, 0.24) !important;
        }

        .theme-logo-ring {
            border-color: rgba(159, 246, 223, 0.46) !important;
            box-shadow: 0 0 0 8px rgba(53, 211, 173, 0.10), 0 0 28px rgba(53, 211, 173, 0.32), 0 18px 34px rgba(0, 0, 0, 0.28) !important;
        }

        .theme-logo-mark {
            color: #dffcf5 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button {
            background:
                linear-gradient(135deg, rgba(53, 211, 173, 0.26), rgba(109, 93, 252, 0.24)),
                linear-gradient(145deg, rgba(25, 38, 52, 0.96), rgba(8, 13, 20, 0.94)) !important;
            border: 1px solid rgba(159, 246, 223, 0.34) !important;
            color: #dffcf5 !important;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.44), 0 0 28px rgba(53, 211, 173, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.11) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button::before {
            background: linear-gradient(115deg, transparent, rgba(159, 246, 223, 0.40), transparent) !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button {
            background:
                linear-gradient(135deg, rgba(53, 211, 173, 0.20), rgba(109, 93, 252, 0.22)),
                linear-gradient(145deg, rgba(21, 32, 44, 0.96), rgba(8, 13, 20, 0.94)) !important;
            border: 1px solid rgba(159, 246, 223, 0.32) !important;
            color: #dffcf5 !important;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.42), 0 0 24px rgba(53, 211, 173, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button::before {
            background: linear-gradient(115deg, transparent, rgba(159, 246, 223, 0.38), transparent) !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(14, 21, 30, 0.92) !important;
            border: 1px solid rgba(136, 160, 181, 0.22) !important;
            color: var(--ink) !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(53, 211, 173, 0.12) !important;
            border-color: rgba(53, 211, 173, 0.32) !important;
        }

        section[data-testid="stSidebar"] label:has(input:checked) {
            background: linear-gradient(145deg, rgba(31, 111, 104, 0.95), rgba(31, 49, 68, 0.96)) !important;
            border-color: rgba(119, 220, 202, 0.28) !important;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.30) !important;
        }

        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.theme-logo-button-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] * {
            color: var(--ink) !important;
        }

        .topbar-shell,
        .page-hero,
        .status-card,
        .info-card,
        div[data-testid="stMetric"],
        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stChatMessage"] {
            background: linear-gradient(145deg, rgba(22, 31, 42, 0.92), rgba(11, 17, 24, 0.86)) !important;
            border: 1px solid rgba(136, 160, 181, 0.22) !important;
            box-shadow: var(--shadow-soft) !important;
            backdrop-filter: blur(16px);
        }

        .topbar-shell,
        .page-hero,
        .status-card,
        .info-card,
        .patient-hero-card,
        .patient-profile-panel,
        .patient-chart-panel,
        .patient-mini-panel,
        .patient-stat-card,
        .provider-card,
        .doctor-action-card,
        .doctor-action-link,
        .assistant-panel-header,
        .assistant-launcher-card,
        .assistant-summary-card,
        .remedy-summary-card,
        .remedy-table {
            position: relative;
            overflow: hidden;
        }

        .topbar-shell::before,
        .page-hero::before,
        .status-card::before,
        .info-card::before,
        .patient-hero-card::before,
        .patient-profile-panel::before,
        .patient-chart-panel::before,
        .patient-mini-panel::before,
        .patient-stat-card::before,
        .provider-card::before,
        .doctor-action-card::before,
        .doctor-action-link::before,
        .assistant-panel-header::before,
        .assistant-launcher-card::before,
        .assistant-summary-card::before,
        .remedy-summary-card::before,
        .remedy-table::before {
            content: "";
            position: absolute;
            inset: -40% 54% -40% -36%;
            pointer-events: none;
            background: linear-gradient(118deg, transparent, rgba(159, 246, 223, 0.12), rgba(174, 184, 255, 0.08), transparent);
            transform: translateX(-100%) rotate(10deg);
            animation: crystalPanelSweep 6.4s ease-in-out infinite;
        }

        .page-hero::after,
        .patient-hero-card::after {
            background: linear-gradient(90deg, rgba(53, 211, 173, 0.18), rgba(251, 184, 107, 0.12)) !important;
        }

        .page-eyebrow,
        .card-kicker,
        .status-label,
        .hero-chip,
        .inline-chip,
        .patient-stat-kicker,
        .patient-read-more,
        .patient-calendar-title,
        .patient-plan-title,
        .provider-chip,
        .assistant-panel-kicker,
        .assistant-launcher-kicker,
        .assistant-summary-label,
        .assistant-quick-title,
        .remedy-summary-label,
        .remedy-symptom-tag {
            color: #9ff6df !important;
        }

        .hero-chip,
        .inline-chip,
        .patient-read-more,
        .assistant-tag,
        .assistant-mode-chip,
        .assistant-symptom-chip,
        .remedy-symptom-tag,
        .provider-chip {
            background: rgba(53, 211, 173, 0.14) !important;
            border-color: rgba(53, 211, 173, 0.24) !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton a,
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(145deg, rgba(24, 35, 48, 0.98), rgba(12, 17, 24, 0.94)) !important;
            border: 1px solid rgba(136, 160, 181, 0.24) !important;
            color: var(--ink) !important;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.32) !important;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stLinkButton a:hover,
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(145deg, rgba(35, 49, 64, 0.98), rgba(15, 23, 32, 0.98)) !important;
            border-color: rgba(53, 211, 173, 0.42) !important;
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.38) !important;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: linear-gradient(145deg, rgba(31, 111, 104, 0.98), rgba(35, 74, 108, 0.98)) !important;
            border-color: rgba(119, 220, 202, 0.26) !important;
            color: #ffffff !important;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.34) !important;
        }

        .stButton > button p,
        .stFormSubmitButton > button p,
        .stLinkButton a p,
        div[data-testid="stButton"] > button p,
        div[data-testid="stFormSubmitButton"] > button p {
            color: currentColor !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        textarea {
            background: rgba(10, 15, 22, 0.94) !important;
            border: 1px solid rgba(136, 160, 181, 0.24) !important;
            color: var(--ink) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
        }

        input,
        textarea,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        [data-baseweb="select"] span {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        input::placeholder,
        textarea::placeholder,
        [data-baseweb="input"] input::placeholder,
        [data-baseweb="base-input"] input::placeholder {
            color: #738397 !important;
            -webkit-text-fill-color: #738397 !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"],
        div[role="listbox"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] > div > div {
            background: #111923 !important;
            border: 1px solid rgba(136, 160, 181, 0.24) !important;
            color: var(--ink) !important;
            box-shadow: 0 22px 54px rgba(0, 0, 0, 0.48) !important;
        }

        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        ul[role="listbox"] *,
        div[role="listbox"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: currentColor !important;
        }

        li[role="option"],
        div[role="option"],
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] [role="option"],
        ul[role="listbox"] li {
            background: transparent !important;
            color: var(--ink) !important;
        }

        li[role="option"]:hover,
        div[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        div[role="option"][aria-selected="true"],
        [data-baseweb="menu"] [aria-selected="true"],
        [data-baseweb="menu"] [data-highlighted="true"] {
            background: rgba(53, 211, 173, 0.16) !important;
            color: #dffcf5 !important;
        }

        div[data-baseweb="select"],
        div[data-testid="stMultiSelect"],
        div[data-testid="stSelectbox"],
        div[data-testid="stRadio"],
        div[data-testid="stCheckbox"] {
            color: var(--ink) !important;
        }

        div[data-baseweb="select"] svg,
        div[data-testid="stMultiSelect"] svg,
        div[data-testid="stSelectbox"] svg {
            color: var(--muted) !important;
            fill: currentColor !important;
        }

        div[data-baseweb="checkbox"] > div,
        label[data-baseweb="checkbox"] > div,
        div[data-testid="stCheckbox"] label > div:first-child {
            background: rgba(10, 15, 22, 0.92) !important;
            border-color: rgba(136, 160, 181, 0.34) !important;
            color: var(--ink) !important;
        }

        div[data-baseweb="checkbox"][aria-checked="true"] > div,
        label[data-baseweb="checkbox"][aria-checked="true"] > div,
        div[data-testid="stCheckbox"] input:checked + div {
            background: #1f6f68 !important;
            border-color: #35d3ad !important;
        }

        div[data-baseweb="tag"],
        .stMultiSelect [data-baseweb="tag"] {
            background: rgba(53, 211, 173, 0.16) !important;
            color: #dffcf5 !important;
            border: 1px solid rgba(53, 211, 173, 0.24) !important;
        }

        div[data-baseweb="tag"] *,
        .stMultiSelect [data-baseweb="tag"] * {
            color: #dffcf5 !important;
            -webkit-text-fill-color: #dffcf5 !important;
        }

        ul,
        ol,
        li,
        table,
        thead,
        tbody,
        tr,
        th,
        td,
        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] *,
        div[data-testid="stJson"] *,
        div[data-testid="stExpander"] *,
        div[data-testid="stMarkdownContainer"] ul,
        div[data-testid="stMarkdownContainer"] ol,
        div[data-testid="stMarkdownContainer"] li {
            color: var(--ink) !important;
            -webkit-text-fill-color: currentColor !important;
        }

        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] [role="row"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stTable"] table,
        div[data-testid="stTable"] th,
        div[data-testid="stTable"] td {
            background-color: rgba(12, 18, 26, 0.94) !important;
            border-color: rgba(136, 160, 181, 0.18) !important;
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricDelta"],
        label,
        .stMarkdown label {
            color: var(--muted) !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
        }

        div[data-testid="stAlert"] {
            background: linear-gradient(145deg, rgba(18, 28, 39, 0.96), rgba(12, 18, 26, 0.94)) !important;
            border: 1px solid rgba(136, 160, 181, 0.24) !important;
            color: var(--ink) !important;
        }

        button[data-baseweb="tab"] {
            color: var(--muted) !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #9ff6df !important;
            border-color: #35d3ad !important;
        }

        .patient-hero-card,
        .patient-profile-panel,
        .patient-chart-panel,
        .patient-mini-panel,
        .patient-stat-card,
        .patient-appointment-item,
        .patient-profile-chip,
        .patient-calendar-day,
        .remedy-summary-card,
        .remedy-table,
        .remedy-purpose,
        .remedy-safety,
        .provider-card,
        .provider-empty,
        .doctor-action-card,
        .doctor-action-link,
        .assistant-panel-header,
        .assistant-launcher-card,
        .assistant-summary-card,
        .assistant-live-card,
        .assistant-prompt-bar,
        .assistant-symptom-rack,
        .assistant-chat-history,
        .assistant-chat-empty {
            background: linear-gradient(145deg, rgba(22, 31, 42, 0.92), rgba(11, 17, 24, 0.86)) !important;
            border: 1px solid rgba(136, 160, 181, 0.22) !important;
            box-shadow: var(--patient-soft-shadow) !important;
        }

        .patient-hero-card {
            background:
                linear-gradient(135deg, rgba(53, 211, 173, 0.18), transparent 52%),
                linear-gradient(145deg, rgba(24, 35, 48, 0.96), rgba(11, 17, 24, 0.92)) !important;
        }

        .patient-stat-card.stat-blue,
        .patient-stat-card.stat-mint,
        .patient-stat-card.stat-yellow,
        .patient-stat-card.stat-rose {
            background: linear-gradient(145deg, rgba(22, 31, 42, 0.94), rgba(13, 19, 27, 0.92)) !important;
        }

        .patient-avatar,
        .assistant-panel-avatar,
        .assistant-launcher-logo,
        .assistant-chat-avatar {
            background: linear-gradient(145deg, #1f6f68, #234a6c) !important;
            color: #ffffff !important;
            border-color: rgba(255, 255, 255, 0.12) !important;
        }

        .patient-calendar-day.active {
            background: linear-gradient(145deg, #1f6f68, #234a6c) !important;
            color: #ffffff !important;
        }

        .patient-calendar-weekday {
            color: rgba(223, 252, 245, 0.72) !important;
        }

        .provider-card-link:hover .provider-card--clickable,
        .assistant-launcher-card--interactive:hover,
        .doctor-action-link:hover {
            border-color: rgba(53, 211, 173, 0.45) !important;
            box-shadow: 0 22px 48px rgba(0, 0, 0, 0.42) !important;
        }

        .provider-card-top strong,
        .provider-map-cta,
        .assistant-inline-status,
        .assistant-live-value,
        .remedy-step-index {
            background: linear-gradient(145deg, #1f6f68, #234a6c) !important;
            color: #ffffff !important;
        }

        .provider-meta span,
        .provider-contact span,
        .provider-contact a,
        .remedy-step,
        .remedy-care-table td,
        .remedy-care-table th {
            color: var(--muted) !important;
            border-color: rgba(136, 160, 181, 0.18) !important;
        }

        .remedy-care-table,
        .remedy-care-table thead,
        .remedy-care-table tr {
            background: transparent !important;
            border-color: rgba(136, 160, 181, 0.18) !important;
        }

        .assistant-chat-bubble {
            background: rgba(20, 30, 42, 0.94) !important;
            border: 1px solid rgba(136, 160, 181, 0.22) !important;
            color: var(--ink) !important;
        }

        .assistant-chat-bubble--user {
            background: linear-gradient(135deg, rgba(25, 185, 150, 0.24), rgba(109, 93, 252, 0.22)) !important;
            border-color: rgba(53, 211, 173, 0.32) !important;
        }

        .assistant-chat-text,
        .assistant-chat-label,
        .assistant-chat-empty-title,
        .assistant-chat-empty-copy {
            color: var(--ink) !important;
        }

        .assistant-chat-empty-copy {
            color: var(--muted) !important;
        }

        .city-suggestion-panel,
        .city-suggestion-pill {
            background: rgba(18, 25, 34, 0.94) !important;
            border-color: rgba(136, 160, 181, 0.22) !important;
            color: var(--ink) !important;
        }

        .city-suggestion-title {
            color: #9ff6df !important;
        }

        hr,
        .stDivider,
        div[data-testid="stDivider"] {
            border-color: rgba(136, 160, 181, 0.18) !important;
        }

        div[class*="st-key-ui_theme_toggle"] {
            display: flex !important;
            justify-content: flex-end !important;
            width: 100% !important;
            max-width: none !important;
            margin: 0 0 0.45rem auto !important;
            pointer-events: auto !important;
            z-index: 5 !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] {
            display: flex !important;
            justify-content: flex-end !important;
            width: auto !important;
            min-width: 0 !important;
            pointer-events: auto !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.22rem !important;
            width: 78px !important;
            min-width: 78px !important;
            max-width: 78px !important;
            height: 30px !important;
            min-height: 30px !important;
            max-height: 30px !important;
            padding: 0 0.48rem !important;
            border-radius: 999px !important;
            overflow: hidden !important;
            background:
                linear-gradient(135deg, rgba(53, 211, 173, 0.20), rgba(109, 93, 252, 0.22)),
                linear-gradient(145deg, rgba(21, 32, 44, 0.96), rgba(8, 13, 20, 0.94)) !important;
            border: 1px solid rgba(159, 246, 223, 0.32) !important;
            color: #dffcf5 !important;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.42), 0 0 24px rgba(53, 211, 173, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
            pointer-events: auto !important;
            cursor: pointer !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button p,
        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button span {
            color: inherit !important;
            font-size: 0.72rem !important;
            font-weight: 900 !important;
            line-height: 1 !important;
            pointer-events: none !important;
        }

        div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button [data-testid="stIconMaterial"] {
            font-size: 0.88rem !important;
            animation: themeLogoPulse 2.2s ease-in-out infinite;
        }

        @media (max-width: 720px) {
            div[class*="st-key-ui_theme_toggle"] div[data-testid="stButton"] > button {
                width: 74px !important;
                min-width: 74px !important;
                max-width: 74px !important;
                height: 30px !important;
                min-height: 30px !important;
                max-height: 30px !important;
                padding: 0 0.46rem !important;
            }
        }

        body,
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #070b10 0%, #0e1620 58%, #111827 100%) !important;
        }

        .stApp::before,
        .stApp::after {
            display: none !important;
            animation: none !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton a,
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(145deg, #1b2634, #111827) !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            color: #edf5ff !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28) !important;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stLinkButton a:hover,
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(145deg, #223044, #151f2d) !important;
            border-color: rgba(53, 211, 173, 0.38) !important;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.34) !important;
            transform: translateY(-1px) !important;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: linear-gradient(145deg, #0f766e, #0b5f58) !important;
            border-color: rgba(159, 246, 223, 0.28) !important;
            color: #ffffff !important;
            box-shadow: 0 12px 26px rgba(15, 118, 110, 0.22) !important;
        }

        section[data-testid="stSidebar"] label:has(input:checked) {
            background: linear-gradient(145deg, #0f766e, #0b5f58) !important;
            border-color: rgba(159, 246, 223, 0.28) !important;
            box-shadow: 0 12px 26px rgba(15, 118, 110, 0.22) !important;
        }

        .assistant-panel-header,
        .assistant-launcher-card,
        .assistant-chat-history,
        .assistant-chat-bubble,
        div[data-testid="stChatMessage"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) {
            filter: none !important;
            backdrop-filter: none !important;
            animation: none !important;
        }

        .assistant-prompt-bar,
        .assistant-live-strip,
        .assistant-summary-grid,
        .assistant-connection-note,
        .assistant-inline-status,
        .assistant-launcher-stats,
        .assistant-launcher-tags {
            display: none !important;
        }

        /* Final dark form clamp: catches BaseWeb dropdowns rendered outside normal Streamlit containers. */
        body div[data-baseweb="select"],
        body div[data-baseweb="input"],
        body div[data-baseweb="base-input"],
        body div[data-baseweb="textarea"],
        body div[data-testid="stTextInput"] div[data-baseweb="input"],
        body div[data-testid="stTextArea"] div[data-baseweb="textarea"],
        body div[data-testid="stSelectbox"] div[data-baseweb="select"],
        body div[data-testid="stMultiSelect"] div[data-baseweb="select"] {
            background: #0d151f !important;
            background-color: #0d151f !important;
            border-color: rgba(136, 160, 181, 0.28) !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 10px 24px rgba(0, 0, 0, 0.20) !important;
        }

        body div[data-baseweb="select"] > div,
        body div[data-baseweb="input"] > div,
        body div[data-baseweb="base-input"] > div,
        body div[data-baseweb="textarea"] > div {
            background: #0d151f !important;
            background-color: #0d151f !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-baseweb="select"] input,
        body div[data-baseweb="input"] input,
        body div[data-baseweb="base-input"] input,
        body div[data-baseweb="textarea"] textarea,
        body div[data-baseweb="select"] span,
        body div[data-baseweb="select"] div {
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-testid="stTextInput"]:has(input:-webkit-autofill) div[data-baseweb="input"],
        body div[data-baseweb="input"]:has(input:-webkit-autofill),
        body div[data-baseweb="base-input"]:has(input:-webkit-autofill),
        body div[data-testid="stTextInput"]:has(input:autofill) div[data-baseweb="input"],
        body div[data-baseweb="input"]:has(input:autofill),
        body div[data-baseweb="base-input"]:has(input:autofill) {
            background: #0d151f !important;
            background-color: #0d151f !important;
            border-color: rgba(136, 160, 181, 0.30) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 10px 24px rgba(0, 0, 0, 0.20) !important;
        }

        body input:-webkit-autofill,
        body input:-webkit-autofill:hover,
        body input:-webkit-autofill:focus,
        body input:-webkit-autofill:active,
        body textarea:-webkit-autofill,
        body textarea:-webkit-autofill:hover,
        body textarea:-webkit-autofill:focus,
        body select:-webkit-autofill,
        body select:-webkit-autofill:hover,
        body select:-webkit-autofill:focus {
            background: #0d151f !important;
            background-color: #0d151f !important;
            -webkit-box-shadow: 0 0 0 1000px #0d151f inset !important;
            box-shadow: 0 0 0 1000px #0d151f inset !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
            caret-color: #edf5ff !important;
            border-color: rgba(136, 160, 181, 0.30) !important;
            transition: background-color 999999s ease-in-out 0s, color 999999s ease-in-out 0s !important;
        }

        body input:autofill,
        body textarea:autofill,
        body select:autofill {
            background: #0d151f !important;
            background-color: #0d151f !important;
            box-shadow: 0 0 0 1000px #0d151f inset !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
            caret-color: #edf5ff !important;
        }

        /* Hard override for text/password fields, including saved-password and autofill paint. */
        body div[data-testid="stTextInput"],
        body div[data-testid="stTextArea"],
        body div[data-testid="stNumberInput"],
        body div[data-testid="stDateInput"],
        body div[data-testid="stTimeInput"] {
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-testid="stTextInput"] div[data-baseweb],
        body div[data-testid="stTextArea"] div[data-baseweb],
        body div[data-testid="stNumberInput"] div[data-baseweb],
        body div[data-testid="stDateInput"] div[data-baseweb],
        body div[data-testid="stTimeInput"] div[data-baseweb],
        body div[data-testid="stTextInput"] div[data-baseweb] > div,
        body div[data-testid="stTextArea"] div[data-baseweb] > div,
        body div[data-testid="stNumberInput"] div[data-baseweb] > div,
        body div[data-testid="stDateInput"] div[data-baseweb] > div,
        body div[data-testid="stTimeInput"] div[data-baseweb] > div {
            background: #0d151f !important;
            background-color: #0d151f !important;
            background-image: none !important;
            border-color: rgba(136, 160, 181, 0.32) !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 10px 24px rgba(0, 0, 0, 0.20) !important;
        }

        body div[data-testid="stTextInput"] input,
        body div[data-testid="stTextArea"] textarea,
        body div[data-testid="stNumberInput"] input,
        body div[data-testid="stDateInput"] input,
        body div[data-testid="stTimeInput"] input,
        body input[type="text"],
        body input[type="password"],
        body input[type="email"],
        body input[type="search"],
        body input[type="number"],
        body input[type="tel"],
        body input[type="url"] {
            appearance: none !important;
            background: #0d151f !important;
            background-color: #0d151f !important;
            background-image: none !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
            caret-color: #edf5ff !important;
            border-color: rgba(136, 160, 181, 0.32) !important;
            -webkit-box-shadow: 0 0 0 1000px #0d151f inset !important;
            box-shadow: 0 0 0 1000px #0d151f inset !important;
        }

        body div[data-testid="stTextInput"] input:focus,
        body div[data-testid="stTextArea"] textarea:focus,
        body div[data-testid="stNumberInput"] input:focus,
        body div[data-testid="stDateInput"] input:focus,
        body div[data-testid="stTimeInput"] input:focus {
            background: #0d151f !important;
            background-color: #0d151f !important;
            -webkit-box-shadow: 0 0 0 1000px #0d151f inset, 0 0 0 1px rgba(53, 211, 173, 0.42) !important;
            box-shadow: 0 0 0 1000px #0d151f inset, 0 0 0 1px rgba(53, 211, 173, 0.42) !important;
        }

        body div[data-testid="stTextInput"] input::placeholder,
        body div[data-testid="stTextArea"] textarea::placeholder,
        body input::placeholder,
        body textarea::placeholder {
            color: #738397 !important;
            -webkit-text-fill-color: #738397 !important;
            opacity: 1 !important;
        }

        body div[data-baseweb="popover"],
        body div[data-baseweb="popover"] > div,
        body div[data-baseweb="popover"] > div > div,
        body div[data-baseweb="select-dropdown"],
        body div[data-baseweb="select-dropdown"] > div,
        body div[data-baseweb="select-dropdown"] ul,
        body div[data-baseweb="select-dropdown"] li,
        body div[data-baseweb="menu"],
        body div[data-baseweb="menu"] > div,
        body ul[role="listbox"],
        body div[role="listbox"],
        body [role="presentation"] ul[role="listbox"] {
            background: #0d151f !important;
            background-color: #0d151f !important;
            border-color: rgba(136, 160, 181, 0.26) !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-baseweb="popover"] *,
        body div[data-baseweb="select-dropdown"] *,
        body div[data-baseweb="menu"] *,
        body ul[role="listbox"] *,
        body div[role="listbox"] * {
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-baseweb="select-dropdown"] [role="option"],
        body div[data-baseweb="select-dropdown"] li,
        body div[data-baseweb="menu"] [role="option"],
        body ul[role="listbox"] li,
        body ul[role="listbox"] [role="option"],
        body div[role="listbox"] [role="option"] {
            background: #0d151f !important;
            background-color: #0d151f !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-baseweb="select-dropdown"] [role="option"]:hover,
        body div[data-baseweb="select-dropdown"] li:hover,
        body div[data-baseweb="select-dropdown"] [aria-selected="true"],
        body div[data-baseweb="select-dropdown"] [data-highlighted="true"],
        body div[data-baseweb="menu"] [role="option"]:hover,
        body div[data-baseweb="menu"] [aria-selected="true"],
        body ul[role="listbox"] li:hover,
        body ul[role="listbox"] [aria-selected="true"],
        body div[role="listbox"] [role="option"]:hover,
        body div[role="listbox"] [aria-selected="true"] {
            background: rgba(53, 211, 173, 0.20) !important;
            background-color: rgba(53, 211, 173, 0.20) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        body div[data-testid="stCheckbox"] label,
        body div[data-testid="stRadio"] label,
        body div[data-baseweb="checkbox"],
        body label[data-baseweb="checkbox"],
        body div[data-baseweb="radio"] {
            background: transparent !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-testid="stCheckbox"] label *,
        body div[data-testid="stRadio"] label *,
        body div[data-baseweb="checkbox"] *,
        body label[data-baseweb="checkbox"] *,
        body div[data-baseweb="radio"] * {
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body div[data-baseweb="checkbox"] [role="checkbox"],
        body label[data-baseweb="checkbox"] [role="checkbox"],
        body div[data-baseweb="radio"] [role="radio"],
        body div[data-testid="stCheckbox"] input + div,
        body div[data-testid="stRadio"] input + div {
            background: #0d151f !important;
            background-color: #0d151f !important;
            border-color: rgba(136, 160, 181, 0.36) !important;
        }

        body button[data-testid^="stBaseButton"],
        body div[data-baseweb="button-group"] button,
        body div[data-testid="stButton"] > button,
        body div[data-testid="stFormSubmitButton"] > button,
        body .stLinkButton a {
            background: linear-gradient(145deg, #182332, #0f1722) !important;
            background-color: #0f1722 !important;
            border: 1px solid rgba(136, 160, 181, 0.28) !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28) !important;
        }

        body button[data-testid^="stBaseButton"] *,
        body div[data-baseweb="button-group"] button *,
        body div[data-testid="stButton"] > button *,
        body div[data-testid="stFormSubmitButton"] > button *,
        body .stLinkButton a * {
            color: inherit !important;
            -webkit-text-fill-color: currentColor !important;
        }

        body button[data-testid="stBaseButton-primary"],
        body button[data-testid="stBaseButton-pillsActive"],
        body button[kind="primary"],
        body div[data-testid="stButton"] > button[kind="primary"],
        body div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: linear-gradient(145deg, #0f766e, #0b5f58) !important;
            background-color: #0f766e !important;
            border-color: rgba(159, 246, 223, 0.32) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        body .provider-meta span,
        body .provider-contact span,
        body .provider-chip,
        body .patient-profile-chip,
        body .patient-calendar-day,
        body .city-suggestion-pill,
        body .hero-chip,
        body .inline-chip,
        body .assistant-tag,
        body .assistant-symptom-chip,
        body .remedy-symptom-tag {
            background: rgba(18, 28, 39, 0.96) !important;
            background-color: rgba(18, 28, 39, 0.96) !important;
            border: 1px solid rgba(136, 160, 181, 0.26) !important;
            color: #dfeaf7 !important;
            -webkit-text-fill-color: #dfeaf7 !important;
        }

        body .provider-meta span *,
        body .provider-contact span *,
        body .provider-chip *,
        body .patient-profile-chip *,
        body .city-suggestion-pill *,
        body .hero-chip *,
        body .inline-chip *,
        body .assistant-tag *,
        body .assistant-symptom-chip *,
        body .remedy-symptom-tag * {
            color: inherit !important;
            -webkit-text-fill-color: currentColor !important;
        }

        /* Last-pass dark clamp for BaseWeb popovers and dataframe/table surfaces. */
        body [data-baseweb="popover"],
        body [data-baseweb="popover"] > div,
        body [data-baseweb="popover"] > div > div,
        body [data-baseweb="popover"] [class],
        body [data-baseweb="select-dropdown"],
        body [data-baseweb="select-dropdown"] > div,
        body [data-baseweb="select-dropdown"] [class],
        body [data-baseweb="menu"],
        body [data-baseweb="menu"] > div,
        body [data-baseweb="menu"] [class],
        body [role="presentation"],
        body [role="presentation"] > div,
        body [role="presentation"] > div > div,
        body [role="presentation"] [class],
        body [role="listbox"],
        body [role="listbox"] [class],
        body [role="option"],
        body [role="option"] [class] {
            background: #0d151f !important;
            background-color: #0d151f !important;
            background-image: none !important;
            border-color: rgba(136, 160, 181, 0.32) !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
            box-shadow: none !important;
        }

        body [data-baseweb="popover"],
        body [data-baseweb="select-dropdown"],
        body [data-baseweb="menu"],
        body [role="presentation"] {
            box-shadow: 0 22px 54px rgba(0, 0, 0, 0.44) !important;
        }

        body [role="option"]:hover,
        body [role="option"][aria-selected="true"],
        body [role="option"][aria-current="true"],
        body [data-highlighted="true"],
        body [aria-selected="true"],
        body [data-baseweb="select-dropdown"] li:hover,
        body [data-baseweb="menu"] li:hover {
            background: rgba(53, 211, 173, 0.22) !important;
            background-color: rgba(53, 211, 173, 0.22) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        body div[data-testid="stDataFrame"],
        body div[data-testid="stTable"],
        body div[data-testid="stDataEditor"],
        body div[data-testid="stDataFrameResizable"],
        body div[data-testid="stDataFrameGlideDataEditor"],
        body div[data-testid="stDataFrame"] [class],
        body div[data-testid="stTable"] [class],
        body div[data-testid="stDataEditor"] [class],
        body div[data-testid="stDataFrameResizable"] [class],
        body div[data-testid="stDataFrameGlideDataEditor"] [class],
        body [role="grid"],
        body [role="row"],
        body [role="gridcell"],
        body [role="columnheader"] {
            background: #0d151f !important;
            background-color: #0d151f !important;
            background-image: none !important;
            border-color: rgba(136, 160, 181, 0.22) !important;
            color: #edf5ff !important;
            -webkit-text-fill-color: #edf5ff !important;
        }

        body .dashboard-dark-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin: 0.35rem 0 1rem;
            border: 1px solid rgba(136, 160, 181, 0.24);
            border-radius: 12px;
            background: linear-gradient(145deg, rgba(18, 28, 39, 0.96), rgba(9, 15, 23, 0.98));
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
        }

        body table.dashboard-dark-table {
            width: 100%;
            min-width: 820px;
            border-collapse: collapse;
            color: #edf5ff !important;
            background: transparent !important;
        }

        body table.dashboard-dark-table thead th {
            padding: 0.78rem 0.9rem;
            background: rgba(53, 211, 173, 0.12) !important;
            border-bottom: 1px solid rgba(136, 160, 181, 0.24);
            color: #dffcf5 !important;
            font-weight: 800;
            text-align: left;
            white-space: nowrap;
        }

        body table.dashboard-dark-table tbody td {
            padding: 0.72rem 0.9rem;
            background: rgba(13, 21, 31, 0.92) !important;
            border-bottom: 1px solid rgba(136, 160, 181, 0.16);
            color: #edf5ff !important;
            vertical-align: top;
        }

        body table.dashboard-dark-table tbody tr:nth-child(even) td {
            background: rgba(18, 28, 39, 0.92) !important;
        }

        body table.dashboard-dark-table tbody tr:hover td {
            background: rgba(53, 211, 173, 0.12) !important;
        }

        @keyframes darkGridReveal {
            0% {
                opacity: 0;
            }
            100% {
                opacity: 1;
            }
        }

        @keyframes darkGridDrift {
            0% {
                background-position: 0 0, 0 0;
            }
            100% {
                background-position: 54px 54px, -54px 54px;
            }
        }

        @keyframes crystalFacetShift {
            0% {
                background-position: 0 0, 0 0, -20vw 0;
            }
            100% {
                background-position: 180px 120px, -160px 190px, 120vw 0;
            }
        }

        @keyframes crystalPanelSweep {
            0%, 42% {
                opacity: 0;
                transform: translateX(-100%) rotate(10deg);
            }
            55% {
                opacity: 0.95;
            }
            100% {
                opacity: 0;
                transform: translateX(230%) rotate(10deg);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_symptom_label(symptom_name):
    return str(symptom_name).replace("_", " ").strip().title()


@st.cache_data
def get_image_base64(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def normalize_text_for_matching(text):
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", str(text).lower().replace("_", " "))
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_location_query(text):
    normalized = normalize_text_for_matching(text)
    for alias, canonical in LOCATION_ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(alias)}\b", canonical, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


LOCATION_ADMIN_TOKENS = ("taluk", "taluka", "tehsil", "mandal", "division", "postal code")


def clean_location_display_part(part):
    cleaned = str(part or "").strip()
    normalized_part = normalize_text_for_matching(cleaned)
    if not normalized_part or normalized_part == "india" or normalized_part.isdigit():
        return ""
    if any(token in normalized_part for token in LOCATION_ADMIN_TOKENS):
        return ""
    if "district" in normalized_part:
        cleaned = re.sub(r"\bdistrict\b", "", cleaned, flags=re.IGNORECASE).strip(" ,-")
        normalized_part = normalize_text_for_matching(cleaned)
        if not normalized_part:
            return ""
    return cleaned


def format_location_display_name(place):
    if isinstance(place, dict):
        address = place.get("address", {}) or {}
        location_groups = (
            (
                address.get("neighbourhood"),
                address.get("suburb"),
                address.get("city_district"),
                address.get("locality"),
                address.get("village"),
                address.get("town"),
                address.get("city"),
                address.get("municipality"),
            ),
            (
                address.get("city"),
                address.get("town"),
                address.get("municipality"),
                address.get("county"),
                address.get("state_district"),
            ),
            (address.get("state"),),
        )
        parts = []
        for group in location_groups:
            for candidate in group:
                part = clean_location_display_part(candidate)
                if part and normalize_text_for_matching(part) not in {normalize_text_for_matching(item) for item in parts}:
                    parts.append(part)
                    break
        if parts:
            return ", ".join(parts[:3])
        raw_display_name = str(place.get("display_name", "")).strip()
    else:
        raw_display_name = str(place or "").strip()

    raw_parts = [part.strip() for part in raw_display_name.split(",") if part.strip()]
    filtered_parts = []
    for part in raw_parts:
        part = clean_location_display_part(part)
        normalized_part = normalize_text_for_matching(part)
        if not part:
            continue
        if normalized_part in {normalize_text_for_matching(item) for item in filtered_parts}:
            continue
        filtered_parts.append(part)

    if len(filtered_parts) > 3:
        return ", ".join([filtered_parts[0], filtered_parts[1], filtered_parts[-1]])
    return ", ".join(filtered_parts[:3]) or raw_display_name


def get_city_suggestions(query, limit=30):
    normalized_query = normalize_location_query(query)
    if len(normalized_query) < 2:
        return []

    query_tokens = [token for token in normalized_query.split() if len(token) > 1]
    ranked_matches = []
    for city in CITY_OPTIONS:
        normalized_city = normalize_location_query(city)
        city_tokens = normalized_city.split()
        if normalized_city.startswith(normalized_query):
            ranked_matches.append((0, city))
        elif normalized_query in normalized_city:
            ranked_matches.append((1, city))
        elif query_tokens and all(token in city_tokens for token in query_tokens):
            ranked_matches.append((2, city))
        elif query_tokens and any(city_token.startswith(query_tokens[-1]) for city_token in city_tokens):
            ranked_matches.append((3, city))

    ranked_matches.sort(key=lambda item: (item[0], item[1]))
    suggestions = [city for _, city in ranked_matches[:limit]]

    remote_queries = [normalized_query]
    if len(query_tokens) > 1:
        remote_queries.append(" ".join(reversed(query_tokens)))
        remote_queries.append(query_tokens[-1])

    for remote_query in remote_queries:
        for remote_suggestion in fetch_india_location_suggestions(remote_query, limit=limit):
            if remote_suggestion not in suggestions:
                suggestions.append(remote_suggestion)
            if len(suggestions) >= limit:
                break
        if len(suggestions) >= limit:
            break

    return suggestions[:limit]


def render_action_link(label, url, description, accent="blue"):
    st.markdown(
        f"""
        <a class="doctor-action-link doctor-action-{html.escape(accent)}"
           href="{html.escape(url, quote=True)}"
           target="_blank"
           rel="noopener noreferrer">
            <span class="doctor-action-dot"></span>
            <span class="doctor-action-copy">
                <strong>{html.escape(label)}</strong>
                <em>{html.escape(description)}</em>
            </span>
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_action_card(label, description, accent="blue"):
    st.markdown(
        f"""
        <div class="doctor-action-card doctor-action-{html.escape(accent)}">
            <span class="doctor-action-dot"></span>
            <span class="doctor-action-copy">
                <strong>{html.escape(label)}</strong>
                <em>{html.escape(description)}</em>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fetch_json_payload(url, data=None, timeout=20, content_type=None):
    headers = {
        "User-Agent": "DiseasePredictionDashboard/1.0 educational Streamlit app",
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = content_type

    request = Request(url, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except HTTPError as exc:
        return None, f"Public map service returned HTTP {exc.code}."
    except URLError:
        return None, "Public map service could not be reached."
    except json.JSONDecodeError:
        return None, "Public map service returned an unreadable response."
    except Exception as exc:
        return None, f"Location lookup failed: {exc}"


def get_configured_mapbox_access_token():
    env_key = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
    if env_key:
        return env_key

    try:
        return str(st.secrets.get("MAPBOX_ACCESS_TOKEN", "")).strip()
    except Exception:
        return ""


def get_configured_here_api_key():
    env_key = os.getenv("HERE_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        return str(st.secrets.get("HERE_API_KEY", "")).strip()
    except Exception:
        return ""


def get_map_provider_cache_token():
    configured_keys = (
        ("mapbox", get_configured_mapbox_access_token()),
        ("here", get_configured_here_api_key()),
    )
    token_parts = [
        f"{name}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"
        for name, value in configured_keys
        if value
    ]
    return "|".join(token_parts) if token_parts else "public-map-sources"


def extract_map_api_error(payload, provider_name, fallback_message):
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = clean_provider_text(error.get("message"))
            status = clean_provider_text(error.get("status")) or clean_provider_text(error.get("code"))
            if message and status:
                return f"{provider_name} returned {status}: {message}"
            if message:
                return f"{provider_name} returned: {message}"
        if isinstance(error, str):
            return f"{provider_name} returned: {clean_provider_text(error)}"
        for key in ("message", "error_message", "title", "cause", "action"):
            message = clean_provider_text(payload.get(key))
            if message:
                return f"{provider_name} returned: {message}"
    return fallback_message


def fetch_map_provider_payload(url, provider_name, timeout=18):
    request = Request(
        url,
        headers={
            "User-Agent": "DiseasePredictionDashboard/1.0 educational Streamlit app",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(raw_error)
        except json.JSONDecodeError:
            error_payload = {"error_message": raw_error}
        return None, extract_map_api_error(
            error_payload,
            provider_name,
            f"{provider_name} returned HTTP {exc.code}.",
        )
    except URLError:
        return None, f"{provider_name} could not be reached."
    except json.JSONDecodeError:
        return None, f"{provider_name} returned an unreadable response."
    except Exception as exc:
        return None, f"{provider_name} lookup failed: {exc}"


def geocode_location_with_mapbox(search_query, access_token):
    params = {
        "q": search_query,
        "access_token": access_token,
        "country": "in",
        "language": "en",
        "limit": 1,
    }
    payload, error = fetch_map_provider_payload(
        f"{MAPBOX_GEOCODING_URL}?{urlencode(params)}",
        "Mapbox Geocoding",
        timeout=14,
    )
    if error:
        return None, error
    if not isinstance(payload, dict):
        return None, "Mapbox Geocoding returned an unreadable response."

    features = payload.get("features", [])
    if not features:
        return None, "Mapbox Geocoding did not find this city or area."

    feature = features[0]
    properties = feature.get("properties", {}) or {}
    coordinates = properties.get("coordinates", {}) or {}
    try:
        lat = float(coordinates["latitude"])
        lon = float(coordinates["longitude"])
    except (KeyError, TypeError, ValueError):
        geometry_coordinates = feature.get("geometry", {}).get("coordinates", [])
        try:
            lon = float(geometry_coordinates[0])
            lat = float(geometry_coordinates[1])
        except (IndexError, TypeError, ValueError):
            return None, "Mapbox location match did not include usable coordinates."

    name = clean_provider_text(properties.get("name"))
    place = (
        clean_provider_text(properties.get("full_address"))
        or clean_provider_text(properties.get("place_formatted"))
        or clean_provider_text(feature.get("place_name"))
    )
    display_name = ", ".join(part for part in (name, place) if part) or clean_provider_text(search_query, "Selected location")
    return {
        "lat": lat,
        "lon": lon,
        "display_name": display_name,
        "source": "Mapbox Geocoding",
    }, ""


def geocode_location_with_here(search_query, api_key):
    params = {
        "q": search_query,
        "apikey": api_key,
        "in": "countryCode:IND",
        "lang": "en-US",
        "limit": 1,
    }
    payload, error = fetch_map_provider_payload(
        f"{HERE_GEOCODE_URL}?{urlencode(params)}",
        "HERE Geocoding",
        timeout=14,
    )
    if error:
        return None, error
    if not isinstance(payload, dict):
        return None, "HERE Geocoding returned an unreadable response."

    items = payload.get("items", [])
    if not items:
        return None, "HERE Geocoding did not find this city or area."

    match = items[0]
    position = match.get("position", {}) or {}
    try:
        lat = float(position["lat"])
        lon = float(position["lng"])
    except (KeyError, TypeError, ValueError):
        return None, "HERE location match did not include usable coordinates."

    display_name = (
        clean_provider_text(match.get("title"))
        or clean_provider_text((match.get("address") or {}).get("label"))
        or clean_provider_text(search_query, "Selected location")
    )
    return {
        "lat": lat,
        "lon": lon,
        "display_name": display_name,
        "source": "HERE Geocoding",
    }, ""


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_india_location_suggestions(location_query, limit=20):
    cleaned_query = normalize_location_query(location_query)
    if len(cleaned_query) < 2:
        return []

    url = (
        "https://nominatim.openstreetmap.org/search"
        f"?format=json&limit={int(limit)}&addressdetails=1&countrycodes=in&q={quote_plus(cleaned_query)}"
    )
    payload, error = fetch_json_payload(url, timeout=8)
    if error or not payload:
        return []

    allowed_classes = {"place", "boundary"}
    allowed_types = {
        "administrative",
        "city",
        "city_district",
        "county",
        "hamlet",
        "locality",
        "municipality",
        "neighbourhood",
        "residential",
        "state",
        "state_district",
        "suburb",
        "town",
        "village",
    }
    suggestions = []
    seen = set()
    for place in payload:
        display_name = str(place.get("display_name", "")).strip()
        normalized_display = normalize_text_for_matching(display_name)
        if not display_name or "india" not in normalized_display:
            continue
        place_class = str(place.get("class", "")).lower()
        place_type = str(place.get("type", "")).lower()
        address_type = str(place.get("addresstype", "")).lower()
        if (
            place_class not in allowed_classes
            and place_type not in allowed_types
            and address_type not in allowed_types
        ):
            continue
        clean_display_name = format_location_display_name(place)
        normalized_clean_display = normalize_location_query(clean_display_name)
        if not clean_display_name or normalized_clean_display in seen:
            continue
        suggestions.append(clean_display_name)
        seen.add(normalized_clean_display)
        if len(suggestions) >= limit:
            break
    return suggestions


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_location(location_query, maps_key_token=None):
    cleaned_query = normalize_location_query(location_query)
    if len(cleaned_query) < 2:
        return None, "Please enter at least 2 letters of a city or area."

    search_query = cleaned_query

    lookup_errors = []
    nominatim_url = (
        "https://nominatim.openstreetmap.org/search"
        f"?format=json&limit=1&addressdetails=1&q={quote_plus(search_query)}"
    )
    payload, error = fetch_json_payload(nominatim_url, timeout=14)
    if not error and payload:
        match = payload[0]
        display_name = format_location_display_name(match) or location_query
        try:
            return (
                {
                    "lat": float(match["lat"]),
                    "lon": float(match["lon"]),
                    "display_name": display_name,
                    "source": "OpenStreetMap Nominatim",
                },
                "",
            )
        except (KeyError, TypeError, ValueError):
            error = "The matched location did not include usable coordinates."
    if error:
        lookup_errors.append(error)

    photon_url = (
        "https://photon.komoot.io/api/"
        f"?q={quote_plus(search_query)}&limit=1&lang=en"
    )
    photon_payload, photon_error = fetch_json_payload(photon_url, timeout=14)
    if not photon_error and isinstance(photon_payload, dict):
        features = photon_payload.get("features", [])
        if features:
            feature = features[0]
            coordinates = feature.get("geometry", {}).get("coordinates", [])
            properties = feature.get("properties", {})
            try:
                lon, lat = float(coordinates[0]), float(coordinates[1])
                display_parts = [
                    clean_provider_text(properties.get("name")),
                    clean_provider_text(properties.get("city")),
                    clean_provider_text(properties.get("state")),
                    clean_provider_text(properties.get("country")),
                ]
                display_name = ", ".join(part for part in display_parts if part) or location_query
                return (
                    {
                        "lat": lat,
                        "lon": lon,
                        "display_name": display_name,
                        "source": "Photon",
                    },
                    "",
                )
            except (IndexError, TypeError, ValueError):
                photon_error = "The fallback location match did not include usable coordinates."
    if photon_error:
        lookup_errors.append(photon_error)

    mapbox_token = get_configured_mapbox_access_token()
    if mapbox_token:
        mapbox_location, mapbox_error = geocode_location_with_mapbox(search_query, mapbox_token)
        if mapbox_location:
            return mapbox_location, ""
        if mapbox_error:
            lookup_errors.append(mapbox_error)

    here_api_key = get_configured_here_api_key()
    if here_api_key:
        here_location, here_error = geocode_location_with_here(search_query, here_api_key)
        if here_location:
            return here_location, ""
        if here_error:
            lookup_errors.append(here_error)

    if lookup_errors:
        return None, " ".join(dict.fromkeys(lookup_errors))
    return None, "No matching city or area was found."


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_healthcare_elements(lat, lon, radius_m=5000, search_mode="all"):
    search_mode = clean_provider_text(search_mode, "all").lower()
    hospital_queries = f"""
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["healthcare"="hospital"](around:{radius_m},{lat},{lon});
      way["healthcare"="hospital"](around:{radius_m},{lat},{lon});
      relation["healthcare"="hospital"](around:{radius_m},{lat},{lon});
    """
    doctor_queries = f"""
      node["amenity"~"doctors|clinic"](around:{radius_m},{lat},{lon});
      way["amenity"~"doctors|clinic"](around:{radius_m},{lat},{lon});
      relation["amenity"~"doctors|clinic"](around:{radius_m},{lat},{lon});
      node["healthcare"~"doctor|clinic"](around:{radius_m},{lat},{lon});
      way["healthcare"~"doctor|clinic"](around:{radius_m},{lat},{lon});
      relation["healthcare"~"doctor|clinic"](around:{radius_m},{lat},{lon});
    """
    lab_queries = f"""
      node["healthcare"~"laboratory|diagnostic|sample_collection|imaging|radiology"](around:{radius_m},{lat},{lon});
      way["healthcare"~"laboratory|diagnostic|sample_collection|imaging|radiology"](around:{radius_m},{lat},{lon});
      relation["healthcare"~"laboratory|diagnostic|sample_collection|imaging|radiology"](around:{radius_m},{lat},{lon});
      node["amenity"~"laboratory"](around:{radius_m},{lat},{lon});
      way["amenity"~"laboratory"](around:{radius_m},{lat},{lon});
      relation["amenity"~"laboratory"](around:{radius_m},{lat},{lon});
      node["name"~"diagnostic|diagnostics|pathology|laboratory|labs|(^| )lab($| )|scan|imaging|radiology|mri|ct scan|x ray", i](around:{radius_m},{lat},{lon});
      way["name"~"diagnostic|diagnostics|pathology|laboratory|labs|(^| )lab($| )|scan|imaging|radiology|mri|ct scan|x ray", i](around:{radius_m},{lat},{lon});
      relation["name"~"diagnostic|diagnostics|pathology|laboratory|labs|(^| )lab($| )|scan|imaging|radiology|mri|ct scan|x ray", i](around:{radius_m},{lat},{lon});
    """
    if search_mode == "hospitals":
        query_body = hospital_queries
    elif search_mode == "doctors":
        query_body = doctor_queries
    elif search_mode == "labs":
        query_body = lab_queries
    else:
        query_body = hospital_queries + doctor_queries
    overpass_query = f"""
    [out:json][timeout:18];
    (
    {query_body}
    );
    out center tags {HEALTHCARE_MAX_RESULTS};
    """
    data = f"data={quote_plus(overpass_query)}".encode("utf-8")
    overpass_endpoints = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )
    last_error = ""
    for endpoint in overpass_endpoints:
        payload, error = fetch_json_payload(
            endpoint,
            data=data,
            timeout=20,
            content_type="application/x-www-form-urlencoded; charset=UTF-8",
        )
        if not error and isinstance(payload, dict):
            return payload.get("elements", []), ""
        last_error = error
    return [], last_error or "Public healthcare listing service did not return results."


def calculate_distance_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * earth_radius_km * asin(sqrt(a))


def classify_healthcare_provider(tags):
    amenity = str(tags.get("amenity", "")).lower()
    healthcare = str(tags.get("healthcare", "")).lower()
    name = str(tags.get("name", "")).lower()
    speciality = str(tags.get("healthcare:speciality", "") or tags.get("speciality", "")).lower()
    combined = f"{amenity} {healthcare} {name} {speciality}"
    lab_terms = ("diagnostic", "diagnostics", "pathology", "laboratory", " labs", " lab ", "scan", "imaging", "radiology", "mri", "ct scan", "x ray")
    if any(term in combined for term in lab_terms):
        return "Diagnostic Lab"
    if "hospital" in combined:
        return "Hospital"
    if "clinic" in combined:
        return "Clinic"
    if "dentist" in combined:
        return "Dentist"
    if "doctor" in combined or "doctors" in combined:
        return "Doctor"
    return "Healthcare"


def clean_provider_text(value, fallback=""):
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    normalized = normalize_text_for_matching(text)
    if normalized in {"none", "null", "nan", "na", "n a", "not available", "unknown"}:
        return fallback
    return text


def clean_provider_number(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def build_provider_address(tags):
    def _address_value(key):
        return clean_provider_text(tags.get(key))

    address_parts = [
        _address_value("addr:housename"),
        _address_value("addr:housenumber"),
        _address_value("addr:street"),
        _address_value("addr:suburb"),
        _address_value("addr:city"),
        _address_value("addr:state"),
    ]
    address = ", ".join(part for part in address_parts if part)
    return address or clean_provider_text(tags.get("addr:full")) or "Address not published in public map data"


def get_specialist_search_terms(specialist, disease_name):
    normalized_text = normalize_text_for_matching(f"{specialist} {disease_name}")
    term_map = {
        "general physician": ("general", "physician", "family", "medicine", "doctor"),
        "dermatologist": ("dermatology", "dermatologist", "skin"),
        "orthopedic": ("orthopedic", "orthopaedic", "ortho", "bone", "joint"),
        "neurologist": ("neurology", "neurologist", "brain", "nerve"),
        "gastroenterologist": ("gastroenterology", "gastro", "digestive", "stomach"),
        "pulmonologist": ("pulmonology", "pulmonologist", "chest", "respiratory", "lung"),
        "cardiologist": ("cardiology", "cardiologist", "heart"),
        "endocrinologist": ("endocrinology", "diabetes", "thyroid", "hormone"),
        "urologist": ("urology", "urologist", "urinary"),
        "hepatologist": ("hepatology", "liver", "gastro"),
        "infectious": ("infectious", "infection", "fever", "tropical"),
        "emergency": ("emergency", "trauma", "urgent", "casualty"),
        "rheumatologist": ("rheumatology", "rheumatologist", "arthritis"),
        "vascular": ("vascular", "vein", "surgery"),
    }
    disease_term_map = {
        "fever_infection": {
            "keys": (
                "dengue",
                "malaria",
                "typhoid",
                "chicken pox",
                "common cold",
                "pneumonia",
                "tuberculosis",
                "aids",
                "hepatitis",
                "gastroenteritis",
                "urinary tract infection",
                "fungal infection",
                "allergy",
            ),
            "terms": ("fever", "infection", "infectious", "medicine", "general", "physician", "emergency"),
        },
        "respiratory": {
            "keys": ("asthma", "pneumonia", "tuberculosis", "common cold"),
            "terms": ("respiratory", "pulmonology", "pulmonologist", "chest", "lung", "medicine"),
        },
        "skin": {
            "keys": ("acne", "impetigo", "psoriasis", "fungal infection", "drug reaction", "allergy", "chicken pox"),
            "terms": ("skin", "dermatology", "dermatologist", "allergy"),
        },
        "digestive": {
            "keys": ("gerd", "peptic", "gastroenteritis", "piles", "cholestasis"),
            "terms": ("gastro", "gastroenterology", "digestive", "stomach", "medicine"),
        },
        "liver": {
            "keys": ("hepatitis", "jaundice", "alcoholic hepatitis", "cholestasis"),
            "terms": ("liver", "hepatology", "gastro", "medicine"),
        },
        "cardiac": {
            "keys": ("heart attack", "hypertension"),
            "terms": ("cardiology", "cardiologist", "heart", "cardiac", "emergency"),
        },
        "endocrine": {
            "keys": ("diabetes", "hypoglycemia", "thyroid", "hyperthyroidism", "hypothyroidism"),
            "terms": ("endocrinology", "diabetes", "thyroid", "hormone", "medicine"),
        },
        "neuro": {
            "keys": ("migraine", "vertigo", "paralysis", "brain"),
            "terms": ("neurology", "neurologist", "brain", "nerve", "stroke"),
        },
        "orthopedic": {
            "keys": ("arthritis", "osteo", "spondylosis", "joint"),
            "terms": ("orthopedic", "orthopaedic", "ortho", "bone", "joint"),
        },
        "urinary": {
            "keys": ("urinary", "uti", "micturition"),
            "terms": ("urology", "urologist", "urinary", "kidney", "medicine"),
        },
    }
    terms = {token for token in normalized_text.split() if len(token) > 3}
    for key, aliases in term_map.items():
        if key in normalized_text:
            terms.update(aliases)
    for profile in disease_term_map.values():
        if any(key in normalized_text for key in profile["keys"]):
            terms.update(profile["terms"])
    return terms


def get_provider_care_focus(specialist, disease_name):
    normalized_text = normalize_text_for_matching(f"{specialist} {disease_name}")
    if any(term in normalized_text for term in ("dengue", "malaria", "typhoid", "fever", "infection", "common cold", "chicken pox")):
        return "Fever and infection care"
    if any(term in normalized_text for term in ("asthma", "pneumonia", "tuberculosis", "respiratory", "pulmonologist")):
        return "Respiratory care"
    if any(term in normalized_text for term in ("skin", "dermat", "acne", "psoriasis", "impetigo", "fungal")):
        return "Skin care"
    if any(term in normalized_text for term in ("heart", "cardio", "hypertension")):
        return "Heart care"
    if any(term in normalized_text for term in ("diabetes", "thyroid", "hypoglycemia", "endocr")):
        return "Endocrine care"
    if any(term in normalized_text for term in ("hepatitis", "jaundice", "liver", "cholestasis")):
        return "Liver care"
    if any(term in normalized_text for term in ("gastro", "gerd", "ulcer", "stomach", "piles")):
        return "Digestive care"
    if any(term in normalized_text for term in ("neuro", "migraine", "vertigo", "paralysis", "brain")):
        return "Neurology care"
    if any(term in normalized_text for term in ("ortho", "arthritis", "joint", "bone", "spondylosis")):
        return "Bone and joint care"
    if any(term in normalized_text for term in ("urinary", "urology", "kidney")):
        return "Urinary care"
    return "Relevant care"


def care_focus_key(care_focus):
    focus = normalize_text_for_matching(care_focus)
    if "fever and infection" in focus:
        return "general_infection"
    if "respiratory" in focus:
        return "respiratory"
    if "skin" in focus:
        return "skin"
    if "heart" in focus:
        return "cardiac"
    if "endocrine" in focus:
        return "endocrine"
    if "liver" in focus:
        return "liver"
    if "digestive" in focus:
        return "digestive"
    if "neurology" in focus:
        return "neuro"
    if "bone" in focus or "joint" in focus:
        return "orthopedic"
    if "urinary" in focus:
        return "urinary"
    return "general"


def infer_provider_focus_keys(provider_text):
    text = normalize_text_for_matching(provider_text)
    focus_terms = {
        "general": (
            "general",
            "multi specialty",
            "multispeciality",
            "multispecialty",
            "medical college",
            "medical centre",
            "medical center",
            "medicine",
            "emergency",
            "trauma",
            "district hospital",
            "government hospital",
            "community health",
        ),
        "respiratory": ("respiratory", "pulmonology", "pulmonologist", "chest", "lung", "tb", "tuberculosis"),
        "skin": ("skin", "dermatology", "dermatologist"),
        "cardiac": ("cardiac", "cardiology", "cardiologist", "heart"),
        "endocrine": ("endocrine", "diabetes", "thyroid", "hormone"),
        "liver": ("liver", "hepatology", "hepatitis", "gastro"),
        "digestive": ("gastro", "digestive", "stomach", "piles", "proctology", "colorectal"),
        "neuro": ("neuro", "neurology", "neurologist", "brain", "stroke"),
        "orthopedic": ("ortho", "orthopedic", "orthopaedic", "bone", "joint"),
        "urinary": ("urology", "urologist", "urinary", "kidney", "renal"),
        "eye": ("eye", "ophthalmology", "ophthalmologist", "vision", "retina"),
        "dental": ("dental", "dentist", "orthodont"),
        "maternity": ("maternity", "ivf", "fertility", "obstetric", "gynecology", "gynaecology"),
        "cosmetic": ("cosmetic", "aesthetic", "hair transplant", "laser"),
        "cancer": ("cancer", "oncology", "oncologist"),
        "diagnostic": ("diagnostic", "diagnostics", "pathology", "laboratory", " lab", "scan", "imaging", "radiology", "mri", "ct scan", "x ray"),
        "surgical": ("surgery", "surgical", "surgeon"),
    }
    matches = set()
    for key, terms in focus_terms.items():
        if any(term in text for term in terms):
            matches.add(key)
    return matches


def provider_care_precision(provider_text, provider_kind, specialist, disease_name):
    care_focus = get_provider_care_focus(specialist, disease_name)
    target_key = care_focus_key(care_focus)
    focus_keys = infer_provider_focus_keys(provider_text)
    unrelated_specialties = {"eye", "dental", "maternity", "cosmetic", "cancer", "diagnostic", "surgical"}

    if target_key in focus_keys:
        return 22, f"Precise {care_focus.lower()} match"
    if "general" in focus_keys:
        return 14, f"General hospital for {care_focus.lower()}"
    if provider_kind == "Hospital" and not focus_keys:
        return 8, f"Nearby hospital for {care_focus.lower()}"
    if focus_keys & unrelated_specialties and target_key not in focus_keys:
        return -28, "Specialty hospital outside this care focus"
    if focus_keys and target_key not in focus_keys:
        return -14, f"Limited {care_focus.lower()} match"
    return 0, f"Nearby {care_focus.lower()}"


def build_provider_map_url(name, address, lat, lon):
    provider_name = clean_provider_text(name)
    provider_address = clean_provider_text(address)
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError):
        lat_value = None
        lon_value = None

    if lat_value is not None and lon_value is not None:
        return f"https://www.openstreetmap.org/?mlat={lat_value:.6f}&mlon={lon_value:.6f}#map=17/{lat_value:.6f}/{lon_value:.6f}"
    if provider_name and provider_address:
        query = f"{provider_name}, {provider_address}"
    else:
        query = provider_name or provider_address or "hospital"
    return f"https://www.openstreetmap.org/search?query={quote_plus(query)}"


def build_osm_provider_url(element):
    element_type = clean_provider_text(element.get("type"))
    element_id = clean_provider_text(element.get("id"))
    if element_type in {"node", "way", "relation"} and element_id:
        return f"https://www.openstreetmap.org/{element_type}/{element_id}"
    return ""


def score_provider_match(provider_text, provider_kind, distance_km, specialist, disease_name):
    if provider_kind == "Diagnostic Lab":
        diagnostic_terms = ("diagnostic", "diagnostics", "pathology", "laboratory", " lab", "labs", "scan", "imaging", "radiology", "mri", "ct scan", "x ray")
        matched_diagnostic_terms = [term for term in diagnostic_terms if term in provider_text]
        disease_terms = get_specialist_search_terms(specialist, disease_name)
        matched_disease_terms = [
            term for term in disease_terms
            if term in provider_text and term not in {"care", "doctor", "doctors", "hospital", "nearby", "specialist"}
        ]
        distance_penalty = min(distance_km * 4.0, 35)
        diagnostic_bonus = min(len(matched_diagnostic_terms) * 12, 36)
        disease_bonus = min(len(matched_disease_terms) * 8, 20)
        score = max(30, min(99, 70 + diagnostic_bonus + disease_bonus - distance_penalty))
        return round(score), "Nearby diagnostic lab"

    terms = get_specialist_search_terms(specialist, disease_name)
    generic_terms = {"care", "clinic", "disease", "doctor", "doctors", "hospital", "nearby", "specialist"}
    meaningful_terms = {term for term in terms if term not in generic_terms and len(term) > 3}
    matched_terms = [term for term in meaningful_terms if term in provider_text]
    kind_bonus = {"Doctor": 22, "Clinic": 18, "Hospital": 14, "Dentist": 10}.get(provider_kind, 8)
    specialist_bonus = min(len(matched_terms) * 16, 40)
    precision_bonus, precision_label = provider_care_precision(provider_text, provider_kind, specialist, disease_name)
    distance_penalty = min(distance_km * 4.5, 35)
    base_score = 64 if matched_terms else 54
    score = max(20, min(99, base_score + kind_bonus + specialist_bonus + precision_bonus - distance_penalty))
    care_focus = get_provider_care_focus(specialist, disease_name)
    if precision_bonus > 0:
        match_label = precision_label
    elif matched_terms:
        match_label = f"{care_focus} match"
    else:
        match_label = precision_label
    return round(score), match_label


def extract_provider_from_osm(element, origin_lat, origin_lon, specialist, disease_name):
    tags = element.get("tags", {}) or {}
    name = (
        clean_provider_text(tags.get("name"))
        or clean_provider_text(tags.get("official_name"))
        or clean_provider_text(tags.get("operator"))
    )
    if not name:
        return None

    center = element.get("center", {}) or {}
    try:
        lat = float(element.get("lat", center.get("lat")))
        lon = float(element.get("lon", center.get("lon")))
    except (TypeError, ValueError):
        return None

    kind = classify_healthcare_provider(tags)
    distance_km = calculate_distance_km(origin_lat, origin_lon, lat, lon)
    specialty_text = " ".join(
        clean_provider_text(tags.get(key))
        for key in ("healthcare:speciality", "speciality", "description", "operator", "name")
        if clean_provider_text(tags.get(key))
    )
    tag_value_text = " ".join(clean_provider_text(value) for value in tags.values() if clean_provider_text(value))
    provider_text = normalize_text_for_matching(
        f"{name} {specialty_text} {tag_value_text}"
    )
    score, match_label = score_provider_match(provider_text, kind, distance_km, specialist, disease_name)

    provider_name = clean_provider_text(name, "Healthcare provider")
    provider_address = build_provider_address(tags)
    return {
        "name": provider_name,
        "kind": kind,
        "address": provider_address,
        "distance_km": round(distance_km, 1),
        "score": score,
        "match_label": match_label,
        "phone": (
            clean_provider_text(tags.get("contact:phone"))
            or clean_provider_text(tags.get("phone"))
            or clean_provider_text(tags.get("mobile"))
        ),
        "website": clean_provider_text(tags.get("contact:website")) or clean_provider_text(tags.get("website")),
        "opening_hours": clean_provider_text(tags.get("opening_hours")),
        "speciality": (
            clean_provider_text(tags.get("healthcare:speciality"))
            or clean_provider_text(tags.get("speciality"))
            or clean_provider_text(specialist, "Healthcare")
        ),
        "lat": lat,
        "lon": lon,
        "maps_url": build_provider_map_url(provider_name, provider_address, lat, lon),
        "osm_url": build_osm_provider_url(element),
        "location_label": "Map coordinate available",
        "source": "OpenStreetMap",
        "search_text": provider_text,
    }


def extract_lab_provider_from_photon(feature, origin_lat, origin_lon, specialist, disease_name):
    properties = feature.get("properties", {}) or {}
    geometry = feature.get("geometry", {}) or {}
    coordinates = geometry.get("coordinates", [])
    provider_name = clean_provider_text(properties.get("name"))
    if not provider_name:
        return None

    country_code = clean_provider_text(properties.get("countrycode")).upper()
    if country_code and country_code != "IN":
        return None

    provider_text = normalize_text_for_matching(
        " ".join(
            clean_provider_text(properties.get(key))
            for key in ("name", "street", "district", "city", "state", "country", "osm_value")
            if clean_provider_text(properties.get(key))
        )
    )
    diagnostic_terms = (
        "diagnostic",
        "diagnostics",
        "pathology",
        "radiology",
        "imaging",
        "scan",
        "mri",
        "ct scan",
        "x ray",
        "blood bank",
        "collection center",
        "laboratory",
    )
    non_medical_terms = (
        "engineering",
        "electronics",
        "material testing",
        "physics",
        "research laboratory",
        "school",
        "college",
        "university",
    )
    if not any(term in provider_text for term in diagnostic_terms):
        return None
    if any(term in provider_text for term in non_medical_terms):
        return None

    try:
        lon, lat = float(coordinates[0]), float(coordinates[1])
    except (IndexError, TypeError, ValueError):
        return None

    address_parts = [
        clean_provider_text(properties.get("housenumber")),
        clean_provider_text(properties.get("street")),
        clean_provider_text(properties.get("district")),
        clean_provider_text(properties.get("city")),
        clean_provider_text(properties.get("state")),
    ]
    provider_address = ", ".join(part for part in address_parts if part) or "Address not published in public map data"
    distance_km = calculate_distance_km(origin_lat, origin_lon, lat, lon)
    score, match_label = score_provider_match(provider_text, "Diagnostic Lab", distance_km, specialist, disease_name)

    osm_url = ""
    osm_type = clean_provider_text(properties.get("osm_type")).lower()
    osm_id = clean_provider_text(properties.get("osm_id"))
    osm_type_map = {"n": "node", "w": "way", "r": "relation"}
    if osm_type in osm_type_map and osm_id:
        osm_url = f"https://www.openstreetmap.org/{osm_type_map[osm_type]}/{osm_id}"

    return {
        "name": provider_name,
        "kind": "Diagnostic Lab",
        "address": provider_address,
        "distance_km": round(distance_km, 1),
        "score": score,
        "match_label": match_label,
        "phone": "",
        "website": "",
        "opening_hours": "",
        "speciality": "Diagnostic lab",
        "lat": lat,
        "lon": lon,
        "maps_url": build_provider_map_url(provider_name, provider_address, lat, lon),
        "osm_url": osm_url,
        "location_label": "Map coordinate available",
        "source": "Photon",
        "search_text": provider_text,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def search_diagnostic_labs_with_photon(location_query, origin_lat, origin_lon, specialist, disease_name):
    cleaned_location = clean_provider_text(location_query)
    if not cleaned_location:
        return [], ""

    queries = (
        f"diagnostic lab {cleaned_location} India",
        f"pathology laboratory {cleaned_location} India",
        f"radiology imaging {cleaned_location} India",
    )
    providers = []
    errors = []
    for query in queries:
        url = (
            "https://photon.komoot.io/api/"
            f"?q={quote_plus(query)}&limit=8&lang=en"
        )
        payload, error = fetch_json_payload(url, timeout=14)
        if error:
            errors.append(error)
            continue
        if not isinstance(payload, dict):
            continue
        for feature in payload.get("features", []):
            provider = extract_lab_provider_from_photon(
                feature,
                origin_lat,
                origin_lon,
                specialist,
                disease_name,
            )
            if provider:
                providers.append(provider)

    providers = dedupe_provider_results(providers)
    providers.sort(key=lambda item: (-item["score"], item["distance_km"], item["name"]))
    return providers[:10], errors[-1] if errors and not providers else ""


def get_here_category_names(item):
    categories = item.get("categories", [])
    if not isinstance(categories, list):
        return []
    return [
        clean_provider_text(category.get("name"))
        for category in categories
        if isinstance(category, dict) and clean_provider_text(category.get("name"))
    ]


def classify_here_place_provider(item, search_mode):
    category_names = get_here_category_names(item)
    address_label = clean_provider_text((item.get("address") or {}).get("label"))
    combined = normalize_text_for_matching(
        f"{item.get('title', '')} {address_label} {' '.join(category_names)} {item.get('resultType', '')}"
    )
    if any(term in combined for term in ("medical lab", "diagnostic", "pathology", "laboratory", "radiology", "imaging", "scan")):
        return "Diagnostic Lab"
    if "hospital" in combined:
        return "Hospital"
    if "doctor" in combined or "physician" in combined:
        return "Doctor"
    if "clinic" in combined or "health center" in combined or "health centre" in combined:
        return "Clinic"
    if search_mode == "hospitals":
        return "Hospital"
    if search_mode == "labs":
        return "Diagnostic Lab"
    if search_mode == "doctors":
        return "Clinic"
    return "Healthcare"


def get_here_opening_summary(item):
    opening = item.get("openingHours")
    if isinstance(opening, dict):
        opening_entries = [opening]
    elif isinstance(opening, list):
        opening_entries = [entry for entry in opening if isinstance(entry, dict)]
    else:
        opening_entries = []

    for entry in opening_entries:
        if entry.get("isOpen") is True:
            return "Open now"
        if entry.get("isOpen") is False:
            return "Closed now"
        text_value = entry.get("text")
        if isinstance(text_value, list) and text_value:
            return clean_provider_text(text_value[0], "Hours listed")
        if clean_provider_text(text_value):
            return clean_provider_text(text_value, "Hours listed")
    return ""


def get_here_contact_details(item):
    phone = ""
    website = ""
    contacts = item.get("contacts", [])
    if not isinstance(contacts, list):
        return phone, website

    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        for phone_entry in contact.get("phone", []) or []:
            if isinstance(phone_entry, dict):
                phone = phone or clean_provider_text(phone_entry.get("value"))
            else:
                phone = phone or clean_provider_text(phone_entry)
        for web_entry in contact.get("www", []) or []:
            if isinstance(web_entry, dict):
                website = website or clean_provider_text(web_entry.get("value"))
            else:
                website = website or clean_provider_text(web_entry)
        if phone and website:
            break
    return phone, website


def extract_provider_from_here_place(item, origin_lat, origin_lon, specialist, disease_name, search_mode):
    if not isinstance(item, dict):
        return None

    provider_name = clean_provider_text(item.get("title"))
    if not provider_name:
        return None

    position = item.get("position", {}) or {}
    try:
        lat = float(position["lat"])
        lon = float(position["lng"])
    except (KeyError, TypeError, ValueError):
        return None

    category_names = get_here_category_names(item)
    provider_address = clean_provider_text(
        (item.get("address") or {}).get("label"),
        "Address not published by map provider",
    )
    kind = classify_here_place_provider(item, search_mode)
    primary_type = category_names[0] if category_names else clean_provider_text(specialist, "Healthcare")
    provider_text = normalize_text_for_matching(
        f"{provider_name} {provider_address} {primary_type} {' '.join(category_names)}"
    )
    distance_km = calculate_distance_km(origin_lat, origin_lon, lat, lon)
    score, match_label = score_provider_match(provider_text, kind, distance_km, specialist, disease_name)
    rating = clean_provider_number(item.get("averageRating") or item.get("rating"), None)
    if rating:
        score = min(99, score + min(max(rating - 3.5, 0) * 4, 8))

    phone, website = get_here_contact_details(item)
    return {
        "name": provider_name,
        "kind": kind,
        "address": provider_address,
        "distance_km": round(distance_km, 1),
        "score": round(score),
        "match_label": match_label,
        "phone": phone,
        "website": website,
        "opening_hours": get_here_opening_summary(item),
        "speciality": primary_type,
        "lat": lat,
        "lon": lon,
        "maps_url": build_provider_map_url(provider_name, provider_address, lat, lon),
        "osm_url": "",
        "place_id": clean_provider_text(item.get("id")),
        "rating": rating,
        "user_rating_count": int(clean_provider_number(item.get("reviewCount"), 0)),
        "location_label": "HERE location match",
        "source": "HERE",
        "search_text": provider_text,
    }


def build_healthcare_search_queries(location_query, specialist, disease_name, search_mode):
    location_label = clean_provider_text(location_query, "nearby")
    specialist_label = clean_provider_text(specialist, "doctor")
    disease_label = clean_provider_text(disease_name)
    care_focus = get_provider_care_focus(specialist, disease_name)

    if search_mode == "hospitals":
        queries = [
            f"{specialist_label} hospital near {location_label}",
            f"{care_focus} hospital near {location_label}",
            f"best hospital near {location_label}",
        ]
    elif search_mode == "doctors":
        queries = [
            f"{specialist_label} doctor near {location_label}",
            f"{specialist_label} clinic near {location_label}",
            f"{care_focus} doctor near {location_label}",
        ]
    elif search_mode == "labs":
        queries = [
            f"diagnostic lab near {location_label}",
            f"pathology laboratory near {location_label}",
            f"radiology imaging center near {location_label}",
        ]
    else:
        queries = [
            f"{specialist_label} doctor near {location_label}",
            f"{specialist_label} hospital near {location_label}",
            f"{care_focus} healthcare near {location_label}",
        ]

    if disease_label and search_mode in {"all", "doctors", "hospitals"}:
        queries.insert(0, f"{disease_label} {specialist_label} near {location_label}")

    unique_queries = []
    seen = set()
    for query in queries:
        normalized_query = normalize_text_for_matching(query)
        if normalized_query and normalized_query not in seen:
            unique_queries.append(query)
            seen.add(normalized_query)
    return unique_queries


@st.cache_data(ttl=3600, show_spinner=False)
def search_healthcare_places_with_here(location_query, origin_lat, origin_lon, specialist, disease_name, search_mode="all", maps_key_token=None):
    api_key = get_configured_here_api_key()
    if not api_key:
        return [], "", 0

    providers = []
    errors = []
    search_radius_m = HEALTHCARE_SEARCH_RADII_M[-1]
    for query in build_healthcare_search_queries(location_query, specialist, disease_name, search_mode):
        params = {
            "q": query,
            "at": f"{float(origin_lat):.6f},{float(origin_lon):.6f}",
            "limit": 20,
            "lang": "en-US",
            "apikey": api_key,
        }
        response_payload, error = fetch_map_provider_payload(
            f"{HERE_DISCOVER_URL}?{urlencode(params)}",
            "HERE Search",
            timeout=18,
        )
        if error:
            errors.append(error)
            continue
        if not isinstance(response_payload, dict):
            continue

        for item in response_payload.get("items", []):
            provider = extract_provider_from_here_place(
                item,
                origin_lat,
                origin_lon,
                specialist,
                disease_name,
                search_mode,
            )
            if provider:
                providers.append(provider)

    providers = dedupe_provider_results(providers)
    care_focus = get_provider_care_focus(specialist, disease_name)
    if search_mode != "labs":
        providers = [
            provider for provider in providers if provider_is_relevant_for_care_focus(provider, care_focus, disease_name)
        ]
    providers.sort(key=lambda item: (-item["score"], item["distance_km"], item["name"]))
    return providers[:25], errors[-1] if errors and not providers else "", search_radius_m


def dedupe_provider_results(providers):
    unique = {}
    for provider in providers:
        place_id = clean_provider_text(provider.get("place_id"))
        if place_id:
            source_name = normalize_text_for_matching(provider.get("source")) or "map-provider"
            key = (source_name, place_id)
        else:
            key = (
                normalize_text_for_matching(provider["name"]),
                normalize_text_for_matching(provider["address"])[:50],
            )
        previous = unique.get(key)
        if not previous or (provider["score"], -provider["distance_km"]) > (previous["score"], -previous["distance_km"]):
            unique[key] = provider
    return list(unique.values())


def format_search_radius(radius_m):
    if not radius_m:
        return ""
    if radius_m >= 1000:
        return f"{radius_m / 1000:.0f} km"
    return f"{radius_m} m"


def get_compatible_provider_focuses(care_focus, disease_name):
    target_key = care_focus_key(care_focus)
    disease_text = normalize_text_for_matching(disease_name)
    compatible = {"general", target_key}
    if any(term in disease_text for term in ("common cold", "asthma", "pneumonia", "tuberculosis")):
        compatible.add("respiratory")
    if any(term in disease_text for term in ("gastro", "gerd", "peptic", "piles", "cholestasis")):
        compatible.update({"digestive", "surgical"})
    if any(term in disease_text for term in ("hepatitis", "jaundice")):
        compatible.update({"liver", "digestive"})
    if any(term in disease_text for term in ("urinary", "uti", "micturition")):
        compatible.add("urinary")
    if any(term in disease_text for term in ("arthritis", "osteo", "spondylosis", "joint")):
        compatible.update({"orthopedic", "surgical"})
    if any(term in disease_text for term in ("heart", "hypertension")):
        compatible.update({"cardiac", "surgical"})
    if any(term in disease_text for term in ("diabetes", "thyroid", "hypoglycemia")):
        compatible.add("endocrine")
    if any(term in disease_text for term in ("migraine", "vertigo", "paralysis", "brain")):
        compatible.add("neuro")
    if any(term in disease_text for term in ("acne", "impetigo", "psoriasis", "fungal", "drug reaction", "allergy", "chicken pox")):
        compatible.add("skin")
    return compatible


def provider_is_relevant_for_care_focus(provider, care_focus, disease_name=""):
    text = normalize_text_for_matching(
        f"{provider.get('name', '')} {provider.get('address', '')} {provider.get('search_text', '')}"
    )
    name_text = normalize_text_for_matching(provider.get("name", ""))
    focus_keys = infer_provider_focus_keys(text)
    compatible_focuses = get_compatible_provider_focuses(care_focus, disease_name)
    if provider.get("kind") == "Diagnostic Lab":
        return True
    if provider.get("kind") == "Hospital":
        return True

    focus = normalize_text_for_matching(care_focus)
    if "fever and infection" in focus:
        unrelated_terms = (
            "aesthetic",
            "cosmetic",
            "dental",
            "dentist",
            "hair",
            "ivf",
            "laser",
            "maternity",
            "skin",
            "sonography",
            "vision",
        )
        strong_terms = ("fever", "infection", "infectious", "general", "medicine", "physician")
        if any(term in text for term in unrelated_terms) and not any(term in text for term in strong_terms):
            return False
    return True


def split_provider_result_groups(providers):
    doctors = [
        provider
        for provider in providers
        if provider.get("kind") in {"Doctor", "Clinic", "Healthcare"}
    ]
    hospitals = [provider for provider in providers if provider.get("kind") == "Hospital"]
    labs = [provider for provider in providers if provider.get("kind") == "Diagnostic Lab"]
    doctors.sort(key=lambda item: (-item["score"], item["distance_km"], item["name"]))
    hospitals.sort(key=lambda item: (-item["score"], item["distance_km"], item["name"]))
    labs.sort(key=lambda item: (-item["score"], item["distance_km"], item["name"]))
    return doctors, hospitals, labs


def provider_groups_match_search_mode(search_mode, doctors, hospitals, labs):
    if search_mode == "doctors":
        return bool(doctors)
    if search_mode == "hospitals":
        return bool(hospitals)
    if search_mode == "labs":
        return bool(labs)
    return bool(doctors or hospitals or labs)


def build_healthcare_search_result(
    geocoded_location,
    providers,
    search_radius_m,
    search_mode,
    care_focus,
    source,
    error="",
):
    doctors, hospitals, labs = split_provider_result_groups(providers)
    return {
        "ok": bool(doctors or hospitals or labs),
        "error": error if not (doctors or hospitals or labs) else "",
        "origin": geocoded_location,
        "doctors": doctors,
        "hospitals": hospitals,
        "labs": labs,
        "search_radius_m": search_radius_m,
        "search_mode": search_mode,
        "care_focus": care_focus,
        "source": source,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def search_nearby_healthcare(location_query, specialist, disease_name, search_mode="all", maps_key_token=None):
    search_mode = clean_provider_text(search_mode, "all").lower()
    if search_mode not in {"all", "doctors", "hospitals", "labs"}:
        search_mode = "all"
    geocoded_location, geocode_error = geocode_location(location_query, maps_key_token)
    if geocode_error:
        return {
            "ok": False,
            "error": geocode_error,
            "origin": None,
            "doctors": [],
            "hospitals": [],
            "labs": [],
            "search_radius_m": 0,
            "search_mode": search_mode,
        }

    care_focus = "Diagnostic lab" if search_mode == "labs" else get_provider_care_focus(specialist, disease_name)
    provider_errors = []

    if search_mode == "labs":
        photon_labs, photon_error = search_diagnostic_labs_with_photon(
            location_query,
            geocoded_location["lat"],
            geocoded_location["lon"],
            specialist,
            disease_name,
        )
        if photon_labs:
            search_radius_m = int(max(provider.get("distance_km", 0) for provider in photon_labs) * 1000) or HEALTHCARE_SEARCH_RADII_M[0]
            return build_healthcare_search_result(
                geocoded_location,
                photon_labs,
                search_radius_m,
                search_mode,
                "Diagnostic lab",
                "Photon/OpenStreetMap public diagnostic listings",
            )
        if photon_error:
            provider_errors.append(photon_error)

    providers = []
    search_radius_m = HEALTHCARE_SEARCH_RADII_M[0]
    for radius_m in HEALTHCARE_SEARCH_RADII_M:
        elements, provider_error = fetch_healthcare_elements(
            geocoded_location["lat"],
            geocoded_location["lon"],
            radius_m=radius_m,
            search_mode=search_mode,
        )
        search_radius_m = radius_m
        if provider_error:
            provider_errors.append(provider_error)

        for element in elements:
            provider = extract_provider_from_osm(
                element,
                geocoded_location["lat"],
                geocoded_location["lon"],
                specialist,
                disease_name,
            )
            if provider:
                providers.append(provider)

        providers = dedupe_provider_results(providers)
        relevant_providers = [
            provider for provider in providers if provider_is_relevant_for_care_focus(provider, care_focus, disease_name)
        ]
        hospitals_found = any(provider["kind"] == "Hospital" for provider in relevant_providers)
        doctors_found = any(provider["kind"] in {"Doctor", "Clinic", "Healthcare"} for provider in relevant_providers)
        labs_found = any(provider["kind"] == "Diagnostic Lab" for provider in relevant_providers)
        if (
            (search_mode == "hospitals" and hospitals_found)
            or (search_mode == "doctors" and doctors_found)
            or (search_mode == "labs" and labs_found)
            or (search_mode == "all" and hospitals_found and doctors_found)
        ):
            break

    providers = [provider for provider in providers if provider_is_relevant_for_care_focus(provider, care_focus, disease_name)]
    doctors, hospitals, labs = split_provider_result_groups(providers)
    if provider_groups_match_search_mode(search_mode, doctors, hospitals, labs):
        return build_healthcare_search_result(
            geocoded_location,
            providers,
            search_radius_m,
            search_mode,
            care_focus,
            "OpenStreetMap public healthcare listings",
        )

    here_providers, here_error, here_radius_m = search_healthcare_places_with_here(
        location_query,
        geocoded_location["lat"],
        geocoded_location["lon"],
        specialist,
        disease_name,
        search_mode,
        maps_key_token,
    )
    if here_error:
        provider_errors.append(here_error)
    if here_providers:
        combined_providers = dedupe_provider_results(providers + here_providers)
        combined_providers = [
            provider for provider in combined_providers if provider_is_relevant_for_care_focus(provider, care_focus, disease_name)
        ]
        combined_doctors, combined_hospitals, combined_labs = split_provider_result_groups(combined_providers)
        if provider_groups_match_search_mode(search_mode, combined_doctors, combined_hospitals, combined_labs):
            return build_healthcare_search_result(
                geocoded_location,
                combined_providers,
                here_radius_m or search_radius_m,
                search_mode,
                care_focus,
                "OpenStreetMap public listings + HERE fallback",
            )

    return build_healthcare_search_result(
        geocoded_location,
        providers,
        search_radius_m,
        search_mode,
        care_focus,
        "OpenStreetMap public healthcare listings",
        provider_errors[-1] if provider_errors and not providers else "",
    )


def build_provider_card_html(provider):
    contact_lines = []
    provider_kind = clean_provider_text(provider.get("kind"), "Healthcare")
    provider_name = clean_provider_text(provider.get("name"), "Healthcare provider")
    provider_address = clean_provider_text(provider.get("address"), "Address not published in public map data")
    provider_match = clean_provider_text(provider.get("match_label"), "Nearby healthcare")
    provider_speciality = clean_provider_text(provider.get("speciality"), "Healthcare")
    location_label = clean_provider_text(provider.get("location_label"), "Map location available")
    maps_url = clean_provider_text(provider.get("maps_url")) or build_provider_map_url(
        provider_name,
        provider_address,
        provider.get("lat"),
        provider.get("lon"),
    )
    osm_url = clean_provider_text(provider.get("osm_url"))
    source_name = clean_provider_text(provider.get("source"))
    phone = clean_provider_text(provider.get("phone"))
    opening_hours = clean_provider_text(provider.get("opening_hours"))
    website = clean_provider_text(provider.get("website"))
    rating = clean_provider_number(provider.get("rating"), None)
    user_rating_count = int(clean_provider_number(provider.get("user_rating_count"), 0))
    if rating:
        if user_rating_count:
            contact_lines.append(f"Rating: {rating:.1f} ({user_rating_count} Google reviews)")
        else:
            contact_lines.append(f"Rating: {rating:.1f}")
    if phone:
        contact_lines.append(f"Phone: {html.escape(phone)}")
    if opening_hours:
        contact_lines.append(f"Hours: {html.escape(opening_hours)}")
    if website:
        website_url = website
        if website_url and not re.match(r"^https?://", website_url, flags=re.IGNORECASE):
            website_url = f"https://{website_url}"
        contact_lines.append("Website listed")
    if source_name:
        contact_lines.append(f"Source: {html.escape(source_name)}")
    if osm_url:
        contact_lines.append("OSM source")
    contact_html = "".join(f"<span>{line}</span>" for line in contact_lines)
    if not contact_html:
        contact_html = "<span>Contact details not published in map data</span>"

    if provider_kind == "Diagnostic Lab":
        map_cta = "Open lab location"
    elif provider_kind == "Hospital":
        map_cta = "Open hospital location"
    else:
        map_cta = "Open provider location"

    return (
        f"<a class='provider-card-link' href='{html.escape(maps_url, quote=True)}' target='_blank' rel='noopener noreferrer' "
        f"aria-label='Open {html.escape(provider_name, quote=True)} in maps'>"
        '<div class="provider-card provider-card--clickable">'
        '<div class="provider-card-top">'
        f'<span class="provider-chip">{html.escape(provider_kind)}</span>'
        f"<strong>{int(clean_provider_number(provider.get('score')))}%</strong>"
        "</div>"
        f"<h4>{html.escape(provider_name)}</h4>"
        f'<p class="provider-address">{html.escape(provider_address)}</p>'
        '<div class="provider-meta">'
        f"<span>{clean_provider_number(provider.get('distance_km')):.1f} km away</span>"
        f"<span>{html.escape(provider_match)}</span>"
        f"<span>{html.escape(provider_speciality)}</span>"
        f"<span>{html.escape(location_label)}</span>"
        "</div>"
        f'<div class="provider-contact">{contact_html}</div>'
        f"<div class='provider-map-cta'>{html.escape(map_cta)}</div>"
        "</div>"
        "</a>"
    )


def render_provider_results(title, providers, empty_message):
    st.markdown(f"<div class='provider-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
    if not providers:
        st.markdown(
            f"<div class='provider-empty'>{html.escape(empty_message)}</div>",
            unsafe_allow_html=True,
        )
        return

    cards_html = "".join(build_provider_card_html(provider) for provider in providers)
    st.markdown(f"<div class='provider-directory-grid'>{cards_html}</div>", unsafe_allow_html=True)


def format_provider_option(provider):
    provider_name = clean_provider_text(provider.get("name"), "Healthcare provider")
    provider_kind = clean_provider_text(provider.get("kind"), "Healthcare")
    return f"{provider_name} | {provider_kind} | {clean_provider_number(provider.get('distance_km')):.1f} km"


def build_booking_fields_from_provider(provider, specialist):
    provider_name = clean_provider_text(provider.get("name"))
    provider_kind = clean_provider_text(provider.get("kind"), "Healthcare")
    specialist_label = clean_provider_text(specialist, "Doctor")
    if not provider_name:
        return "", ""

    if provider_kind == "Hospital":
        return specialist_label, provider_name

    if provider_kind == "Clinic":
        return specialist_label, provider_name

    if provider_kind == "Healthcare":
        provider_text = normalize_text_for_matching(provider_name)
        if provider_text.startswith(("dr ", "doctor ")):
            return provider_name, ""
        if any(term in provider_text for term in ("hospital", "clinic", "medical centre", "medical center", "care", "health")):
            return specialist_label, provider_name
        return provider_name, ""

    return provider_name, ""


def get_active_booking_provider(current_provider_context):
    if st.session_state.selected_booking_provider_context != current_provider_context:
        return None
    return st.session_state.selected_booking_provider


def message_has_any_phrase(normalized_message, phrases):
    padded_message = f" {normalized_message} "
    for phrase in phrases:
        normalized_phrase = normalize_text_for_matching(phrase)
        if normalized_phrase and f" {normalized_phrase} " in padded_message:
            return True
    return False


@st.cache_data
def build_symptom_alias_map(feature_names_tuple):
    manual_aliases = {
        "high_fever": ["high fever", "high temperature", "very high fever", "temperature above 102"],
        "mild_fever": ["fever", "feaver", "feverish", "low fever", "light fever", "slight fever", "temperature", "temprature", "body hot", "hot body"],
        "stomach_pain": ["abdominal pain", "belly pain", "stomach ache", "stomach pain", "stomach is paining"],
        "abdominal_pain": ["abdominal pain", "belly pain", "stomach ache", "pain in stomach", "stomach pain", "stomach cramps"],
        "belly_pain": ["belly pain", "abdominal pain", "stomach ache"],
        "joint_pain": ["joint ache", "body joint pain"],
        "muscle_pain": ["body ache", "muscle ache", "muscle pain", "muscl pain", "body pain", "body is paining", "body paining", "whole body pain"],
        "breathlessness": ["shortness of breath", "short of breath", "difficulty breathing", "breathing difficulty", "breathing problem", "breathing issue", "problem breathing", "unable to breathe", "can't breathe", "cant breathe"],
        "chest_pain": ["pain in chest", "chest tightness", "chest is paining", "chest paining", "heart pain"],
        "continuous_sneezing": ["sneezing", "sneeze", "frequent sneezing"],
        "vomiting": ["throwing up", "throw up", "puking"],
        "nausea": ["feeling nauseous", "nauseous"],
        "diarrhoea": ["diarrhea", "loose motion", "loose motions", "loose stool", "loose stools", "watery stool"],
        "constipation": ["hard stool", "difficulty passing stool"],
        "headache": ["head pain", "head ache", "headac", "headach", "head is paining", "head paining", "pain in head", "migraine"],
        "sinus_pressure": ["sinus pain", "sinus", "face pressure", "facial pressure", "pressure near nose", "pressure near eyes"],
        "runny_nose": ["running nose", "watery nose", "water from nose", "nose running", "runny nose", "common cold", "cold"],
        "congestion": ["blocked nose", "block nose", "nose block", "stuffy nose", "nose blocked", "nose is blocked", "nasal blockage", "nasal congestion", "blocked nostril", "blocked nostrils", "stuffy nostril", "stuffy nostrils"],
        "throat_irritation": ["sore throat", "throat pain", "throat irritation", "itchy throat", "throat itching"],
        "redness_of_eyes": ["red eyes", "eye redness"],
        "watering_from_eyes": ["watery eyes", "water from eyes", "teary eyes"],
        "loss_of_smell": ["cannot smell", "can't smell", "loss smell", "smell loss", "reduced smell"],
        "phlegm": ["sputum", "mucus", "mucous"],
        "mucoid_sputum": ["thick mucus", "thick sputum"],
        "skin_rash": ["rash"],
        "fatigue": ["tiredness", "tired", "fatgue", "fatique", "fategue", "exhausted", "low energy", "drained", "worn out"],
        "burning_micturition": ["burning urination", "urine burning", "pain while urinating"],
        "loss_of_appetite": ["no appetite", "appetite loss"],
        "weight_loss": ["losing weight"],
        "weight_gain": ["gaining weight"],
        "acidity": ["acid reflux", "heartburn"],
        "yellowish_skin": ["yellow skin"],
        "yellowing_of_eyes": ["yellow eyes"],
        "cough": ["coughing", "dry cough", "khansi"],
        "itching": ["itchy", "itchiness"],
        "chills": ["chill"],
        "shivering": ["shivers"],
        "body_pain": ["body pain", "body ache", "full body pain", "pain all over", "general pain", "pain everywhere"],
        "sore_throat": ["sore throat", "throat sore", "throat hurts", "painful throat", "scratchy throat"],
        "dry_mouth": ["dry mouth", "mouth dry", "mouth dryness", "no saliva", "parched mouth", "thirsty mouth"],
        "ear_pain": ["ear pain", "earache", "ear ache", "pain in ear", "ear hurts", "ear infection"],
        "eye_pain": ["eye pain", "pain in eye", "eyes hurting", "eye ache", "pain behind eye"],
        "burning_sensation": ["burning sensation", "burning feeling", "feels like burning", "burning skin", "body burning"],
        "night_sweats": ["night sweats", "sweating at night", "sweats at night", "sweating during sleep", "waking up sweating"],
        "dry_cough": ["dry cough", "non productive cough", "cough without phlegm", "tickly cough"],
        "wheezing": ["wheezing", "wheeze", "wheezing sound", "whistling breath", "whistling breathing"],
        "loss_of_taste": ["loss of taste", "cannot taste", "no taste", "taste loss", "food tasteless"],
        "swollen_glands": ["swollen glands", "swollen lymph nodes", "glands swollen", "lumps in neck", "neck lumps"],
        "blood_pressure_fluctuation": ["blood pressure fluctuation", "bp fluctuation", "bp up down", "unstable bp", "blood pressure change", "bp high low"],
        "frequent_urination_at_night": ["frequent urination at night", "urinating at night", "waking to urinate", "nighttime urination", "peeing at night", "nocturia"],
        "tingling_sensation": ["tingling sensation", "tingling", "pins and needles", "numbness tingling", "prickling", "tingling hands", "tingling feet"],
        "sudden_weight_change": ["sudden weight change", "sudden weight loss", "sudden weight gain", "unexpected weight change", "rapid weight change"],
        "dizziness": ["dizzy", "giddy", "light headed", "lightheaded", "feeling faint"],
        "palpitations": ["heart beating fast", "fast heartbeat", "heart racing", "palpitation"],
        "blurred_and_distorted_vision": ["blurred vision", "blurry vision", "vision blur", "cannot see clearly"],
        "loss_of_appetite": ["no appetite", "appetite loss", "not hungry", "dont feel hungry"],
        "sweating": ["sweat", "sweating too much", "excess sweating"],
        "lethargy": ["lethargic", "very weak", "sleepy", "drowsy"],
    }

    global_patient_language_aliases = {
        "abdominal_pain": [
            "abdomen pain", "tummy pain", "tummy ache", "lower belly pain", "upper belly pain",
            "abdominal cramps", "cramps in stomach", "gastric pain", "intestinal pain",
        ],
        "abnormal_menstruation": [
            "irregular periods", "period problem", "heavy periods", "heavy bleeding during periods",
            "missed period", "periods not regular", "menstrual problem", "painful periods",
        ],
        "acidity": [
            "gastric acid", "acid in stomach", "sour belching", "sour burp", "burning chest after food",
            "acidic burps", "reflux", "gerd symptoms",
        ],
        "altered_sensorium": [
            "confused", "confusion", "not responding properly", "disoriented", "not alert",
            "acting confused", "mental confusion",
        ],
        "anxiety": [
            "panic", "panic feeling", "nervous", "restless mind", "worried feeling", "fearful",
            "anxious feeling",
        ],
        "back_pain": [
            "backache", "lower back pain", "upper back pain", "spine pain", "pain in back",
            "back is paining",
        ],
        "blackheads": ["black heads", "black spots on face", "blocked pores"],
        "bladder_discomfort": [
            "bladder pain", "pain near bladder", "urine pressure", "pressure in bladder",
            "lower urinary pain",
        ],
        "blister": ["blisters", "skin blister", "water blister", "fluid filled bumps"],
        "blood_in_sputum": [
            "blood in cough", "coughing blood", "blood with cough", "bloody sputum", "blood in mucus",
            "hemoptysis",
        ],
        "bloody_stool": [
            "blood in stool", "blood in motion", "blood in poop", "bloody motion", "red stool",
            "rectal bleeding",
        ],
        "blurred_and_distorted_vision": [
            "blurred eyesight", "blurry eyesight", "distorted vision", "double vision",
            "vision not clear", "fuzzy vision",
        ],
        "breathlessness": [
            "air hunger", "cannot breathe properly", "breath problem", "breath short", "breathless",
            "saans problem", "saans lene me dikkat", "sob",
        ],
        "brittle_nails": ["nails breaking", "weak nails", "fragile nails"],
        "bruising": ["easy bruising", "blue marks", "purple marks", "bruises"],
        "burning_micturition": [
            "burning pee", "burning while peeing", "burning urine", "painful urination",
            "urine pain", "burning when passing urine", "dysuria",
        ],
        "chest_pain": [
            "chest pressure", "chest heaviness", "tight chest", "left chest pain", "right chest pain",
            "pain around heart", "angina like pain",
        ],
        "chills": ["cold chills", "feeling chills", "chilly", "rigors"],
        "cold_hands_and_feets": [
            "cold hands", "cold feet", "hands and feet cold", "cold fingers", "cold toes",
        ],
        "coma": ["unconscious", "not waking up", "loss of consciousness"],
        "congestion": [
            "nose congestion", "blocked nasal passage", "nasal stuffiness", "sinus blockage",
            "nose jam", "naak band", "naak blocked",
        ],
        "constipation": [
            "constipated", "no stool", "cannot pass stool", "stool hard", "bowel not clear",
            "motion not coming",
        ],
        "continuous_feel_of_urine": [
            "always feel to pee", "constant urge to urinate", "urine urgency", "frequent urge to pee",
            "urine not empty",
        ],
        "continuous_sneezing": ["sneezing again and again", "sneeze repeatedly", "achoo", "chheenk"],
        "cough": [
            "wet cough", "productive cough", "cough with mucus", "khasi", "khaansi", "cough attack",
            "persistent cough",
        ],
        "cramps": ["muscle cramps", "leg cramps", "cramp", "spasm", "muscle spasm"],
        "dark_urine": ["dark pee", "tea colored urine", "brown urine", "cola colored urine"],
        "dehydration": [
            "dehydrated", "very thirsty", "dry tongue", "less urine", "not passing urine",
            "fluid loss",
        ],
        "depression": ["sadness", "low mood", "feeling hopeless", "depressed mood"],
        "diarrhoea": [
            "loose motions", "watery motion", "watery diarrhea", "frequent stools", "many stools",
            "pet kharab", "loose bowel",
        ],
        "dischromic _patches": [
            "skin discoloration", "discolored patches", "white patches on skin", "dark skin patches",
            "patches on skin",
        ],
        "distention_of_abdomen": [
            "bloated stomach", "abdominal bloating", "stomach swelling", "belly swelling",
            "abdomen swollen",
        ],
        "dizziness": [
            "light headedness", "lightheadedness", "faint feeling", "about to faint", "chakkar",
            "feeling dizzy",
        ],
        "dry_cough": ["dry khansi", "khansi without mucus", "nonproductive cough", "tickle cough"],
        "dry_mouth": ["mouth feels dry", "throat dry", "dry tongue", "excessive thirst"],
        "drying_and_tingling_lips": [
            "dry lips", "tingling lips", "lips tingling", "lips dry and tingling",
        ],
        "ear_pain": ["ear hurting", "ear pressure", "ear blocked with pain", "pain inside ear"],
        "enlarged_thyroid": ["neck swelling thyroid", "thyroid swelling", "goiter", "goitre"],
        "excessive_hunger": ["very hungry", "always hungry", "increased hunger", "polyphagia"],
        "eye_pain": ["eyes pain", "pain behind eyes", "eye hurts", "pressure in eyes"],
        "family_history": [
            "family has same disease", "runs in family", "family history", "parents had this",
        ],
        "fast_heart_rate": ["rapid pulse", "heart beat fast", "tachycardia", "pulse fast"],
        "fatigue": [
            "tired all the time", "extreme tiredness", "no energy", "feeling drained", "weakness all day",
        ],
        "foul_smell_of urine": [
            "smelly urine", "bad smell urine", "urine smells bad", "foul urine smell",
        ],
        "frequent_urination_at_night": [
            "night urination", "pee many times at night", "waking up to pee", "nocturnal urination",
        ],
        "headache": [
            "head pressure", "forehead pain", "temple pain", "pain behind head", "sir dard",
            "migraine headache",
        ],
        "high_fever": [
            "very high temperature", "temperature 102", "temperature 103", "temperature 104",
            "fever above 102", "burning fever",
        ],
        "hip_joint_pain": ["hip pain", "pain in hip", "hip joint ache"],
        "history_of_alcohol_consumption": [
            "alcohol history", "drinks alcohol", "heavy drinking", "alcohol use",
        ],
        "increased_appetite": ["more appetite", "eating more", "hungry more than usual"],
        "indigestion": ["digestive problem", "food not digesting", "upset stomach", "bad digestion"],
        "internal_itching": ["itching inside", "inside body itching"],
        "irregular_sugar_level": [
            "sugar level high", "sugar level low", "blood sugar fluctuating", "diabetes sugar up down",
            "glucose fluctuation",
        ],
        "irritability": ["irritable", "angry easily", "mood irritation"],
        "irritation_in_anus": ["anal itching", "anus irritation", "burning anus", "itching anus"],
        "itching": ["pruritus", "skin itching", "itching skin", "khujli"],
        "joint_pain": [
            "pain in joints", "joint pains", "arthralgia", "joints hurting", "joints paining",
        ],
        "knee_pain": ["pain in knee", "knee ache", "knee joint pain"],
        "lack_of_concentration": ["cannot focus", "poor concentration", "focus problem"],
        "lethargy": ["low energy", "sluggish", "no strength", "feeling lazy and weak"],
        "loss_of_appetite": [
            "reduced appetite", "not eating", "food aversion", "no hunger", "bhook nahi",
        ],
        "loss_of_balance": ["balance problem", "cannot balance", "unsteady balance"],
        "loss_of_smell": ["anosmia", "smell gone", "no smell", "lost smell"],
        "loss_of_taste": ["ageusia", "taste gone", "lost taste", "cannot taste food"],
        "malaise": ["general unwell", "feeling unwell", "not feeling well", "body discomfort"],
        "mild_fever": ["bukhar", "jwar", "fever low", "temperature 99", "temperature 100", "temperature 101"],
        "mood_swings": ["mood changes", "mood up down", "emotional changes"],
        "movement_stiffness": ["stiff movement", "hard to move", "body stiffness"],
        "mucoid_sputum": ["sticky mucus", "white sputum", "mucus cough"],
        "muscle_pain": ["myalgia", "muscles hurting", "muscle soreness", "body aches"],
        "muscle_wasting": ["muscle loss", "muscles shrinking", "loss of muscle"],
        "muscle_weakness": ["weak muscles", "muscles weak", "cannot lift properly"],
        "nausea": ["feeling like vomiting", "vomit feeling", "queasy", "motion sickness feeling"],
        "neck_pain": ["pain in neck", "neck ache", "neck is paining"],
        "night_sweats": ["sweating in sleep", "drenching night sweat", "night sweating"],
        "nodal_skin_eruptions": ["skin bumps", "raised skin bumps", "nodules on skin"],
        "obesity": ["overweight", "weight too much", "obese"],
        "pain_behind_the_eyes": ["retro orbital pain", "pain behind eyes", "eye socket pain"],
        "pain_during_bowel_movements": [
            "pain while passing stool", "pain during motion", "pain when pooping",
        ],
        "pain_in_anal_region": ["anal pain", "rectal pain", "pain near anus"],
        "painful_walking": ["pain while walking", "walking pain", "difficult to walk"],
        "palpitations": ["heart pounding", "irregular heartbeat", "fluttering heart"],
        "passage_of_gases": ["gas passing", "excess gas", "flatulence", "burping gas"],
        "patches_in_throat": ["white patches throat", "spots in throat", "throat patches"],
        "phlegm": ["cough phlegm", "chest mucus", "green mucus", "yellow mucus"],
        "polyuria": ["too much urine", "urinating often", "frequent urination", "passing urine often"],
        "prominent_veins_on_calf": ["visible calf veins", "bulging calf veins", "leg veins visible"],
        "puffy_face_and_eyes": ["puffy eyes", "face swelling", "swollen face", "eye swelling"],
        "pus_filled_pimples": ["pimple with pus", "pus pimples", "boils on face"],
        "red_sore_around_nose": ["redness around nose", "sore nose", "nose sore"],
        "red_spots_over_body": ["red spots", "spots over body", "red rash spots"],
        "redness_of_eyes": ["bloodshot eyes", "eyes red", "conjunctival redness"],
        "restlessness": ["cannot sit still", "restless", "agitated"],
        "runny_nose": ["nasal discharge", "nose water", "running nose water", "naak behna"],
        "rusty_sputum": ["rust colored sputum", "brown sputum", "rusty mucus"],
        "scurring": ["scarring", "skin scars", "acne scars"],
        "shivering": ["shaking chills", "body shaking", "trembling with cold"],
        "silver_like_dusting": ["silvery scales", "silver skin flakes", "psoriasis scales"],
        "sinus_pressure": ["sinus headache", "pressure in face", "cheek pressure"],
        "skin_peeling": ["peeling skin", "skin flakes", "flaky skin"],
        "skin_rash": ["rashes", "skin eruption", "red rash", "allergic rash"],
        "slurred_speech": ["speech slurring", "cannot speak clearly", "unclear speech"],
        "small_dents_in_nails": ["nail pits", "pitted nails", "small nail dents"],
        "sore_throat": ["throat hurts", "pain while swallowing", "swallowing pain"],
        "spinning_movements": ["room spinning", "vertigo", "spinning sensation"],
        "spotting_ urination": ["blood spotting urine", "urine spotting", "spotting during urination"],
        "stiff_neck": ["neck stiffness", "neck rigid", "cannot bend neck"],
        "stomach_bleeding": ["blood vomiting", "vomit blood", "black stool with stomach pain"],
        "stomach_pain": ["gas pain stomach", "pet dard", "stomach cramps", "tummy cramps"],
        "sunken_eyes": ["eyes sunken", "hollow eyes", "deep eyes"],
        "sweating": ["perspiration", "sweaty", "profuse sweating", "cold sweat"],
        "swelled_lymph_nodes": ["lymph node swelling", "swollen lymph node", "swollen nodes"],
        "swelling_joints": ["joint swelling", "swollen joints", "joints swollen"],
        "swelling_of_stomach": ["swollen stomach", "stomach swollen", "abdominal swelling"],
        "swollen_blood_vessels": ["swollen veins", "blood vessels swollen", "veins swollen"],
        "swollen_extremeties": ["swollen hands", "swollen feet", "limb swelling", "hand foot swelling"],
        "swollen_glands": ["neck glands swollen", "swollen neck nodes", "lymph glands swollen"],
        "swollen_legs": ["leg swelling", "feet swelling", "ankle swelling", "edema legs"],
        "throat_irritation": ["scratchy throat", "throat tickle", "throat discomfort"],
        "tingling_sensation": ["paresthesia", "numb tingling", "pins needles", "numbness"],
        "toxic_look_(typhos)": ["very ill looking", "toxic look", "looks very sick"],
        "ulcers_on_tongue": ["mouth ulcer", "tongue ulcer", "sores on tongue"],
        "unsteadiness": ["unsteady walking", "wobbly", "unstable walking"],
        "visual_disturbances": ["vision disturbance", "flashes in vision", "spots in vision"],
        "vomiting": ["vomit", "vomited", "emesis", "puke", "puked", "ulti"],
        "watering_from_eyes": ["eyes watering", "tears from eyes", "watery eye"],
        "weakness_in_limbs": ["weak arms", "weak legs", "limb weakness", "hands legs weak"],
        "weakness_of_one_body_side": ["one side weakness", "left side weakness", "right side weakness"],
        "weight_gain": ["putting on weight", "weight increased", "gained weight"],
        "weight_loss": ["lost weight", "weight decreased", "unintentional weight loss"],
        "wheezing": ["wheezy", "whistling chest", "breathing whistle"],
        "yellow_crust_ooze": ["yellow crust", "yellow oozing", "crusty yellow discharge"],
        "yellow_urine": ["yellow pee", "deep yellow urine", "bright yellow urine"],
        "yellowing_of_eyes": ["eyes yellow", "yellow sclera", "yellow white of eyes"],
        "yellowish_skin": ["skin yellow", "jaundice skin", "yellow body"],
    }

    for symptom, aliases in global_patient_language_aliases.items():
        if symptom in feature_names_tuple:
            existing_aliases = manual_aliases.setdefault(symptom, [])
            for alias in aliases:
                if alias not in existing_aliases:
                    existing_aliases.append(alias)

    alias_map = {}
    for symptom in feature_names_tuple:
        aliases = {
            normalize_text_for_matching(symptom),
            normalize_text_for_matching(format_symptom_label(symptom)),
        }
        for alias in manual_aliases.get(symptom, []):
            aliases.add(normalize_text_for_matching(alias))

        alias_map[symptom] = sorted(alias for alias in aliases if alias)

    return alias_map


def normalized_phrase_present(alias, normalized_text):
    alias = normalize_text_for_matching(alias)
    padded_text = f" {normalized_text} "
    if not alias:
        return False
    if f" {alias} " in padded_text:
        return True
    if " " not in alias and len(alias) >= 5:
        return bool(re.search(rf"\b{re.escape(alias)}(?:s|es|ing|ed)?\b", normalized_text))
    return False


def detect_symptoms_from_text(text, feature_names):
    normalized_text = normalize_text_for_matching(text)
    alias_map = build_symptom_alias_map(tuple(feature_names))
    matched = []

    for symptom in feature_names:
        for alias in alias_map.get(symptom, []):
            if normalized_phrase_present(alias, normalized_text):
                matched.append(symptom)
                break

    negated_symptoms = set(detect_negated_symptoms_from_text(text, feature_names))
    return sorted(symptom for symptom in set(matched) if symptom not in negated_symptoms)


def detect_negated_symptoms_from_text(text, feature_names):
    normalized_text = f" {normalize_text_for_matching(text)} "
    alias_map = build_symptom_alias_map(tuple(feature_names))
    negated = []
    negation_templates = (
        "no {alias}",
        "not {alias}",
        "without {alias}",
        "dont have {alias}",
        "don t have {alias}",
        "do not have {alias}",
        "not having {alias}",
        "not feeling {alias}",
        "no to {alias}",
        "without any {alias}",
        "no symptom of {alias}",
        "{alias} absent",
        "{alias} not present",
    )

    def alias_in_negated_clause(alias):
        alias_match = re.search(rf"\b{re.escape(alias)}\b", normalized_text)
        if not alias_match:
            return False
        prefix = normalized_text[max(0, alias_match.start() - 90) : alias_match.start()]
        for separator in (" but ", " however ", " except "):
            if separator in prefix:
                prefix = prefix.split(separator)[-1]
        return bool(
            re.search(
                r"\b(?:no|not|without|dont have|don t have|do not have|not having)\b",
                prefix,
            )
        )

    for symptom in feature_names:
        for alias in alias_map.get(symptom, []):
            for template in negation_templates:
                if f" {template.format(alias=alias)} " in normalized_text:
                    negated.append(symptom)
                    break
                if alias_in_negated_clause(alias):
                    negated.append(symptom)
                    break
            if symptom in negated:
                break

    generic_fever_denied = re.search(
        r"\b(?:no|without|dont have|don t have|do not have|not having)\s+(?:any\s+)?fever\b",
        normalized_text,
    ) or re.search(r"\bfever\s+(?:no|absent|not present)\b", normalized_text)
    if generic_fever_denied:
        for fever_symptom in ("mild_fever", "high_fever"):
            if fever_symptom in feature_names:
                negated.append(fever_symptom)

    return sorted(set(negated))


SYMPTOM_EDUCATION_GUIDE = {
    "fatigue": "unusual tiredness, weariness, or low energy that can make normal activity feel harder than usual.",
    "headache": "pain or discomfort in the head, scalp, face, or neck area.",
    "muscle_pain": "aching, soreness, or pain in the muscles; many people describe it as body ache.",
    "body_pain": "general aching or pain across the body, often felt in muscles or joints.",
    "mild_fever": "a raised body temperature or feverish feeling that may come with chills, weakness, or body ache.",
    "high_fever": "a stronger fever or high temperature that needs closer attention, especially if it is persistent or severe.",
    "chills": "feeling cold or shivery, sometimes with shaking, even when the room is not cold.",
    "shivering": "involuntary shaking or trembling, often linked with fever, chills, anxiety, or cold exposure.",
    "sweating": "more sweat than usual, which may happen with fever, heat, anxiety, pain, or low blood sugar.",
    "cough": "air being forced out from the lungs to clear irritation, mucus, infection, or other triggers.",
    "dry_cough": "a cough without much mucus or phlegm.",
    "phlegm": "mucus from the chest or throat, often noticed while coughing.",
    "runny_nose": "watery or mucus-like fluid coming from the nose.",
    "congestion": "a blocked or stuffy nose that can make breathing through the nose harder.",
    "sore_throat": "pain, scratchiness, or irritation in the throat, often worse while swallowing.",
    "throat_irritation": "scratchy, itchy, or uncomfortable feeling in the throat.",
    "nausea": "feeling like you may vomit, even if vomiting does not happen.",
    "vomiting": "throwing up stomach contents through the mouth.",
    "diarrhoea": "loose or watery stools, usually happening more often than normal.",
    "abdominal_pain": "pain or discomfort in the belly or abdomen.",
    "stomach_pain": "pain, cramps, or discomfort around the stomach or belly area.",
    "breathlessness": "feeling short of breath or having difficulty breathing.",
    "chest_pain": "pain, pressure, tightness, or discomfort in the chest area.",
    "dizziness": "feeling light-headed, unsteady, faint, or like the surroundings may spin.",
    "skin_rash": "a visible change on the skin, such as redness, spots, bumps, or irritation.",
    "itching": "an uncomfortable skin feeling that makes you want to scratch.",
    "dehydration": "not having enough fluid in the body, often linked with thirst, dry mouth, dizziness, or less urination.",
    "loss_of_appetite": "not feeling hungry or eating less than usual.",
    "joint_pain": "pain, aching, or stiffness around one or more joints.",
}

SYMPTOM_INFORMATION_WORD_FIXES = {
    "fatgue": "fatigue",
    "fatique": "fatigue",
    "fategue": "fatigue",
    "headac": "headache",
    "headach": "headache",
    "head ache": "headache",
    "muscl pain": "muscle pain",
}


def normalize_symptom_information_query(text):
    normalized = normalize_text_for_matching(text)
    if not normalized:
        return ""

    spacing_patterns = (
        (r"\b(what|wht|wat|whats)\s+is([a-z][a-z0-9]{3,})\b", r"\1 is \2"),
        (r"\b(what|wht|wat|whats)\s+are([a-z][a-z0-9]{3,})\b", r"\1 are \2"),
        (r"\bmeaning\s+of([a-z][a-z0-9]{3,})\b", r"meaning of \1"),
        (r"\bexplain([a-z][a-z0-9]{3,})\b", r"explain \1"),
    )
    for pattern, replacement in spacing_patterns:
        normalized = re.sub(pattern, replacement, normalized)

    for typo, canonical in SYMPTOM_INFORMATION_WORD_FIXES.items():
        normalized = re.sub(rf"\b{re.escape(typo)}\b", canonical, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def has_symptom_information_intent(normalized_text):
    if not normalized_text:
        return False
    education_patterns = (
        r"^(?:what|wht|wat|whats)\s+(?:is|are)\s+",
        r"^(?:what|wht|wat|whats)\s+does\s+.+\s+mean\b",
        r"\bmeaning\s+of\s+",
        r"\b.+\s+meaning\b",
        r"\b.+\s+means\b",
        r"^(?:define|explain)\s+",
        r"^(?:tell me|can you tell me|can you explain|please explain)\s+(?:about\s+)?",
        r"^(?:give me|show me)\s+(?:information|info)\s+(?:about|on)\s+",
    )
    return any(re.search(pattern, normalized_text) for pattern in education_patterns)


def detect_symptom_information_request(text, feature_names):
    normalized_text = normalize_symptom_information_query(text)
    if not has_symptom_information_intent(normalized_text):
        return []

    alias_map = build_symptom_alias_map(tuple(feature_names))
    matched = []
    for symptom in feature_names:
        for alias in alias_map.get(symptom, []):
            if normalized_phrase_present(alias, normalized_text):
                matched.append(symptom)
                break
    return sorted(set(matched))


def build_symptom_information_reply(symptoms):
    display_symptoms = list(symptoms or [])[:4]
    lines = ["Quick answer:"]

    for symptom in display_symptoms:
        label = format_symptom_label(symptom)
        explanation = SYMPTOM_EDUCATION_GUIDE.get(
            symptom,
            f"a symptom where the patient notices {label.lower()}. It can have many possible causes, so context matters.",
        )
        lines.append(f"- {label}: {explanation}")

    if len(symptoms or []) > len(display_symptoms):
        lines.append("- I can explain the remaining symptoms one by one if you want.")

    lines.append("This explains the word only; it does not confirm any one disease.")
    if st.session_state.get("assistant_pending_question"):
        lines.append("When ready, answer my previous question so I can continue the screening.")
    else:
        lines.append("Are you asking generally, or are you having this symptom right now?")
    return "\n".join(lines)


def extract_duration_detail(message_text):
    text = normalize_text_for_matching(message_text)
    duration_match = re.search(
        r"\b(?:for|from|since)?\s*(\d+\s*(?:hour|hours|day|days|week|weeks|month|months))\b",
        text,
    )
    if duration_match:
        return duration_match.group(1).strip()

    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    word_duration_match = re.search(
        r"\b(?:for|from|since)?\s*(one|two|three|four|five|six|seven|eight|nine|ten)\s*(hour|hours|day|days|week|weeks|month|months)\b",
        text,
    )
    if word_duration_match:
        return f"{number_words[word_duration_match.group(1)]} {word_duration_match.group(2)}"

    relative_patterns = (
        ("last few days", "a few days"),
        ("few days ago", "a few days"),
        ("few days", "a few days"),
        ("couple of days", "a couple of days"),
        ("couple days", "a couple of days"),
        ("several days", "several days"),
        ("since yesterday", "since yesterday"),
        ("from yesterday", "since yesterday"),
        ("yesterday", "since yesterday"),
        ("today", "today"),
        ("this morning", "since this morning"),
        ("morning", "since this morning"),
        ("last night", "since last night"),
        ("night", "since last night"),
    )
    for phrase, label in relative_patterns:
        if phrase in text:
            return label
    return ""


def extract_temperature_detail(message_text):
    text = normalize_text_for_matching(message_text)
    temp_match = re.search(r"\b(9[8-9]|10[0-6])(?:\s*(?:f|degree|degrees))?\b", text)
    if temp_match:
        return f"{temp_match.group(1)} F"
    return ""


def extract_severity_detail(message_text):
    text = normalize_text_for_matching(message_text)
    if any(term in text for term in ("very severe", "severe", "unbearable", "too much", "extreme")):
        return "severe"
    if any(term in text for term in ("moderate", "medium")):
        return "moderate"
    if any(term in text for term in ("mild", "slight", "little", "light")):
        return "mild"
    if any(term in text for term in ("high fever", "high temperature")):
        return "high"
    return ""


def extract_medicine_detail(message_text):
    text = normalize_text_for_matching(message_text)
    if any(term in text for term in ("no medicine", "no medicines", "not taken medicine", "did not take medicine", "havent taken medicine", "haven t taken medicine", "no tablet", "nothing taken")):
        return "No medicine taken yet"

    medicine_terms = (
        "medicine",
        "medicines",
        "tablet",
        "tablets",
        "drug",
        "drugs",
        "dose",
        "dosage",
        "paracetamol",
        "dolo",
        "crocin",
        "ibuprofen",
        "aspirin",
        "cetirizine",
        "levocetirizine",
        "azithromycin",
        "amoxicillin",
        "antibiotic",
        "steroid",
        "nasal spray",
        "syrup",
        "cough syrup",
        "antacid",
        "pantoprazole",
        "omeprazole",
        "ors",
        "inhaler",
        "salbutamol",
        "insulin",
        "bp medicine",
        "blood pressure medicine",
        "diabetes medicine",
    )
    if any(term in text for term in medicine_terms):
        return str(message_text).strip()
    return ""


def extract_progression_detail(message_text):
    text = normalize_text_for_matching(message_text)
    if any(term in text for term in ("getting worse", "worsening", "increasing", "more severe", "not improving", "worse")):
        return "worsening"
    if any(term in text for term in ("getting better", "improving", "reduced", "less now", "better")):
        return "improving"
    if any(term in text for term in ("same", "no change", "stable")):
        return "same"
    return ""


def extract_risk_context_detail(message_text):
    text = normalize_text_for_matching(message_text)
    risk_items = []
    risk_patterns = {
        "asthma": ("asthma", "inhaler"),
        "diabetes": ("diabetes", "sugar patient", "high sugar"),
        "pregnancy": ("pregnant", "pregnancy"),
        "heart condition": ("heart patient", "heart disease", "cardiac"),
        "high blood pressure": ("bp patient", "blood pressure", "hypertension"),
        "low immunity": ("low immunity", "cancer treatment", "chemotherapy", "steroid"),
    }
    for label, terms in risk_patterns.items():
        if any(term in text for term in terms):
            risk_items.append(label)
    return sorted(set(risk_items))


def extract_clinical_note_update(message_text, feature_names):
    denied_symptoms = detect_negated_symptoms_from_text(message_text, feature_names)
    update = {}
    duration = extract_duration_detail(message_text)
    temperature = extract_temperature_detail(message_text)
    severity = extract_severity_detail(message_text)
    medicine = extract_medicine_detail(message_text)
    progression = extract_progression_detail(message_text)
    risk_context = extract_risk_context_detail(message_text)
    if duration:
        update["duration"] = duration
    if temperature:
        update["temperature"] = temperature
    if severity:
        update["severity"] = severity
    if medicine:
        update["medicine"] = medicine
    if progression:
        update["progression"] = progression
    if risk_context:
        update["risk_context"] = risk_context
    if denied_symptoms:
        update["denied_symptoms"] = denied_symptoms
    return update


def update_assistant_clinical_notes(note_update):
    notes = dict(st.session_state.get("assistant_clinical_notes", {}) or {})
    for key, value in note_update.items():
        if key == "denied_symptoms":
            existing = set(notes.get("denied_symptoms", []))
            existing.update(value)
            notes["denied_symptoms"] = sorted(existing)
        elif key == "risk_context":
            existing = set(notes.get("risk_context", []))
            existing.update(value)
            notes["risk_context"] = sorted(existing)
        else:
            notes[key] = value
    st.session_state.assistant_clinical_notes = notes
    return notes


def has_clinical_detail(note_update):
    return any(
        note_update.get(key)
        for key in (
            "duration",
            "temperature",
            "severity",
            "medicine",
            "progression",
            "risk_context",
            "exposure_context",
            "differential_context",
            "other_symptoms",
            "denied_symptoms",
        )
    )


def get_symptom_question_profile(symptoms):
    symptom_set = set(symptoms)
    if symptom_set & {"congestion", "runny_nose", "sinus_pressure", "continuous_sneezing", "throat_irritation", "loss_of_smell", "redness_of_eyes", "watering_from_eyes"}:
        return {
            "name": "nose, throat, and cold symptoms",
            "related": ["mild_fever", "high_fever", "cough", "throat_irritation", "continuous_sneezing", "headache", "sinus_pressure", "runny_nose", "loss_of_smell", "breathlessness"],
            "questions": [
                "From how many days are you having the blocked or runny nose?",
                "Do you also have fever, cough, throat pain, sneezing, headache, body pain, or shivering?",
                "Is the nose discharge watery/clear or thick/yellow/green? Do you feel sinus or face pressure?",
                "Have you taken any medicine such as paracetamol, cetirizine, a nasal spray, or antibiotics?",
                "Any breathing difficulty, chest pain, wheezing, asthma history, or severe weakness?",
            ],
        }
    if symptom_set & {"high_fever", "mild_fever", "chills", "shivering", "sweating"}:
        return {
            "name": "fever pattern",
            "related": ["chills", "shivering", "sweating", "headache", "muscle_pain", "cough", "runny_nose", "vomiting", "diarrhoea", "red_spots_over_body", "dehydration"],
            "questions": [
                "From how many days do you have fever, and what was the highest temperature if you checked it?",
                "Do you have chills, shivering, sweating, headache, body pain, cough, vomiting, loose motions, or rash?",
                "Have you taken paracetamol or any other medicine? Did the fever reduce after medicine?",
                "Any warning signs like breathlessness, chest pain, confusion, severe weakness, dehydration, or persistent vomiting?",
            ],
        }
    if symptom_set & {"cough", "phlegm", "mucoid_sputum", "rusty_sputum", "breathlessness", "chest_pain"}:
        return {
            "name": "cough and breathing symptoms",
            "related": ["high_fever", "mild_fever", "phlegm", "breathlessness", "chest_pain", "throat_irritation", "runny_nose", "fatigue"],
            "questions": [
                "From how many days do you have cough or breathing symptoms?",
                "Is the cough dry or with phlegm? If phlegm is present, what color is it?",
                "Do you also have fever, chest pain, throat irritation, runny nose, fatigue, or breathlessness?",
                "Have you taken cough syrup, antibiotics, inhaler, steam, or any other medicine?",
            ],
        }
    if symptom_set & {"vomiting", "nausea", "diarrhoea", "stomach_pain", "abdominal_pain", "belly_pain", "indigestion", "acidity", "constipation"}:
        return {
            "name": "stomach and digestion symptoms",
            "related": ["vomiting", "nausea", "diarrhoea", "abdominal_pain", "stomach_pain", "high_fever", "dehydration", "loss_of_appetite"],
            "questions": [
                "From how many days are you having stomach or digestion symptoms?",
                "Do you have vomiting, loose motions, fever, dehydration, loss of appetite, or severe abdominal pain?",
                "Did symptoms start after outside food, spicy food, alcohol, or any new medicine?",
                "Have you taken ORS, antacid, antibiotics, painkiller, or any other medicine?",
            ],
        }
    if symptom_set & {"skin_rash", "itching", "red_spots_over_body", "pus_filled_pimples", "blister", "yellow_crust_ooze"}:
        return {
            "name": "skin symptoms",
            "related": ["itching", "skin_rash", "red_spots_over_body", "high_fever", "blister", "pus_filled_pimples", "yellow_crust_ooze"],
            "questions": [
                "From how many days do you have the skin symptom?",
                "Is there itching, rash spreading, blisters, pus, fever, or pain?",
                "Did it start after a new medicine, food, soap, cream, or allergy exposure?",
                "Have you applied any cream or taken any allergy medicine?",
            ],
        }
    return {
        "name": "current symptoms",
        "related": ["high_fever", "mild_fever", "headache", "fatigue", "muscle_pain", "vomiting", "cough", "breathlessness"],
        "questions": [
            "From how many days are you feeling this?",
            "How severe is it: mild, moderate, or severe?",
            "Do you have any other symptoms like fever, headache, cough, vomiting, body pain, rash, or breathing difficulty?",
            "Have you taken any medicine already? If yes, which one and did it help?",
        ],
    }


def build_clinical_notes_summary(notes):
    parts = []
    if notes.get("duration"):
        parts.append(f"duration: {notes['duration']}")
    if notes.get("temperature"):
        parts.append(f"temperature: {notes['temperature']}")
    if notes.get("severity"):
        parts.append(f"severity: {notes['severity']}")
    if notes.get("progression"):
        parts.append(f"trend: {notes['progression']}")
    if notes.get("medicine"):
        parts.append(f"medicine: {notes['medicine']}")
    if notes.get("risk_context"):
        parts.append(f"risk context: {format_natural_list(notes['risk_context'], 3)}")
    if notes.get("exposure_context"):
        parts.append(f"exposure/context: {notes['exposure_context']}")
    if notes.get("differential_context"):
        parts.append(f"extra clues: {notes['differential_context']}")
    if notes.get("other_symptoms"):
        parts.append(f"other symptoms: {notes['other_symptoms']}")
    if notes.get("denied_symptoms"):
        parts.append(f"not present: {format_symptom_summary(notes['denied_symptoms'])}")
    return "; ".join(parts) if parts else "no duration, severity, or medicine details yet"


def medicine_history_is_empty(medicine_text):
    normalized = normalize_text_for_matching(medicine_text)
    return any(
        phrase in normalized
        for phrase in (
            "no medicine",
            "no medicines",
            "no tablet",
            "not taken",
            "did not take",
            "nothing taken",
            "no medicine taken",
        )
    )


def medicine_text_has_any(medicine_text, terms):
    normalized = normalize_text_for_matching(medicine_text)
    return any(term in normalized for term in terms)


def build_medicine_safety_guidance(notes, cumulative_symptoms=None, care_plan=None):
    medicine_text = str((notes or {}).get("medicine", "")).strip()
    symptoms = set(cumulative_symptoms or [])
    if not medicine_text:
        return ""

    lines = ["Medicine safety:"]
    if medicine_history_is_empty(medicine_text):
        lines.append("- Since you have not taken medicine yet, do not self-start antibiotics, steroids, or leftover prescription tablets.")
        lines.append("- Basic care like rest, fluids, ORS for loose motions, or steam/saline for cold symptoms can be considered if suitable.")
        return "\n".join(lines)

    lines.append(f"- I noted: {medicine_text}.")

    if medicine_text_has_any(medicine_text, ("antibiotic", "azithromycin", "amoxicillin", "cefixime", "ciprofloxacin")):
        lines.append("- If this antibiotic was prescribed, do not stop or change it without the prescribing doctor. If it was not prescribed, do not continue self-medicating.")

    if medicine_text_has_any(medicine_text, ("paracetamol", "dolo", "crocin", "acetaminophen")):
        lines.append("- For paracetamol/Dolo/Crocin, avoid taking multiple products with the same ingredient and be extra careful with liver disease or alcohol use.")

    if medicine_text_has_any(medicine_text, ("ibuprofen", "aspirin", "diclofenac", "nsaid", "painkiller")):
        lines.append("- Painkillers like ibuprofen/aspirin can be risky with acidity/ulcer, kidney disease, dehydration, blood thinners, pregnancy, or dengue-like fever.")

    if medicine_text_has_any(medicine_text, ("cetirizine", "levocetirizine", "antihistamine", "allergy")):
        lines.append("- Allergy tablets can cause sleepiness, so avoid driving and avoid mixing with alcohol or sedating medicines.")

    if medicine_text_has_any(medicine_text, ("nasal spray", "decongestant spray", "xylometazoline", "oxymetazoline")):
        lines.append("- Medicated nasal sprays should not be overused; saline spray is usually safer for simple blockage.")

    if medicine_text_has_any(medicine_text, ("cough syrup", "syrup")):
        lines.append("- Cough syrups can cause drowsiness and may not be safe for every cough, especially with breathing trouble.")

    if medicine_text_has_any(medicine_text, ("inhaler", "salbutamol")):
        lines.append("- If an inhaler was prescribed for asthma/wheezing, use it only as directed; urgent care is needed if breathing difficulty is not relieved.")

    if medicine_text_has_any(medicine_text, ("steroid", "prednisolone", "dexamethasone", "steroid cream")):
        lines.append("- Do not start or suddenly stop steroid tablets/creams without medical advice.")

    chronic_terms = (
        "bp medicine",
        "blood pressure",
        "hypertension",
        "diabetes medicine",
        "insulin",
        "metformin",
        "thyroid",
        "heart medicine",
        "seizure",
    )
    if medicine_text_has_any(medicine_text, chronic_terms):
        lines.append("- Continue regular chronic medicines unless a doctor tells you to change them; sudden stopping can be unsafe.")

    allergy_symptoms = symptoms & {"skin_rash", "itching", "red_spots_over_body", "blister", "yellow_crust_ooze", "breathlessness"}
    if allergy_symptoms or medicine_text_has_any(medicine_text, ("rash", "swelling", "allergy", "breathing", "wheezing", "faint", "skin peeling")):
        lines.append("- If symptoms started after a new medicine, or there is lip/face swelling, wheezing, fainting, severe rash, or skin peeling, seek urgent care and do not take another dose until a clinician advises.")

    if care_plan:
        medicine_note = format_assistant_items(care_plan.get("medicines", []), 2)
        if medicine_note:
            lines.append(f"- For this symptom pattern, discuss this with a doctor/pharmacist: {medicine_note}.")

    if len(lines) == 2:
        lines.append("- Do not change prescribed medicines by yourself; tell a doctor/pharmacist what you took and whether it helped or caused side effects.")

    return "\n".join(lines[:6])


def build_medicine_advice_reply(detected_now, cumulative_symptoms, preview, care_plan, notes, urgent_note, allow_predict=True):
    guidance = build_medicine_safety_guidance(notes, cumulative_symptoms, care_plan)
    lines = []
    if detected_now:
        lines.append(f"I noted {format_related_symptoms_for_question(detected_now, max_items=3)}.")
    elif cumulative_symptoms:
        lines.append(f"I am considering your saved symptoms: {format_symptom_summary(cumulative_symptoms)}.")
    else:
        lines.append("I can help with medicine safety, but I need symptoms too.")

    if guidance:
        lines.append(guidance)
    else:
        lines.append(
            "Medicine safety:\n"
            "- I cannot prescribe a new medicine here or change a prescription.\n"
            "- Tell me what medicine you already took, whether it was prescribed, and whether it helped or caused side effects."
        )

    if urgent_note:
        lines.append(urgent_note)

    next_question = get_next_patient_question(cumulative_symptoms, notes, allow_predict=allow_predict, preview=preview)
    lines.append(next_question)
    return "\n\n".join(lines)


def get_symptom_reasoning_line(symptoms, notes, preview=None, care_plan=None):
    symptoms = list(symptoms or [])
    notes = notes or {}
    if not symptoms:
        return "I need one clear symptom first so I can guide the screening properly."

    profile = get_symptom_question_profile(symptoms)
    profile_name = profile.get("name", "symptoms")
    pattern_label = profile_name if "pattern" in profile_name or "symptoms" in profile_name else f"{profile_name} pattern"
    symptom_text = format_related_symptoms_for_question(symptoms, max_items=4)
    detail_bits = []
    if notes.get("duration"):
        duration_text = str(notes["duration"])
        if duration_text.startswith(("since", "from", "today")):
            detail_bits.append(duration_text)
        else:
            detail_bits.append(f"for {duration_text}")
    if notes.get("severity"):
        detail_bits.append(f"{notes['severity']} severity")
    if notes.get("progression"):
        detail_bits.append(f"{notes['progression']} trend")
    if notes.get("risk_context"):
        detail_bits.append(f"with {format_natural_list(notes['risk_context'], 2)} noted")

    detail_text = f" ({', '.join(detail_bits)})" if detail_bits else ""
    if care_plan:
        return (
            f"This looks like a {pattern_label} from {symptom_text}{detail_text}; "
            f"the safest next step is {care_plan['specialist']} guidance if it continues or worsens."
        )
    if preview:
        return (
            f"This looks like a {pattern_label} from {symptom_text}{detail_text}; "
            "I am checking duration, severity, trend, and risk details before the final screening."
        )
    return f"This looks like a {pattern_label} from {symptom_text}{detail_text}."


def format_related_symptoms_for_question(symptoms, max_items=7):
    friendly_labels = {
        "breathlessness": "breathing difficulty",
        "congestion": "blocked nose",
        "continuous_sneezing": "sneezing",
        "high_fever": "high fever",
        "mild_fever": "fever",
        "throat_irritation": "throat pain",
        "sinus_pressure": "sinus pressure",
        "runny_nose": "runny nose",
        "loss_of_smell": "loss of smell",
        "redness_of_eyes": "red eyes",
        "watering_from_eyes": "watery eyes",
        "muscle_pain": "body pain",
        "diarrhoea": "loose motions",
        "red_spots_over_body": "rash or red spots",
    }
    labels = []
    fever_added = False
    for symptom in symptoms:
        if symptom in {"high_fever", "mild_fever"}:
            if not fever_added:
                labels.append("fever")
                fever_added = True
            continue
        labels.append(friendly_labels.get(symptom, format_symptom_label(symptom).lower()))

    labels = [label for idx, label in enumerate(labels) if label and label not in labels[:idx]]
    visible_labels = labels[:max_items]
    if not visible_labels:
        return "any other symptom"
    if len(visible_labels) == 1:
        return visible_labels[0]
    if len(visible_labels) == 2:
        return " or ".join(visible_labels)
    return ", ".join(visible_labels[:-1]) + f", or {visible_labels[-1]}"


def parse_simple_yes_no_answer(message_text):
    text = normalize_text_for_matching(message_text)
    yes_terms = {"yes", "yes i do", "yes i have", "yeah", "yep", "yup", "haan", "han", "ha"}
    no_terms = {"no", "no i dont", "no i do not", "nope", "nah", "not really", "nahi", "na"}
    if text in yes_terms:
        return "yes"
    if text in no_terms:
        return "no"
    if re.fullmatch(r"yes[,\s]+.*", text):
        return "yes"
    if re.fullmatch(r"no[,\s]+.*", text):
        return "no"
    return ""


def merge_note_updates(*updates):
    merged = {}
    denied = set()
    risk_context = set()
    for update in updates:
        for key, value in (update or {}).items():
            if key == "denied_symptoms":
                denied.update(value)
            elif key == "risk_context":
                risk_context.update(value)
            elif value:
                merged[key] = value
    if denied:
        merged["denied_symptoms"] = sorted(denied)
    if risk_context:
        merged["risk_context"] = sorted(risk_context)
    return merged


def resolve_pending_question_answer(message_text, feature_names):
    pending = dict(st.session_state.get("assistant_pending_question", {}) or {})
    st.session_state.assistant_pending_question = {}
    result = {"detected_symptoms": [], "note_update": {}, "action": None, "error": None}
    if not pending:
        return result

    question_type = pending.get("type", "")
    target_symptoms = list(pending.get("target_symptoms", []) or [])
    positive_symptoms = list(pending.get("positive_symptoms", []) or target_symptoms)
    note_key = pending.get("note_key")
    positive_value = pending.get("positive_value")
    negative_value = pending.get("negative_value")
    explicit_detected = set(detect_symptoms_from_text(message_text, feature_names))
    explicit_denied = set(detect_negated_symptoms_from_text(message_text, feature_names))
    simple_answer = parse_simple_yes_no_answer(message_text)

    is_valid = False

    if question_type == "symptom":
        if explicit_detected & set(target_symptoms):
            result["detected_symptoms"] = sorted(explicit_detected & set(target_symptoms))
            is_valid = True
        elif explicit_denied & set(target_symptoms):
            result["note_update"] = {"denied_symptoms": sorted(explicit_denied & set(target_symptoms))}
            is_valid = True
        elif simple_answer == "yes":
            result["detected_symptoms"] = [symptom for symptom in positive_symptoms if symptom in feature_names]
            is_valid = True
        elif simple_answer == "no":
            result["note_update"] = {"denied_symptoms": [symptom for symptom in target_symptoms if symptom in feature_names]}
            is_valid = True
        else:
            result["error"] = "Please answer 'yes' or 'no', or describe your symptom clearly."

    elif question_type == "duration":
        duration_text = extract_duration_detail(message_text)
        if duration_text:
            result["note_update"] = {"duration": duration_text}
            is_valid = True
        else:
            numbers = re.findall(r"\d+", message_text)
            if numbers:
                result["note_update"] = {"duration": f"{numbers[0]} days"}
                is_valid = True
            else:
                result["error"] = "Please specify the duration as a number, for example: '3 days' or '1 week'."

    elif question_type == "medicine":
        medicine = extract_medicine_detail(message_text)
        if medicine:
            result["note_update"] = {"medicine": medicine}
            is_valid = True
        elif simple_answer == "no":
            result["note_update"] = {"medicine": "No medicine taken yet"}
            is_valid = True
        elif simple_answer == "yes":
            result["note_update"] = {"medicine": "Medicine taken, name not specified"}
            is_valid = True
        else:
            result["error"] = "Please tell me the name of the medicine, or say 'no' if you haven't taken any."

    elif question_type == "severity":
        severity = extract_severity_detail(message_text)
        if severity:
            result["note_update"] = {"severity": severity}
            is_valid = True
        else:
            result["error"] = "Please describe the severity (e.g. mild, moderate, severe)."

    elif question_type == "temperature":
        temperature = extract_temperature_detail(message_text)
        if temperature:
            result["note_update"] = {"temperature": temperature}
            is_valid = True
        else:
            result["error"] = "Please provide your temperature (e.g. 101, high, normal)."

    elif question_type == "progression":
        progression = extract_progression_detail(message_text)
        if progression:
            result["note_update"] = {"progression": progression}
            is_valid = True
        elif simple_answer == "no":
            result["note_update"] = {"progression": "same"}
            is_valid = True
        else:
            result["error"] = "Please say whether it is getting better, getting worse, or staying the same."

    elif question_type == "risk_context":
        risk_context = extract_risk_context_detail(message_text)
        if risk_context:
            result["note_update"] = {"risk_context": risk_context}
            is_valid = True
        elif simple_answer == "no":
            result["note_update"] = {"risk_context": ["none mentioned"]}
            is_valid = True
        elif simple_answer == "yes":
            result["error"] = "Please tell me which one: asthma, diabetes, pregnancy, heart disease, high BP, or low immunity."
        else:
            result["error"] = "Please answer with the condition name, or say 'no' if none of these apply."

    elif question_type == "context_yes_no":
        if simple_answer == "yes":
            if note_key and positive_value:
                result["note_update"] = {note_key: positive_value}
            is_valid = True
        elif simple_answer == "no":
            if note_key and negative_value:
                result["note_update"] = {note_key: negative_value}
            is_valid = True
        elif len(message_text.strip()) > 2:
            if note_key:
                result["note_update"] = {note_key: message_text.strip()}
            is_valid = True
        else:
            result["error"] = "Please answer 'yes' or 'no', or describe it briefly."

    elif question_type == "predict_confirmation":
        if simple_answer == "yes":
            result["action"] = "predict"
            is_valid = True
        elif simple_answer == "no":
            result["note_update"] = {}
            is_valid = True
        else:
            result["error"] = "Please answer 'yes' or 'no'."

    elif question_type == "open":
        pending_key = pending.get("key", "")
        if pending_key == "other_symptom" and simple_answer == "no":
            result["note_update"] = {"other_symptoms": "none mentioned"}
            is_valid = True
        elif pending_key == "other_symptom" and explicit_detected:
            result["detected_symptoms"] = sorted(explicit_detected)
            result["note_update"] = {"other_symptoms": "additional symptoms mentioned"}
            is_valid = True
        elif pending_key == "other_symptom" and len(message_text.strip()) > 2 and simple_answer not in ("yes", "no"):
            result["note_update"] = {"other_symptoms": message_text.strip()}
            is_valid = True
        elif pending_key == "other_symptom" and simple_answer == "yes":
            result["error"] = "Please tell me the other symptom, for example: cough, vomiting, rash, or body pain."
        elif len(message_text.strip()) > 2 and simple_answer not in ("yes", "no"):
            is_valid = True
        elif simple_answer == "no":
            is_valid = True
        else:
            result["error"] = "Please describe the symptom, or say 'no' if you don't have any others."

    if not is_valid:
        context_update = extract_clinical_note_update(message_text, feature_names)
        if has_clinical_detail(context_update):
            result["note_update"] = merge_note_updates(result.get("note_update", {}), context_update)
            result["error"] = None
            is_valid = True

    if not is_valid:
        st.session_state.assistant_pending_question = pending

    return result


def remember_assistant_question(
    question_key,
    question_type,
    target_symptoms=None,
    positive_symptoms=None,
    note_key=None,
    positive_value=None,
    negative_value=None,
):
    asked_keys = list(st.session_state.get("assistant_asked_question_keys", []) or [])
    if question_key not in asked_keys:
        asked_keys.append(question_key)
    st.session_state.assistant_asked_question_keys = asked_keys
    st.session_state.assistant_pending_question = {
        "key": question_key,
        "type": question_type,
        "target_symptoms": list(target_symptoms or []),
        "positive_symptoms": list(positive_symptoms or []),
        "note_key": note_key,
        "positive_value": positive_value,
        "negative_value": negative_value,
    }


def has_unanswered_differential_question(preview, cumulative_symptoms, notes):
    profile_key = get_differential_profile_key(preview)
    if not profile_key or not prediction_preview_is_low_specificity(preview, cumulative_symptoms, notes):
        return False

    asked_keys = set(st.session_state.get("assistant_asked_question_keys", []) or [])
    symptom_set = set(cumulative_symptoms or [])
    denied_set = set((notes or {}).get("denied_symptoms", []) or [])
    no_more_symptoms = patient_declined_more_symptoms(notes)

    for item in DIFFERENTIAL_CLARIFICATION_PROFILES.get(profile_key, []):
        if item["key"] in asked_keys:
            continue
        if item.get("type", "symptom") == "symptom":
            if no_more_symptoms:
                continue
            targets = [symptom for symptom in item.get("target_symptoms", []) if symptom not in symptom_set and symptom not in denied_set]
            if targets:
                return True
        elif item.get("type") == "context_yes_no":
            note_key = item.get("note_key")
            if note_key and not notes.get(note_key):
                return True
    return False


def is_ready_for_assistant_prediction(cumulative_symptoms, notes, preview=None):
    symptoms = list(cumulative_symptoms or [])
    notes = notes or {}
    patient_finished_symptom_list = patient_declined_more_symptoms(notes)
    if (
        preview
        and float(preview.get("prediction_percent", 0) or 0) < ASSISTANT_MIN_PREDICTION_CONFIDENCE
        and not patient_finished_symptom_list
    ):
        return False
    if has_unanswered_differential_question(preview, symptoms, notes):
        return False
    if not symptoms or not notes.get("duration"):
        return False
    answered_context = len(notes.get("denied_symptoms", []) or [])
    answered_context += 1 if notes.get("medicine") else 0
    answered_context += 1 if notes.get("severity") else 0
    answered_context += 1 if notes.get("temperature") else 0
    answered_context += 1 if notes.get("progression") else 0
    answered_context += 1 if notes.get("risk_context") else 0
    answered_context += 1 if notes.get("exposure_context") else 0
    answered_context += 1 if notes.get("differential_context") else 0
    answered_context += 1 if notes.get("other_symptoms") else 0
    if len(symptoms) >= 3:
        return answered_context >= 1
    if len(symptoms) >= 2:
        return answered_context >= 2
    return answered_context >= 3


def has_confident_prediction_preview(preview):
    return bool(preview) and float(preview.get("prediction_percent", 0) or 0) >= ASSISTANT_MIN_PREDICTION_CONFIDENCE


def get_duration_question(profile_name):
    if "nose" in profile_name or "cold" in profile_name:
        return "How many days has your nose been blocked or runny?"
    if "fever" in profile_name:
        return "How many days have you had the fever?"
    if "cough" in profile_name or "breathing" in profile_name:
        return "How many days have you had the cough or breathing problem?"
    if "stomach" in profile_name or "digestion" in profile_name:
        return "How many days have you had the stomach problem?"
    if "skin" in profile_name:
        return "How many days have you had the skin problem?"
    return "How many days have you been feeling this?"


def get_related_question_targets(profile, cumulative_symptoms, notes):
    symptom_set = set(cumulative_symptoms or [])
    denied_set = set((notes or {}).get("denied_symptoms", []) or [])
    targets = []
    fever_group_added = False

    for symptom in profile.get("related", []):
        if symptom in {"high_fever", "mild_fever"}:
            if fever_group_added:
                continue
            fever_group_added = True
            fever_targets = [item for item in ("mild_fever", "high_fever") if item not in symptom_set and item not in denied_set]
            if fever_targets and not symptom_set & {"mild_fever", "high_fever"}:
                targets.append(
                    {
                        "key": "related:fever",
                        "target_symptoms": fever_targets,
                        "positive_symptoms": ["mild_fever"],
                        "question": "Do you also have fever?",
                    }
                )
            continue

        if symptom in symptom_set or symptom in denied_set:
            continue
        targets.append(
            {
                "key": f"related:{symptom}",
                "target_symptoms": [symptom],
                "positive_symptoms": [symptom],
                "question": f"Do you also have {format_related_symptoms_for_question([symptom], max_items=1)}?",
            }
        )

    return targets


DIFFERENTIAL_CLARIFICATION_PROFILES = {
    "malaria": [
        {
            "key": "malaria:fever",
            "type": "symptom",
            "target_symptoms": ["mild_fever", "high_fever"],
            "positive_symptoms": ["mild_fever"],
            "question": "Do you have fever, or fever that comes and goes?",
        },
        {
            "key": "malaria:chills",
            "type": "symptom",
            "target_symptoms": ["chills", "shivering", "sweating"],
            "positive_symptoms": ["chills"],
            "question": "Do you get chills, shivering, or unusual sweating?",
        },
        {
            "key": "malaria:exposure",
            "type": "context_yes_no",
            "note_key": "exposure_context",
            "positive_value": "mosquito exposure or malaria-risk area mentioned",
            "negative_value": "no mosquito exposure or malaria-risk area mentioned",
            "question": "Any recent mosquito bites, stagnant-water exposure, or stay/travel in a malaria-risk area?",
        },
        {
            "key": "malaria:stomach",
            "type": "symptom",
            "target_symptoms": ["nausea", "vomiting", "diarrhoea"],
            "positive_symptoms": ["nausea"],
            "question": "Do you also have nausea, vomiting, or loose motions?",
        },
    ],
    "dengue": [
        {
            "key": "dengue:fever",
            "type": "symptom",
            "target_symptoms": ["mild_fever", "high_fever"],
            "positive_symptoms": ["high_fever"],
            "question": "Do you have fever, especially sudden or high fever?",
        },
        {
            "key": "dengue:eye_joint_pain",
            "type": "symptom",
            "target_symptoms": ["pain_behind_the_eyes", "joint_pain", "muscle_pain"],
            "positive_symptoms": ["pain_behind_the_eyes"],
            "question": "Do you have pain behind the eyes or strong joint/body pain?",
        },
        {
            "key": "dengue:rash_bleeding",
            "type": "symptom",
            "target_symptoms": ["skin_rash", "red_spots_over_body", "bleeding_gums"],
            "positive_symptoms": ["skin_rash"],
            "question": "Do you have rash, red spots, bleeding gums, nose bleeding, or black stool?",
        },
        {
            "key": "dengue:exposure",
            "type": "context_yes_no",
            "note_key": "exposure_context",
            "positive_value": "mosquito exposure or dengue-risk area mentioned",
            "negative_value": "no mosquito exposure or dengue-risk area mentioned",
            "question": "Any mosquito exposure or dengue cases near your home/workplace recently?",
        },
    ],
    "typhoid": [
        {
            "key": "typhoid:fever",
            "type": "symptom",
            "target_symptoms": ["mild_fever", "high_fever"],
            "positive_symptoms": ["high_fever"],
            "question": "Do you have fever that has continued for several days?",
        },
        {
            "key": "typhoid:stomach",
            "type": "symptom",
            "target_symptoms": ["abdominal_pain", "stomach_pain", "diarrhoea", "constipation", "loss_of_appetite"],
            "positive_symptoms": ["abdominal_pain"],
            "question": "Do you have stomach pain, constipation/loose motions, or loss of appetite?",
        },
        {
            "key": "typhoid:food_water",
            "type": "context_yes_no",
            "note_key": "exposure_context",
            "positive_value": "outside food or unsafe water exposure mentioned",
            "negative_value": "no outside food or unsafe water exposure mentioned",
            "question": "Did this start after outside food, unsafe water, or food from a place you do not usually eat?",
        },
    ],
    "viral": [
        {
            "key": "viral:respiratory",
            "type": "symptom",
            "target_symptoms": ["cough", "throat_irritation", "runny_nose", "congestion", "continuous_sneezing"],
            "positive_symptoms": ["cough"],
            "question": "Do you also have cough, sore throat, runny/blocked nose, or sneezing?",
        },
        {
            "key": "viral:fever_chills",
            "type": "symptom",
            "target_symptoms": ["mild_fever", "high_fever", "chills"],
            "positive_symptoms": ["mild_fever"],
            "question": "Do you have fever or chills with the body pain?",
        },
        {
            "key": "viral:contact",
            "type": "context_yes_no",
            "note_key": "exposure_context",
            "positive_value": "recent contact with sick person or viral illness exposure mentioned",
            "negative_value": "no known sick contact mentioned",
            "question": "Was anyone around you recently sick with fever, cold, cough, or flu-like illness?",
        },
    ],
    "pneumonia": [
        {
            "key": "pneumonia:cough",
            "type": "symptom",
            "target_symptoms": ["cough", "phlegm", "mucoid_sputum", "rusty_sputum"],
            "positive_symptoms": ["cough"],
            "question": "Do you have cough, and is there phlegm?",
        },
        {
            "key": "pneumonia:breathing",
            "type": "symptom",
            "target_symptoms": ["breathlessness", "chest_pain"],
            "positive_symptoms": ["breathlessness"],
            "question": "Do you have breathing difficulty or chest pain?",
        },
        {
            "key": "pneumonia:fever",
            "type": "symptom",
            "target_symptoms": ["mild_fever", "high_fever"],
            "positive_symptoms": ["high_fever"],
            "question": "Do you also have fever with the cough or chest symptoms?",
        },
    ],
    "jaundice": [
        {
            "key": "jaundice:yellow",
            "type": "symptom",
            "target_symptoms": ["yellowish_skin", "yellowing_of_eyes"],
            "positive_symptoms": ["yellowing_of_eyes"],
            "question": "Do your eyes or skin look yellow?",
        },
        {
            "key": "jaundice:urine",
            "type": "symptom",
            "target_symptoms": ["dark_urine", "itching", "abdominal_pain", "loss_of_appetite"],
            "positive_symptoms": ["dark_urine"],
            "question": "Do you have dark urine, itching, stomach pain, or loss of appetite?",
        },
    ],
    "diabetes": [
        {
            "key": "diabetes:thirst",
            "type": "symptom",
            "target_symptoms": ["excessive_thirst", "polyuria", "increased_appetite"],
            "positive_symptoms": ["excessive_thirst"],
            "question": "Do you have excessive thirst, frequent urination, or increased hunger?",
        },
        {
            "key": "diabetes:weight",
            "type": "symptom",
            "target_symptoms": ["weight_loss", "fatigue"],
            "positive_symptoms": ["weight_loss"],
            "question": "Have you had unexplained weight loss or unusual tiredness for many days?",
        },
    ],
    "skin_reaction": [
        {
            "key": "skin_reaction:new_medicine",
            "type": "context_yes_no",
            "note_key": "exposure_context",
            "positive_value": "new medicine/food/cream exposure before rash mentioned",
            "negative_value": "no new medicine/food/cream exposure before rash mentioned",
            "question": "Did this start after a new medicine, food, cream, soap, or allergy exposure?",
        },
        {
            "key": "skin_reaction:danger",
            "type": "symptom",
            "target_symptoms": ["blister", "high_fever", "breathlessness"],
            "positive_symptoms": ["blister"],
            "question": "Do you have blisters, skin peeling, fever, swelling of lips/face, or breathing difficulty?",
        },
    ],
}


def get_differential_profile_key(preview):
    if not preview:
        return ""
    disease_name = normalize_disease_name(preview.get("predicted_disease", ""))
    if not disease_name:
        return ""
    if "malaria" in disease_name:
        return "malaria"
    if "dengue" in disease_name:
        return "dengue"
    if "typhoid" in disease_name:
        return "typhoid"
    if any(term in disease_name for term in ("common cold", "flu", "viral", "allergy")):
        return "viral"
    if any(term in disease_name for term in ("pneumonia", "bronchial asthma", "tuberculosis")):
        return "pneumonia"
    if any(term in disease_name for term in ("jaundice", "hepatitis")):
        return "jaundice"
    if any(term in disease_name for term in ("diabetes", "hypoglycemia", "hyperthyroidism", "hypothyroidism")):
        return "diabetes"
    if any(term in disease_name for term in ("drug reaction", "fungal infection", "allergy", "rash", "psoriasis", "impetigo")):
        return "skin_reaction"
    return ""


def prediction_preview_is_low_specificity(preview, cumulative_symptoms, notes):
    if not preview:
        return True
    symptoms = set(cumulative_symptoms or [])
    denied = set((notes or {}).get("denied_symptoms", []) or [])
    confidence = float(preview.get("prediction_percent", 0) or 0)
    broad_symptoms = {"fatigue", "headache", "muscle_pain", "malaise", "weakness", "body_pain"}
    has_only_broad_core = bool(symptoms) and symptoms.issubset(broad_symptoms | denied)
    return confidence < ASSISTANT_MIN_PREDICTION_CONFIDENCE or has_only_broad_core


def patient_declined_more_symptoms(notes):
    other_symptoms = normalize_text_for_matching((notes or {}).get("other_symptoms", ""))
    return other_symptoms in {"none mentioned", "no other symptoms", "no more symptoms", "no"}


def get_differential_clarification_question(preview, cumulative_symptoms, notes):
    profile_key = get_differential_profile_key(preview)
    if not profile_key or not prediction_preview_is_low_specificity(preview, cumulative_symptoms, notes):
        return ""

    asked_keys = set(st.session_state.get("assistant_asked_question_keys", []) or [])
    symptom_set = set(cumulative_symptoms or [])
    denied_set = set((notes or {}).get("denied_symptoms", []) or [])
    no_more_symptoms = patient_declined_more_symptoms(notes)

    for item in DIFFERENTIAL_CLARIFICATION_PROFILES.get(profile_key, []):
        key = item["key"]
        if key in asked_keys:
            continue
        question_type = item.get("type", "symptom")
        if question_type == "symptom":
            if no_more_symptoms:
                continue
            targets = [symptom for symptom in item.get("target_symptoms", []) if symptom not in symptom_set and symptom not in denied_set]
            if not targets:
                continue
            remember_assistant_question(
                key,
                "symptom",
                target_symptoms=targets,
                positive_symptoms=[symptom for symptom in item.get("positive_symptoms", []) if symptom in targets] or targets[:1],
            )
            return item["question"]
        if question_type == "context_yes_no":
            note_key = item.get("note_key")
            if note_key and notes.get(note_key):
                continue
            remember_assistant_question(
                key,
                "context_yes_no",
                note_key=note_key,
                positive_value=item.get("positive_value"),
                negative_value=item.get("negative_value"),
            )
            return item["question"]

    return ""


def needs_more_differential_clarification(preview, cumulative_symptoms, notes):
    return has_unanswered_differential_question(preview, cumulative_symptoms, notes)


def get_next_patient_question(cumulative_symptoms, notes, allow_predict=True, preview=None):
    notes = notes or {}
    asked_keys = set(st.session_state.get("assistant_asked_question_keys", []) or [])
    profile = get_symptom_question_profile(cumulative_symptoms)
    profile_name = profile.get("name", "current symptoms")
    no_more_symptoms = patient_declined_more_symptoms(notes)

    if not cumulative_symptoms:
        key = "main_symptom"
        if key not in asked_keys:
            remember_assistant_question(key, "open")
            return "What symptom is bothering you the most right now?"

    duration_key = f"duration:{profile_name}"
    if cumulative_symptoms and not notes.get("duration") and duration_key not in asked_keys:
        remember_assistant_question(duration_key, "duration")
        return get_duration_question(profile_name)

    if set(cumulative_symptoms or []) & {"high_fever", "mild_fever"} and not notes.get("temperature"):
        key = "temperature:fever"
        if key not in asked_keys:
            remember_assistant_question(key, "temperature")
            return "Did you check your temperature? If yes, how much was it?"

    differential_question = get_differential_clarification_question(preview, cumulative_symptoms, notes)
    if differential_question:
        return differential_question

    if not no_more_symptoms:
        related_asked_count = len([key for key in asked_keys if str(key).startswith("related:")])
        for target in get_related_question_targets(profile, cumulative_symptoms, notes):
            if related_asked_count >= 3 and is_ready_for_assistant_prediction(cumulative_symptoms, notes, preview):
                break
            if target["key"] in asked_keys:
                continue
            remember_assistant_question(
                target["key"],
                "symptom",
                target_symptoms=target["target_symptoms"],
                positive_symptoms=target["positive_symptoms"],
            )
            return target["question"]

    medicine_key = f"medicine:{profile_name}"
    if cumulative_symptoms and not notes.get("medicine") and medicine_key not in asked_keys:
        remember_assistant_question(medicine_key, "medicine")
        return "Have you taken any medicine for this?"

    severity_key = f"severity:{profile_name}"
    if cumulative_symptoms and not notes.get("severity") and severity_key not in asked_keys:
        remember_assistant_question(severity_key, "severity")
        return "Is it mild, moderate, or severe right now?"

    progression_key = f"progression:{profile_name}"
    if cumulative_symptoms and not notes.get("progression") and progression_key not in asked_keys:
        remember_assistant_question(progression_key, "progression")
        return "Is it getting better, getting worse, or staying the same?"

    risk_key = f"risk:{profile_name}"
    if (
        cumulative_symptoms
        and not notes.get("risk_context")
        and risk_key not in asked_keys
        and (set(cumulative_symptoms) & {"breathlessness", "chest_pain", "high_fever", "mild_fever", "cough", "vomiting", "diarrhoea"})
    ):
        remember_assistant_question(risk_key, "risk_context")
        return "Do you have asthma, diabetes, pregnancy, heart disease, high BP, or low immunity?"

    if allow_predict and is_ready_for_assistant_prediction(cumulative_symptoms, notes, preview):
        key = "predict_confirmation"
        remember_assistant_question(key, "predict_confirmation")
        return "I have enough details to run the screening now. Should I predict it?"

    if not no_more_symptoms:
        for target in get_related_question_targets(profile, cumulative_symptoms, notes):
            if target["key"] in asked_keys:
                continue
            remember_assistant_question(
                target["key"],
                "symptom",
                target_symptoms=target["target_symptoms"],
                positive_symptoms=target["positive_symptoms"],
            )
            return target["question"]

    key = "other_symptom"
    if key in asked_keys:
        if cumulative_symptoms:
            return "I saved those details. If nothing else is present, type 'predict now' and I will run the screening with what we have."
        return ""
    remember_assistant_question(key, "open")
    return "Is there any other symptom you have not told me yet?"


def build_human_prediction_reply(preview, care_plan, cumulative_symptoms, urgent_note):
    if not preview:
        return "I need at least one clear symptom before I can run the screening."

    notes = st.session_state.get("assistant_clinical_notes", {}) or {}
    lines = [
        f"The screening result points toward **{preview['predicted_disease']}**.",
        f"I considered: {format_symptom_summary(cumulative_symptoms)}.",
        get_symptom_reasoning_line(cumulative_symptoms, notes, preview, care_plan),
    ]
    notes_summary = build_clinical_notes_summary(notes)
    if notes_summary and "no duration" not in notes_summary:
        lines.append(f"Extra details saved: {notes_summary}.")

    external_evidence = get_external_prediction_evidence(preview["predicted_disease"], tuple(cumulative_symptoms))
    evidence_text = format_external_evidence_for_chat(external_evidence)
    if evidence_text:
        lines.append(evidence_text)

    lab_guidance = build_lab_test_guidance(care_plan)
    if lab_guidance:
        lines.append(lab_guidance)

    medicine_guidance = build_medicine_safety_guidance(notes, cumulative_symptoms, care_plan)
    if medicine_guidance:
        lines.append(medicine_guidance)
    
    # Add home remedies and medicines
    symptom_remedies_df = load_symptom_remedies()
    selected_symptoms_lower = [s.lower() for s in cumulative_symptoms]
    matched = symptom_remedies_df[symptom_remedies_df["symptom"].isin(selected_symptoms_lower)]
    
    if not matched.empty:
        lines.append("\nSuggested home care:")
        for _, row in matched.drop_duplicates(subset=["symptom"]).iterrows():
            symptom_name = row["symptom"].replace("_", " ").title()
            remedies = [item.strip() for item in str(row["home_remedies"]).split(";") if item.strip()]
            remedy_text = ", ".join(remedies[:3])
            lines.append(f"- {symptom_name}: {remedy_text}")

    if care_plan:
        lines.append(
            f"\nNext step: consider a {care_plan['specialist']} consultation if symptoms continue, worsen, or feel unusual. "
            f"Care level: {care_plan['severity'].lower()}."
        )
        warning_text = format_assistant_items(care_plan.get("warning_signs", []), 3)
        if warning_text:
            lines.append(f"Watch for: {warning_text}.")
    lines.append("\nThis is screening support only, so please confirm it with a qualified doctor.")
    if urgent_note:
        lines.append(urgent_note)
    return "\n".join(lines)


def build_patient_conversation_reply(user_message, detected_now, cumulative_symptoms, notes, preview, care_plan, urgent_note, allow_predict=True):
    question = get_next_patient_question(cumulative_symptoms, notes, allow_predict=allow_predict, preview=preview)
    if detected_now:
        intro = f"I noted {format_related_symptoms_for_question(detected_now, max_items=3)}."
    elif cumulative_symptoms:
        intro = "Thanks, I saved that."
    else:
        intro = "I could not clearly catch a symptom from that message."

    reasoning_line = get_symptom_reasoning_line(cumulative_symptoms, notes, preview, None)
    parts = [intro]
    if cumulative_symptoms:
        parts.append(f"Saved symptoms: {format_symptom_summary(cumulative_symptoms)}.")
    if reasoning_line and reasoning_line != intro:
        parts.append(reasoning_line)
    if preview:
        parts.append("I am using this as screening support, not a confirmed diagnosis.")
    elif cumulative_symptoms:
        parts.append("I need a little more detail before the screening result will be useful.")
    medicine_guidance = build_medicine_safety_guidance(notes, cumulative_symptoms, care_plan)
    if medicine_guidance and medicine_guidance not in parts:
        parts.append(medicine_guidance)
    if urgent_note:
        parts.append(urgent_note)
    elif cumulative_symptoms:
        parts.append("If symptoms become severe, sudden, or rapidly worse, please seek medical care promptly.")
    if question:
        parts.append(question)
    return "\n\n".join(parts)


def build_doctor_style_interview_reply(user_message, detected_now, cumulative_symptoms, notes, preview, care_plan, urgent_note):
    return build_patient_conversation_reply(
        user_message,
        detected_now,
        cumulative_symptoms,
        notes,
        preview,
        care_plan,
        urgent_note,
    )


def predict_from_symptom_list(model, label_encoder, feature_names, selected_symptoms):
    symptom_vector = {symptom: 0 for symptom in feature_names}
    for symptom in selected_symptoms:
        if symptom in symptom_vector:
            symptom_vector[symptom] = 1

    feature_df = pd.DataFrame([symptom_vector.values()], columns=feature_names)
    prediction_value = np.asarray(model.predict(feature_df))[0]
    if np.issubdtype(np.asarray([prediction_value]).dtype, np.floating):
        prediction_value = int(np.rint(prediction_value))

    class_count = len(label_encoder.classes_)
    prediction_value = int(np.clip(prediction_value, 0, class_count - 1))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_df)[0]
        class_labels = model.classes_
    else:
        probabilities = np.zeros(class_count, dtype=float)
        probabilities[prediction_value] = 1.0
        class_labels = np.arange(class_count)

    predicted_disease = label_encoder.inverse_transform([prediction_value])[0]
    predicted_class_pos = np.where(class_labels == prediction_value)[0][0]
    prediction_percent = float(probabilities[predicted_class_pos] * 100)

    return {
        "prediction": prediction_value,
        "predicted_disease": predicted_disease,
        "probabilities": probabilities,
        "class_labels": class_labels,
        "prediction_percent": prediction_percent,
    }


def format_symptom_summary(symptoms):
    if not symptoms:
        return "None yet"
    return ", ".join(format_symptom_label(symptom) for symptom in symptoms)


def strip_live_ai_noise(text):
    cleaned = str(text or "")
    for pattern in ASSISTANT_SUPPRESSED_LIVE_ERROR_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def reset_symptom_assistant_state(clear_connection=False):
    st.session_state.symptom_chat_history = []
    st.session_state.assistant_detected_symptoms = []
    st.session_state.assistant_clinical_notes = {}
    st.session_state.assistant_pending_question = {}
    st.session_state.assistant_asked_question_keys = []
    st.session_state.assistant_trigger_predict = False
    if clear_connection:
        st.session_state.assistant_panel_open = False


def get_prediction_preview(model, label_encoder, feature_names, selected_symptoms):
    if not selected_symptoms:
        return None
    return predict_from_symptom_list(model, label_encoder, feature_names, selected_symptoms)


def get_assistant_reference_prediction(model, label_encoder, feature_names, selected_symptoms):
    preview = get_prediction_preview(model, label_encoder, feature_names, selected_symptoms)
    if preview:
        return preview

    prediction_context = st.session_state.get("prediction_context")
    if not prediction_context:
        return None
    if prediction_context.get("guarded"):
        return None

    return {
        "prediction": int(prediction_context.get("prediction", 0)),
        "predicted_disease": str(prediction_context.get("predicted_disease", "")).strip(),
        "probabilities": np.array(prediction_context.get("probabilities", []), dtype=float),
        "class_labels": np.array(prediction_context.get("class_labels", []), dtype=int),
        "prediction_percent": float(prediction_context.get("confidence", 0.0)),
    }


def build_top_prediction_dataframe(label_encoder, class_labels, probabilities, limit=5):
    probabilities = np.asarray(probabilities, dtype=float)
    class_labels = np.asarray(class_labels, dtype=int)
    usable_count = min(len(probabilities), len(class_labels))
    if usable_count == 0:
        return pd.DataFrame(columns=["Disease", "Model Score"])

    ranked_indices = np.argsort(probabilities[:usable_count])[-limit:][::-1]
    top_class_values = class_labels[ranked_indices]
    top_candidate_diseases = label_encoder.inverse_transform(top_class_values).tolist()
    return pd.DataFrame(
        {
            "Disease": top_candidate_diseases,
            "Model Score": [float(probabilities[idx] * 100) for idx in ranked_indices],
        }
    )


def get_prediction_confidence_summary(confidence, top_probability_df):
    if confidence >= 70:
        label = "Strong"
        tone = "success"
        guidance = "The leading condition is clearly stronger than most alternatives for the symptoms selected."
    elif confidence >= 45:
        label = "Moderate"
        tone = "info"
        guidance = "This is useful for screening, but it still needs symptom review and clinical confirmation."
    elif confidence >= 25:
        label = "Low"
        tone = "warning"
        guidance = "Several diseases can share these symptoms. Add more specific symptoms before trusting the first result."
    else:
        label = "Very low"
        tone = "warning"
        guidance = "The symptom set is too broad right now. Treat this as a shortlist, not a clear disease answer."

    runner_up_gap = None
    if len(top_probability_df) >= 2:
        runner_up_gap = float(
            top_probability_df["Model Score"].iloc[0] - top_probability_df["Model Score"].iloc[1]
        )

    if runner_up_gap is not None and runner_up_gap < 8:
        guidance += " The top diseases are close to each other, so comparing the alternatives is important."

    return {
        "label": label,
        "tone": tone,
        "guidance": guidance,
        "runner_up_gap": runner_up_gap,
    }


def render_prediction_confidence_notice(confidence_summary, model_accuracy):
    intro = (
        f"{confidence_summary['label']} confidence: this percentage is model confidence for the selected symptoms, "
        f"not the saved model accuracy. Saved model accuracy is {model_accuracy:.1%}."
    )
    message = f"{intro} {confidence_summary['guidance']}"
    if confidence_summary["tone"] == "success":
        st.success(message)
    elif confidence_summary["tone"] == "info":
        st.info(message)
    else:
        st.warning(message)


def detect_assistant_action(normalized_message, cumulative_symptoms):
    if not cumulative_symptoms:
        return None

    if message_has_any_phrase(normalized_message, ASSISTANT_PREDICT_TRIGGERS):
        return "predict"
    if message_has_any_phrase(normalized_message, ASSISTANT_LOAD_TRIGGERS):
        return "load"
    return None


def get_recommended_assistant_prompts(detected_symptoms):
    if detected_symptoms:
        return ASSISTANT_FOLLOWUP_PROMPTS
    return ASSISTANT_STARTER_PROMPTS


def get_configured_gemini_api_key():
    env_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        return (
            str(st.secrets.get("GEMINI_API_KEY", "")).strip()
            or str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
        )
    except Exception:
        return ""


def get_configured_nhs_api_key():
    env_key = os.getenv("NHS_WEBSITE_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        return str(st.secrets.get("NHS_WEBSITE_API_KEY", "")).strip()
    except Exception:
        return ""


def clean_external_text(text, max_chars=900):
    cleaned = html.unescape(str(text or ""))
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if max_chars and len(cleaned) > max_chars:
        return cleaned[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return cleaned


def external_source_request(url, headers=None, timeout=14):
    request_headers = {"User-Agent": EXTERNAL_EVIDENCE_USER_AGENT}
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 18)
def query_medlineplus_health_topics(query, limit=3):
    query = str(query or "").strip()
    if not query:
        return []

    url = (
        f"{MEDLINEPLUS_SEARCH_URL}?db=healthTopics&term={quote_plus(query)}"
        f"&retmax={int(limit)}&rettype=brief&tool=disease_prediction_dashboard"
    )
    try:
        raw_xml = external_source_request(url)
        root = ET.fromstring(raw_xml)
    except Exception:
        return []

    results = []
    for document in root.findall(".//document"):
        contents = {}
        for content_node in document.findall("content"):
            name = str(content_node.attrib.get("name", "")).strip()
            value = clean_external_text(content_node.text or "")
            if not name or not value:
                continue
            if name in contents:
                contents[name] = f"{contents[name]} {value}"
            else:
                contents[name] = value
        title = contents.get("title", "").strip()
        summary = contents.get("FullSummary", contents.get("snippet", "")).strip()
        if title:
            results.append(
                {
                    "source": "MedlinePlus",
                    "title": title,
                    "summary": summary,
                    "url": str(document.attrib.get("url", "")).strip(),
                }
            )
    return results


@st.cache_data(show_spinner=False, ttl=60 * 60 * 18)
def query_nlm_conditions(query, limit=5):
    query = str(query or "").strip()
    if not query:
        return []

    url = (
        f"{NLM_CONDITIONS_SEARCH_URL}?terms={quote_plus(query)}"
        f"&maxList={int(limit)}&df=primary_name,consumer_name,info_link_data"
    )
    try:
        payload = json.loads(external_source_request(url))
    except Exception:
        return []

    rows = payload[3] if isinstance(payload, list) and len(payload) > 3 and isinstance(payload[3], list) else []
    results = []
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        title = clean_external_text(row[0])
        consumer_name = clean_external_text(row[1]) if len(row) > 1 else ""
        link_data = str(row[2] if len(row) > 2 else "")
        link_url = ""
        if "," in link_data:
            link_url = link_data.split(",", 1)[0].strip()
        elif link_data.startswith("http"):
            link_url = link_data.strip()
        if title:
            results.append(
                {
                    "source": "NLM Clinical Tables",
                    "title": title,
                    "summary": consumer_name or "Condition vocabulary match from the National Library of Medicine.",
                    "url": link_url,
                }
            )
    return results


def flatten_nhs_text(value):
    pieces = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"text", "description", "headline", "name", "url"}:
                pieces.append(str(item))
            elif isinstance(item, (dict, list)):
                pieces.append(flatten_nhs_text(item))
    elif isinstance(value, list):
        for item in value:
            pieces.append(flatten_nhs_text(item))
    elif isinstance(value, str):
        pieces.append(value)
    return clean_external_text(" ".join(piece for piece in pieces if piece), max_chars=1400)


def slugify_nhs_condition_name(name):
    normalized = normalize_disease_name(name)
    replacements = {
        "diabetes": "diabetes",
        "diabetes ": "diabetes",
        "gerd": "heartburn-and-acid-reflux",
        "paralysis brain hemorrhage": "stroke",
        "dimorphic hemmorhoids piles": "piles-haemorrhoids",
        "peptic ulcer diseae": "stomach-ulcer",
    }
    normalized = replacements.get(normalized, normalized)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug


@st.cache_data(show_spinner=False, ttl=60 * 60 * 18)
def query_nhs_condition_content(predicted_disease):
    api_key = get_configured_nhs_api_key()
    if not api_key:
        return []

    slug = slugify_nhs_condition_name(predicted_disease)
    if not slug:
        return []

    url = f"{NHS_CONTENT_API_BASE_URL}/conditions/{slug}/?modules=true"
    try:
        payload = json.loads(
            external_source_request(
                url,
                headers={"apikey": api_key, "Accept": "application/json"},
                timeout=16,
            )
        )
    except Exception:
        return []

    title = clean_external_text(payload.get("name", predicted_disease))
    summary = flatten_nhs_text(
        {
            "description": payload.get("description", ""),
            "hasPart": payload.get("hasPart", []),
            "about": payload.get("about", []),
        }
    )
    page_url = str(payload.get("url", "")).strip()
    if title:
        return [
            {
                "source": "NHS",
                "title": title,
                "summary": summary,
                "url": page_url,
            }
        ]
    return []


EXTERNAL_DISEASE_QUERY_ALIASES = {
    "paralysis (brain hemorrhage)": ["stroke", "brain hemorrhage", "brain haemorrhage"],
    "gerd": ["acid reflux", "heartburn"],
    "dimorphic hemmorhoids(piles)": ["piles", "hemorrhoids", "haemorrhoids"],
    "peptic ulcer diseae": ["stomach ulcer", "peptic ulcer"],
    "(vertigo) paroymsal positional vertigo": ["vertigo", "benign paroxysmal positional vertigo"],
    "diabetes": ["diabetes"],
}

EXTERNAL_SYMPTOM_QUERY_TERMS = {
    "mild_fever": ["fever", "high temperature"],
    "high_fever": ["high fever", "fever"],
    "muscle_pain": ["muscle pain", "body aches", "myalgia"],
    "body_pain": ["body aches", "body pain"],
    "breathlessness": ["shortness of breath", "breathing difficulty"],
    "diarrhoea": ["diarrhea", "loose stools"],
    "runny_nose": ["runny nose", "nasal discharge"],
    "congestion": ["blocked nose", "stuffy nose", "nasal congestion"],
    "throat_irritation": ["sore throat", "throat irritation"],
    "red_spots_over_body": ["rash", "red spots"],
    "altered_sensorium": ["confusion", "reduced alertness"],
    "blurred_and_distorted_vision": ["blurred vision"],
}


def get_external_disease_queries(predicted_disease):
    normalized = normalize_disease_name(predicted_disease)
    queries = [str(predicted_disease or "").strip()]
    queries.extend(EXTERNAL_DISEASE_QUERY_ALIASES.get(normalized, []))
    cleaned = []
    for query in queries:
        query = clean_external_text(query, max_chars=120)
        if query and normalize_text_for_matching(query) not in {normalize_text_for_matching(item) for item in cleaned}:
            cleaned.append(query)
    return cleaned[:3]


def get_external_symptom_terms(symptoms):
    symptom_terms = {}
    for symptom in symptoms or []:
        terms = [format_symptom_label(symptom).lower()]
        terms.extend(EXTERNAL_SYMPTOM_QUERY_TERMS.get(symptom, []))
        normalized_terms = []
        for term in terms:
            normalized = normalize_text_for_matching(term)
            if normalized and normalized not in normalized_terms:
                normalized_terms.append(normalized)
        symptom_terms[symptom] = normalized_terms
    return symptom_terms


def document_matches_external_terms(document, disease_terms, symptom_terms):
    combined_text = normalize_text_for_matching(
        " ".join(
            str(document.get(key, ""))
            for key in ("title", "summary")
        )
    )
    disease_match = any(term and term in combined_text for term in disease_terms)
    matched_symptoms = []
    for symptom, terms in symptom_terms.items():
        if any(term and term in combined_text for term in terms):
            matched_symptoms.append(symptom)
    return disease_match, matched_symptoms


@st.cache_data(show_spinner=False, ttl=60 * 60 * 18)
def get_external_prediction_evidence(predicted_disease, symptoms_tuple):
    symptoms = tuple(symptoms_tuple or ())
    if not EXTERNAL_MEDICAL_EVIDENCE_ENABLED:
        return {"enabled": False, "support": "disabled", "sources": [], "errors": []}

    disease_queries = get_external_disease_queries(predicted_disease)
    symptom_terms = get_external_symptom_terms(symptoms)
    disease_terms = [normalize_text_for_matching(query) for query in disease_queries if query]
    sources = []

    for query in disease_queries[:2]:
        sources.extend(query_medlineplus_health_topics(query, limit=3))
        sources.extend(query_nlm_conditions(query, limit=4))
    sources.extend(query_nhs_condition_content(predicted_disease))

    unique_sources = []
    seen_keys = set()
    for source in sources:
        key = (source.get("source"), source.get("title"), source.get("url"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_sources.append(source)

    source_summaries = []
    verified_condition_sources = set()
    symptom_support = set()
    disease_symptom_sources = set()
    for source in unique_sources:
        disease_match, matched_symptoms = document_matches_external_terms(source, disease_terms, symptom_terms)
        if disease_match:
            verified_condition_sources.add(source.get("source", ""))
        if matched_symptoms:
            symptom_support.update(matched_symptoms)
        if disease_match and matched_symptoms:
            disease_symptom_sources.add(source.get("source", ""))
        source_summaries.append(
            {
                **source,
                "disease_match": disease_match,
                "matched_symptoms": matched_symptoms,
            }
        )

    required_symptom_matches = min(2, max(1, len(symptoms)))
    if disease_symptom_sources and len(symptom_support) >= required_symptom_matches:
        support = "supported"
    elif verified_condition_sources or symptom_support:
        support = "partial"
    else:
        support = "not_found"

    checked_source_names = sorted({source.get("source", "") for source in unique_sources if source.get("source")})
    if get_configured_nhs_api_key():
        checked_source_names.append("NHS")
        checked_source_names = sorted(set(checked_source_names))

    return {
        "enabled": True,
        "support": support,
        "predicted_disease": predicted_disease,
        "symptom_support": sorted(symptom_support),
        "verified_condition_sources": sorted(source for source in verified_condition_sources if source),
        "disease_symptom_sources": sorted(source for source in disease_symptom_sources if source),
        "checked_sources": checked_source_names,
        "sources": source_summaries[:6],
        "nhs_configured": bool(get_configured_nhs_api_key()),
    }


def format_external_evidence_for_chat(evidence):
    if not evidence or not evidence.get("enabled"):
        return ""

    checked_sources = evidence.get("checked_sources") or ["MedlinePlus", "NLM Clinical Tables"]
    source_text = format_natural_list(checked_sources, 3)
    support = evidence.get("support")
    if support == "supported":
        status = "the external source check supports the local screening direction."
    elif support == "partial":
        status = "the external source check found partial support, so this should stay as a cautious screening result."
    else:
        status = "the external source check did not find enough support for this disease label from the saved symptoms."

    lines = [
        "External source check:",
        f"- Checked: {source_text}.",
        f"- Result: {status}",
    ]
    symptom_support = evidence.get("symptom_support") or []
    if symptom_support:
        lines.append(f"- Symptoms found in external content: {format_symptom_summary(symptom_support)}.")
    top_links = [
        f"{source.get('source')}: {source.get('title')}"
        for source in evidence.get("sources", [])
        if source.get("title")
    ][:2]
    if top_links:
        lines.append(f"- Reference topics: {' | '.join(top_links)}.")
    return "\n".join(lines)


def render_external_evidence_panel(evidence):
    if not evidence or not evidence.get("enabled"):
        return

    render_section_intro(
        "External Source Check",
        "The local model result is compared with trusted medical reference APIs before you use it.",
    )
    support = evidence.get("support")
    if support == "supported":
        st.success("External source check supports this screening direction, but it is still not a confirmed diagnosis.")
    elif support == "partial":
        st.warning("External source check found partial support only. Compare symptoms carefully before trusting the result.")
    else:
        st.error("External source check did not find enough support for this disease label from the selected symptoms.")

    checked_sources = evidence.get("checked_sources") or ["MedlinePlus", "NLM Clinical Tables"]
    st.caption(f"Checked sources: {format_natural_list(checked_sources, 4)}")

    for source in evidence.get("sources", [])[:4]:
        title = source.get("title", "")
        source_name = source.get("source", "External source")
        matched = format_symptom_summary(source.get("matched_symptoms", [])) if source.get("matched_symptoms") else "No saved symptoms matched in this snippet"
        url = source.get("url", "")
        if url:
            st.markdown(f"- **{source_name}:** [{title}]({url}) — {matched}")
        else:
            st.markdown(f"- **{source_name}:** {title} — {matched}")


def get_symptom_specific_remedies(symptoms, max_rows=3):
    if not symptoms:
        return []

    remedies_df = load_symptom_remedies()
    selected_symptoms_lower = [str(symptom).lower() for symptom in symptoms]
    matched = remedies_df[remedies_df["symptom"].isin(selected_symptoms_lower)]
    remedy_lines = []

    for _, row in matched.drop_duplicates(subset=["symptom"]).iterrows():
        remedies = [item.strip() for item in str(row["home_remedies"]).split(";") if item.strip()]
        if remedies:
            remedy_lines.append(f"{row['symptom'].replace('_', ' ').title()}: {', '.join(remedies[:2])}")

    return remedy_lines[:max_rows]


def format_natural_list(items, max_items=3):
    cleaned_items = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned_items:
        return ""

    visible_items = cleaned_items[:max_items]
    if len(cleaned_items) > max_items:
        visible_items.append("more")

    if len(visible_items) == 1:
        return visible_items[0]
    if len(visible_items) == 2:
        return " and ".join(visible_items)
    return ", ".join(visible_items[:-1]) + f", and {visible_items[-1]}"


def build_screening_snapshot(preview, care_plan, cumulative_symptoms):
    symptom_text = format_symptom_summary(cumulative_symptoms)
    if preview and care_plan:
        return (
            f"Saved symptoms: {symptom_text}. Current screening preview: "
            f"{preview['predicted_disease']}. "
            f"Care level: {care_plan['severity'].lower()}. Suggested specialist: {care_plan['specialist']}."
        )
    if cumulative_symptoms:
        return f"Saved symptoms: {symptom_text}. Add more symptoms or ask me to run a prediction."
    return "No symptoms are saved yet."


def build_urgent_symptom_note(symptoms):
    urgent_symptom_notes = {
        "chest_pain": "Chest pain can be urgent, especially with sweating, fainting, or breathlessness.",
        "shortness_of_breath": "Breathing difficulty can become urgent quickly.",
        "breathlessness": "Breathing difficulty can become urgent quickly.",
        "coma": "Loss of consciousness is an emergency.",
        "altered_sensorium": "Confusion or reduced alertness needs urgent medical review.",
        "continuous_vomiting": "Repeated vomiting can cause dehydration and may need urgent care.",
        "dehydration": "Signs of dehydration may need prompt medical care.",
        "high_fever": "High fever needs faster review if it is persistent, severe, or paired with weakness.",
    }
    notes = [urgent_symptom_notes[symptom] for symptom in symptoms if symptom in urgent_symptom_notes]
    if not notes:
        return ""

    return f"Important: {notes[0]} If this is severe, sudden, or worsening, seek urgent medical care now."


HEADACHE_RED_FLAG_SYMPTOMS = {
    "high_fever",
    "stiff_neck",
    "vomiting",
    "continuous_vomiting",
    "altered_sensorium",
    "coma",
    "loss_of_consciousness",
    "slurred_speech",
    "weakness_of_one_body_side",
    "weakness_in_limbs",
    "loss_of_balance",
    "unsteadiness",
    "blurred_and_distorted_vision",
    "visual_disturbances",
    "dizziness",
}

EMERGENCY_MISMATCH_DISEASE_TERMS = (
    "paralysis",
    "brain hemorrhage",
    "brain haemorrhage",
    "stroke",
    "heart attack",
)

COMMON_SINGLE_SYMPTOMS = {"headache", "fatigue", "muscle_pain", "body_pain", "malaise", "weakness"}


def headache_has_emergency_red_flags(symptoms, notes):
    symptom_set = set(symptoms or [])
    if symptom_set & HEADACHE_RED_FLAG_SYMPTOMS:
        return True

    notes = notes or {}
    severity = normalize_text_for_matching(notes.get("severity", ""))
    progression = normalize_text_for_matching(notes.get("progression", ""))
    context_text = normalize_text_for_matching(
        " ".join(str(notes.get(key, "")) for key in ("exposure_context", "differential_context", "other_symptoms"))
    )
    if any(term in severity for term in ("severe", "extreme", "unbearable", "worst")):
        return True
    if any(term in progression for term in ("worse", "worsening", "sudden")):
        return True
    return any(
        term in context_text
        for term in (
            "sudden",
            "worst headache",
            "head injury",
            "confusion",
            "face drooping",
            "arm weakness",
            "speech",
            "vision",
            "faint",
            "stiff neck",
        )
    )


def should_guard_nonspecific_prediction(preview, symptoms, notes):
    if not preview:
        return False

    symptom_set = set(symptoms or [])
    disease_name = normalize_disease_name(preview.get("predicted_disease", ""))
    confidence = float(preview.get("prediction_percent", 0) or 0)
    emergency_mismatch = any(term in disease_name for term in EMERGENCY_MISMATCH_DISEASE_TERMS)

    if symptom_set == {"headache"} and emergency_mismatch and not headache_has_emergency_red_flags(symptom_set, notes):
        return True

    if len(symptom_set) <= 1 and symptom_set.issubset(COMMON_SINGLE_SYMPTOMS) and emergency_mismatch:
        return confidence < ASSISTANT_MIN_PREDICTION_CONFIDENCE

    return False


def build_nonspecific_prediction_guard_reply(preview, symptoms, notes):
    symptom_text = format_symptom_summary(symptoms)
    medicine_guidance = ""
    medicine_text = str((notes or {}).get("medicine", "")).strip()
    if medicine_text:
        medicine_guidance = (
            f"\nMedicine safety: I noted {medicine_text}. Avoid repeating Dolo/Crocin/paracetamol products together, "
            "and be extra careful with liver disease or alcohol use."
        )

    return (
        f"I do not want to label a severe emergency condition from {symptom_text.lower()} alone.\n\n"
        "With a mild headache and no other saved warning symptoms, the symptom set is too nonspecific for a useful disease prediction.\n"
        "Simple care for now: drink fluids, rest in a quiet/dark room, avoid repeated pain-medicine dosing, and track whether it improves."
        f"{medicine_guidance}\n\n"
        "Seek urgent care now if the headache is sudden or severe, follows a head injury, or comes with face drooping, arm/leg weakness, speech trouble, confusion, vision changes, stiff neck with fever, repeated vomiting, fainting, or the worst headache of life.\n"
        "If it persists, keeps recurring, or needs repeated medicine, consult a general physician or neurologist."
    )


def format_assistant_items(items, max_items=4):
    cleaned_items = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned_items:
        return ""
    return "; ".join(cleaned_items[:max_items])


def format_lab_test_summary(lab_tests, max_items=3):
    if not lab_tests:
        return "No specific lab tests are highlighted for the current screening view."

    names = [str(item.get("Test", "")).strip() for item in lab_tests[:max_items] if str(item.get("Test", "")).strip()]
    if not names:
        return "No specific lab tests are highlighted for the current screening view."

    summary = format_natural_list(names, max_items)
    if len(lab_tests) > max_items:
        summary += ", and more"
    return summary


def lab_requirement_status(value):
    normalized_value = normalize_text_for_matching(value)
    if normalized_value == "yes":
        return "yes"
    if normalized_value == "no":
        return "no"
    return "depends"


def format_lab_requirement(label, value):
    status = lab_requirement_status(value)
    if status == "yes":
        return f"- {label}: Yes, usually advised for this screening result."
    if status == "no":
        return f"- {label}: No, not usually needed unless symptoms persist, worsen, or a doctor advises it."
    return f"- {label}: May be needed after a doctor review or if symptoms are persistent, severe, or unclear."


def format_lab_requirement_brief(value):
    status = lab_requirement_status(value)
    if status == "yes":
        return "Yes - usually advised for this screening result."
    if status == "no":
        return "No - not usually needed unless symptoms persist, worsen, or a doctor advises it."
    return "May be needed after doctor review or if symptoms are persistent, severe, or unclear."


def lab_tests_are_required(care_plan):
    if not care_plan:
        return False
    return (
        lab_requirement_status(care_plan.get("lab_test_required", "")) == "yes"
        or lab_requirement_status(care_plan.get("blood_report_required", "")) == "yes"
    )


def build_lab_test_guidance(care_plan):
    if not care_plan:
        return ""

    lines = [
        "Lab test guidance:",
        format_lab_requirement("Blood report", care_plan.get("blood_report_required", "")),
        format_lab_requirement("Lab tests", care_plan.get("lab_test_required", "")),
    ]
    if care_plan.get("lab_tests"):
        lines.append(f"- Suggested tests: {format_lab_test_summary(care_plan['lab_tests'], 4)}.")
        lines.append("- Use Lab Test Booking only for tests that fit your symptoms or were advised by a doctor/lab.")
    else:
        lines.append("- No specific lab test is usually required unless symptoms persist, worsen, or a doctor advises testing.")
    return "\n".join(lines)


def build_structured_assistant_reply(
    detected_now,
    cumulative_symptoms,
    preview,
    care_plan,
    remedy_lines,
    urgent_note,
    focus="overview",
    intro="",
):
    lines = []
    if intro:
        lines.append(intro)

    lines.append("What I understood:")
    if detected_now:
        lines.append(f"- New symptoms found: {format_symptom_summary(detected_now)}.")
    elif cumulative_symptoms:
        lines.append(f"- Saved symptoms: {format_symptom_summary(cumulative_symptoms)}.")
    else:
        lines.append("- I do not have enough saved symptoms yet.")

    lines.append("Screening view:")
    if preview and care_plan:
        lines.append(
            f"- Current app preview: {preview['predicted_disease']}, not a confirmed diagnosis."
        )
        lines.append(f"- Care level: {care_plan['severity']}. Suggested specialist: {care_plan['specialist']}.")
    else:
        lines.append("- I need a clearer symptom set before the app can give a useful prediction preview.")

    if focus == "booking":
        lines.append("Doctor booking guidance:")
        if care_plan:
            lines.append(f"- Best starting point: {care_plan['specialist']}. {care_plan['doctor_urgency']}")
        else:
            lines.append("- Share symptoms or run prediction first, then choose a suitable doctor or clinic.")
        lines.append("- After prediction, use Search Doctors / Find Hospitals, select a recommendation, then save date and time.")
    elif focus == "tests":
        lines.append("Tests and reports:")
        if care_plan:
            lines.append(format_lab_requirement("Blood report", care_plan.get("blood_report_required", "")))
            lines.append(format_lab_requirement("Lab tests", care_plan.get("lab_test_required", "")))
            lines.append(f"- Tests to discuss: {format_lab_test_summary(care_plan['lab_tests'])}")
            lines.append("- If tests are needed, use Lab Test Booking after prediction to add tests to cart and book a slot.")
            lines.append(f"- Note: {care_plan['note']}")
        else:
            lines.append("- Test advice depends on the symptom pattern and should be confirmed by a doctor.")
    elif focus == "remedies":
        lines.append("Care guidance:")
        notes = st.session_state.get("assistant_clinical_notes", {}) or {}
        medicine_guidance = build_medicine_safety_guidance(notes, cumulative_symptoms, care_plan)
        if care_plan:
            home_care = format_assistant_items(care_plan["home_remedies"]) or "rest, hydration, and monitoring symptoms"
            medicine_note = format_assistant_items(care_plan["medicines"], 2) or "use medicines only with medical advice"
            lines.append(f"- Home care to consider: {home_care}.")
            if medicine_guidance:
                lines.append(medicine_guidance)
            else:
                lines.append(f"- Medicine safety: {medicine_note}.")
        elif remedy_lines:
            lines.append(f"- Symptom-wise tips: {' | '.join(remedy_lines)}.")
            if medicine_guidance:
                lines.append(medicine_guidance)
        else:
            lines.append("- Share symptoms first so I can suggest safer home-care guidance.")
    elif focus == "warnings":
        lines.append("Warning signs:")
        if care_plan:
            warning_text = format_assistant_items(care_plan["warning_signs"]) or "symptoms becoming severe, sudden, or rapidly worse"
            lines.append(f"- Watch for: {warning_text}.")
        else:
            lines.append("- Seek urgent care for breathing trouble, chest pain, fainting, confusion, severe weakness, or fast worsening symptoms.")
    elif focus == "specialist":
        lines.append("Specialist guidance:")
        if care_plan:
            lines.append(f"- Consider: {care_plan['specialist']}. {care_plan['doctor_urgency']}")
        else:
            lines.append("- I need symptoms or a prediction before I can suggest the most relevant specialist.")
    else:
        lines.append("What to do now:")
        if care_plan:
            home_care = format_assistant_items(care_plan["home_remedies"], 3)
            lines.append(f"- Doctor consultation: {care_plan['doctor_required']}. {care_plan['doctor_urgency']}")
            if home_care:
                lines.append(f"- Basic care: {home_care}.")
            lines.append(f"- Tests: blood report {care_plan['blood_report_required']}; lab tests {care_plan['lab_test_required']}.")
        else:
            lines.append("- Add duration, severity, and one or two more symptoms for a better screening result.")

    if remedy_lines and focus != "remedies":
        lines.append(f"Symptom-wise tips: {' | '.join(remedy_lines)}.")

    if urgent_note:
        lines.append(f"Safety alert: {urgent_note}")
    elif care_plan:
        warning_text = format_assistant_items(care_plan["warning_signs"], 3)
        if warning_text:
            lines.append(f"Warning signs: {warning_text}.")

    lines.append("Next step in the app:")
    if cumulative_symptoms:
        lines.append("- Say 'predict now' or press Predict to run the saved symptoms in the predictor.")
    else:
        lines.append("- Type symptoms like: fever, cough, headache, vomiting, stomach pain, chest pain, or breathlessness.")

    return "\n".join(lines)


def normalize_gemini_error_message(error_payload, fallback_message):
    if not isinstance(error_payload, dict):
        return fallback_message

    error_code = str(error_payload.get("code", "")).strip().lower()
    error_type = str(error_payload.get("type", "")).strip().lower()
    error_status = str(error_payload.get("status", "")).strip().lower()
    error_message = str(error_payload.get("message", "")).strip()
    combined_text = " ".join(part for part in [error_code, error_type, error_status, error_message.lower()] if part)

    if "quota" in combined_text or "billing" in combined_text or "resource_exhausted" in combined_text:
        return (
            "Your Gemini API key reached a quota or billing limit, so the chatbot switched back to built-in replies. "
            "Check Gemini API quota or billing in Google AI Studio, then try again."
        )

    if "api_key_invalid" in combined_text or "invalid api key" in combined_text or "permission_denied" in combined_text:
        return "The Gemini API key is invalid or expired. Generate a new key and update GEMINI_API_KEY."

    if "rate_limit" in combined_text or "too many requests" in combined_text:
        return "Gemini rate limits were hit for this key. Wait a moment and try the chatbot again."

    return error_message or fallback_message


def extract_api_error_message(payload, fallback_message):
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return normalize_gemini_error_message(error, fallback_message)
        if isinstance(error, str) and error.strip():
            return error.strip()
    return fallback_message


def build_live_ai_context(cumulative_symptoms, preview, notes=None):
    symptom_summary = format_symptom_summary(cumulative_symptoms)
    note_summary = build_clinical_notes_summary(notes or {})
    preview_summary = "No screening preview available yet."
    if preview:
        preview_summary = (
            f"Current app screening preview: {preview['predicted_disease']}."
        )

    return (
        "App context for this conversation:\n"
        f"- Detected symptoms in the app: {symptom_summary}\n"
        f"- Patient details collected: {note_summary}\n"
        f"- {preview_summary}\n"
        "- This app provides screening support only and should not replace a licensed clinician.\n"
        "- Medicine safety rule: consider the named medicine history, but do not prescribe new drugs, give exact doses, or tell the patient to change prescribed medicine without clinician review.\n"
        "- For prescribed antibiotics/chronic medicines, advise doctor/pharmacist guidance before stopping; for severe allergic reaction after a new medicine, advise urgent care and no further dose until a clinician advises.\n"
        "- Differential safety rule: if symptoms are broad or the screening preview is weak, ask discriminating questions before naming a likely disease.\n"
        "- If the user asks a vague question, answer with useful guidance and ask exactly one follow-up question.\n"
        "- Do not show percentages, confidence scores, or internal model details.\n"
        "- Prefer complete, practical replies over one-sentence replies."
    )


def build_gemini_contents(chat_history):
    contents = []

    for message in chat_history[-ASSISTANT_MAX_HISTORY:]:
        role = str(message.get("role", "")).strip().lower()
        if role not in {"assistant", "user"}:
            continue
        content = strip_live_ai_noise(message.get("content", ""))
        if content:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Hello"}]})

    return contents


def extract_gemini_response_text(payload):
    if not isinstance(payload, dict):
        return ""

    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        collected_parts = []
        for part in parts:
            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    collected_parts.append(text_value.strip())
        if collected_parts:
            return "\n".join(collected_parts).strip()

    return ""


def call_live_chat_model(chat_history, cumulative_symptoms, preview, notes=None, local_reply=""):
    api_key = get_configured_gemini_api_key()

    if not api_key:
        return "", "Gemini mode is not configured yet. Set GEMINI_API_KEY in `.streamlit/secrets.toml` to enable live replies."

    live_context = build_live_ai_context(cumulative_symptoms, preview, notes)
    if local_reply:
        live_context = (
            f"{live_context}\n\n"
            "App draft reply to merge with Gemini useful detail:\n"
            f"{strip_live_ai_noise(local_reply)[:1600]}\n"
            "- Keep the same final follow-up question as the app draft if it asks one. "
            "Do not ask a different final follow-up question."
        )

    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        f"{ASSISTANT_SYSTEM_PROMPT}\n\n"
                        f"{live_context}"
                    )
                }
            ]
        },
        "contents": build_gemini_contents(chat_history),
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 1600,
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    request = Request(
        ASSISTANT_DEFAULT_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=45) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(raw_error)
        except json.JSONDecodeError:
            error_payload = {"error": raw_error or f"HTTP {exc.code}"}
        return "", extract_api_error_message(
            error_payload,
            f"Gemini request failed with HTTP {exc.code}.",
        )
    except URLError:
        return "", "Gemini request failed because the API endpoint could not be reached."
    except Exception as exc:
        return "", f"Gemini reply failed: {exc}"

    assistant_text = strip_live_ai_noise(extract_gemini_response_text(response_payload))
    if assistant_text:
        return assistant_text, ""

    return "", extract_api_error_message(
        response_payload,
        "The Gemini model responded, but no assistant text was returned.",
    )


def infer_visible_question_type(reply_text):
    normalized_reply = normalize_text_for_matching(reply_text)
    if "?" not in str(reply_text) and not any(term in normalized_reply for term in ("tell me", "could you", "can you")):
        return ""

    if any(
        term in normalized_reply
        for term in (
            "how long",
            "how many days",
            "how many hours",
            "since when",
            "from when",
            "when did",
            "when first",
            "first noticed",
            "first notice",
            "approximately when",
            "was it today",
            "was it yesterday",
        )
    ):
        return "duration"

    if any(
        term in normalized_reply
        for term in (
            "getting better",
            "getting worse",
            "staying the same",
            "stay the same",
            "same or worse",
            "improving",
            "worsening",
            "progression",
            "changed over time",
        )
    ):
        return "progression"

    if any(term in normalized_reply for term in ("mild moderate or severe", "mild moderate severe", "how severe", "severity")):
        return "severity"

    if any(term in normalized_reply for term in ("temperature", "how much was it", "checked it")):
        return "temperature"

    if any(term in normalized_reply for term in ("taken any medicine", "which medicine", "medicine for this", "paracetamol", "tablet")):
        return "medicine"

    if all(term in normalized_reply for term in ("asthma", "diabetes")) and any(term in normalized_reply for term in ("pregnancy", "heart", "high bp", "low immunity")):
        return "risk_context"

    return ""


def sync_pending_question_with_visible_reply(reply_text, cumulative_symptoms, notes=None):
    visible_question_type = infer_visible_question_type(reply_text)
    if not visible_question_type:
        return

    current_pending = dict(st.session_state.get("assistant_pending_question", {}) or {})
    if current_pending.get("type") == visible_question_type:
        return

    current_key = current_pending.get("key")
    if current_key:
        asked_keys = list(st.session_state.get("assistant_asked_question_keys", []) or [])
        st.session_state.assistant_asked_question_keys = [key for key in asked_keys if key != current_key]

    profile = get_symptom_question_profile(cumulative_symptoms or [])
    profile_name = profile.get("name", "current symptoms")
    remember_assistant_question(f"visible:{visible_question_type}:{profile_name}", visible_question_type)


def extract_last_question(text):
    matches = re.findall(r"([^?]{4,}\?)", str(text or ""), flags=re.MULTILINE)
    if not matches:
        return ""
    return re.sub(r"\s+", " ", matches[-1]).strip()


def get_missing_local_guidance_lines(live_reply, fallback_reply, limit=6):
    live_normalized = normalize_text_for_matching(live_reply)
    missing_lines = []
    for raw_line in str(fallback_reply or "").splitlines():
        line = raw_line.strip()
        if not line or "?" in line:
            continue
        if line.endswith(":"):
            continue
        normalized_line = normalize_text_for_matching(line)
        if not normalized_line or len(normalized_line) < 12:
            continue
        if normalized_line in live_normalized:
            continue
        missing_lines.append(line)
        if len(missing_lines) >= limit:
            break
    return missing_lines


def merge_live_and_local_chat_reply(live_reply, fallback_reply):
    live_reply = strip_live_ai_noise(live_reply)
    fallback_reply = strip_live_ai_noise(fallback_reply)
    if not live_reply:
        return fallback_reply
    if not fallback_reply:
        return live_reply

    merged_reply = live_reply
    if reply_needs_more_detail(live_reply):
        missing_lines = get_missing_local_guidance_lines(live_reply, fallback_reply)
        if missing_lines:
            merged_reply = (
                f"{merged_reply}\n\n"
                "Also keep in mind:\n"
                + "\n".join(missing_lines)
            )

    if "?" not in merged_reply:
        local_question = extract_last_question(fallback_reply)
        if local_question:
            merged_reply = f"{merged_reply}\n\n{local_question}"

    return merged_reply.strip()


def get_live_or_fallback_chat_reply(chat_history, fallback_reply, cumulative_symptoms, preview, notes=None, action=None):
    if action in {"load", "predict"}:
        return fallback_reply

    live_reply, _ = call_live_chat_model(chat_history, cumulative_symptoms, preview, notes, fallback_reply)
    if live_reply:
        merged_reply = merge_live_and_local_chat_reply(live_reply, fallback_reply)
        sync_pending_question_with_visible_reply(merged_reply, cumulative_symptoms, notes)
        return merged_reply
    return fallback_reply


def build_symptom_assistant_reply(user_message, detected_now, cumulative_symptoms, model, label_encoder, feature_names):
    normalized_message = normalize_text_for_matching(user_message)
    clinical_notes = st.session_state.get("assistant_clinical_notes", {}) or {}
    preview = get_assistant_reference_prediction(model, label_encoder, feature_names, cumulative_symptoms)
    care_plan = None
    remedy_lines = get_symptom_specific_remedies(cumulative_symptoms)
    if preview:
        care_plan = get_disease_care_plan(preview["predicted_disease"], preview["prediction_percent"])

    snapshot = build_screening_snapshot(preview, care_plan, cumulative_symptoms)
    urgent_note = build_urgent_symptom_note(cumulative_symptoms)

    if message_has_any_phrase(normalized_message, ASSISTANT_THANKS_TERMS):
        return (
            "You are welcome.\n"
            "- I can still help with prediction, lab-test guidance, home care, warning signs, or doctor booking.\n"
            "- If symptoms are severe, sudden, or worsening, please seek medical care instead of waiting for the app.\n"
            "- To continue, ask a question like: what tests are needed, which doctor should I see, or predict now."
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_GREETING_TERMS) and not detected_now:
        return (
            "Hi, tell me what is bothering you right now. "
            "You can write it simply, like blocked nose, fever, cough, stomach pain, or headache."
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_HELP_TERMS):
        return (
            "Tell me your main symptom first. I will ask one short question at a time, "
            "then I can run the screening when I have enough detail."
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_AI_TERMS) and not detected_now:
        return (
            "The app has Gemini key support through `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.streamlit/secrets.toml`. "
            "Symptom detection and prediction still run inside this app, so the key is not required for detecting symptoms."
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_BOOKING_TERMS) and not detected_now:
        return build_structured_assistant_reply(
            detected_now,
            cumulative_symptoms,
            preview,
            care_plan,
            remedy_lines,
            urgent_note,
            focus="booking",
            intro="Here is how to choose a doctor or hospital from the app.",
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_TEST_TERMS) and not detected_now:
        return build_structured_assistant_reply(
            detected_now,
            cumulative_symptoms,
            preview,
            care_plan,
            remedy_lines,
            urgent_note,
            focus="tests",
            intro="Here is the test/report guidance based on the current symptom set.",
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_MEDICINE_ADVICE_TERMS):
        return build_medicine_advice_reply(
            detected_now,
            cumulative_symptoms,
            preview,
            care_plan,
            clinical_notes,
            urgent_note,
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_REMEDY_TERMS) and not detected_now:
        return build_structured_assistant_reply(
            detected_now,
            cumulative_symptoms,
            preview,
            care_plan,
            remedy_lines,
            urgent_note,
            focus="remedies",
            intro="Here is safer home-care guidance for the current symptom set.",
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_WARNING_TERMS) and not detected_now:
        return build_structured_assistant_reply(
            detected_now,
            cumulative_symptoms,
            preview,
            care_plan,
            remedy_lines,
            urgent_note,
            focus="warnings",
            intro="Here are the warning signs to take seriously.",
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_SPECIALIST_TERMS) and not detected_now:
        return build_structured_assistant_reply(
            detected_now,
            cumulative_symptoms,
            preview,
            care_plan,
            remedy_lines,
            urgent_note,
            focus="specialist",
            intro="Here is the specialist guidance I can provide right now.",
        )

    if message_has_any_phrase(normalized_message, ASSISTANT_LOAD_TRIGGERS) and cumulative_symptoms:
        return (
            "I loaded this symptom set for prediction.\n"
            f"- Symptoms: {format_symptom_summary(cumulative_symptoms)}.\n"
            "- Review the selected symptoms in the predictor before submitting.\n"
            "- If anything is missing, type the extra symptom before running prediction.\n"
            "- After prediction, I can explain tests, home care, warning signs, specialist guidance, and doctor booking."
        )

    if (
        message_has_any_phrase(normalized_message, ASSISTANT_PREDICT_TERMS)
        or message_has_any_phrase(normalized_message, ASSISTANT_DISEASE_TERMS)
    ) and preview and not detected_now:
        if should_guard_nonspecific_prediction(preview, cumulative_symptoms, clinical_notes):
            return build_nonspecific_prediction_guard_reply(preview, cumulative_symptoms, clinical_notes)
        if not is_ready_for_assistant_prediction(cumulative_symptoms, clinical_notes, preview):
            return build_doctor_style_interview_reply(
                user_message,
                detected_now,
                cumulative_symptoms,
                clinical_notes,
                preview,
                care_plan,
                urgent_note,
            )
        if not has_confident_prediction_preview(preview):
            return build_patient_conversation_reply(
                user_message,
                detected_now,
                cumulative_symptoms,
                clinical_notes,
                preview,
                care_plan,
                urgent_note,
                allow_predict=False,
            )
        return build_human_prediction_reply(preview, care_plan, cumulative_symptoms, urgent_note)

    if not detected_now:
        return build_doctor_style_interview_reply(
            user_message,
            detected_now,
            cumulative_symptoms,
            clinical_notes,
            preview,
            care_plan,
            urgent_note,
        )

    return build_doctor_style_interview_reply(
        user_message,
        detected_now,
        cumulative_symptoms,
        clinical_notes,
        preview,
        care_plan,
        urgent_note,
    )


def process_symptom_assistant_message(message_text, model, label_encoder, feature_names):
    cleaned_message = str(message_text).strip()
    if not cleaned_message:
        return "Please type at least one message for the chatbot."

    information_symptoms = detect_symptom_information_request(cleaned_message, feature_names)
    if information_symptoms:
        st.session_state.symptom_chat_history.append({"role": "user", "content": cleaned_message})
        clinical_notes = st.session_state.get("assistant_clinical_notes", {}) or {}
        current_symptoms = st.session_state.get("assistant_detected_symptoms", []) or []
        preview = get_assistant_reference_prediction(model, label_encoder, feature_names, current_symptoms)
        fallback_reply = build_symptom_information_reply(information_symptoms)
        assistant_reply = get_live_or_fallback_chat_reply(
            st.session_state.symptom_chat_history,
            fallback_reply,
            current_symptoms,
            preview,
            clinical_notes,
        )
        st.session_state.symptom_chat_history.append(
            {"role": "assistant", "content": assistant_reply}
        )
        return ""

    pending_result = resolve_pending_question_answer(cleaned_message, feature_names)
    if pending_result.get("error"):
        return pending_result["error"]
    detected_now = detect_symptoms_from_text(cleaned_message, feature_names)
    detected_now = sorted(set(detected_now).union(pending_result.get("detected_symptoms", [])))
    note_update = merge_note_updates(
        extract_clinical_note_update(cleaned_message, feature_names),
        pending_result.get("note_update", {}),
    )
    clinical_notes = update_assistant_clinical_notes(note_update)
    denied_symptoms = set(clinical_notes.get("denied_symptoms", []))
    detected_now = sorted(symptom for symptom in detected_now if symptom not in denied_symptoms)
    merged_symptoms = sorted((set(st.session_state.assistant_detected_symptoms).union(detected_now)) - denied_symptoms)
    st.session_state.assistant_detected_symptoms = merged_symptoms
    st.session_state.symptom_chat_history.append({"role": "user", "content": cleaned_message})

    normalized_message = normalize_text_for_matching(cleaned_message)
    action = pending_result.get("action") or detect_assistant_action(normalized_message, merged_symptoms)
    preview = get_assistant_reference_prediction(model, label_encoder, feature_names, merged_symptoms)

    fallback_reply = build_symptom_assistant_reply(
        cleaned_message,
        detected_now,
        merged_symptoms,
        model,
        label_encoder,
        feature_names,
    )
    fallback_reply = strip_live_ai_noise(fallback_reply)
    assistant_reply = fallback_reply

    if action == "load":
        st.session_state.prediction_symptoms = merged_symptoms
        assistant_reply += "\n\nI loaded the detected symptoms into the predictor for you."
    elif action == "predict":
        preview_for_prediction = get_prediction_preview(model, label_encoder, feature_names, merged_symptoms)
        if should_guard_nonspecific_prediction(preview_for_prediction, merged_symptoms, clinical_notes):
            assistant_reply = build_nonspecific_prediction_guard_reply(
                preview_for_prediction,
                merged_symptoms,
                clinical_notes,
            )
            st.session_state.assistant_trigger_predict = False
            st.session_state.prediction_symptoms = []
            st.session_state.prediction_context = {
                "guarded": True,
                "selected_symptoms": list(merged_symptoms),
                "predicted_disease": preview_for_prediction["predicted_disease"] if preview_for_prediction else "",
                "confidence": float(preview_for_prediction.get("prediction_percent", 0.0)) if preview_for_prediction else 0.0,
                "guard_message": assistant_reply,
            }
            st.session_state.symptom_chat_history.append({"role": "assistant", "content": assistant_reply})
            return ""
        if is_ready_for_assistant_prediction(merged_symptoms, clinical_notes, preview_for_prediction):
            care_plan_for_prediction = None
            if preview_for_prediction:
                care_plan_for_prediction = get_disease_care_plan(
                    preview_for_prediction["predicted_disease"],
                    preview_for_prediction["prediction_percent"],
                )
            patient_finished_symptom_list = patient_declined_more_symptoms(clinical_notes)
            if has_confident_prediction_preview(preview_for_prediction) or patient_finished_symptom_list:
                assistant_reply = build_human_prediction_reply(
                    preview_for_prediction,
                    care_plan_for_prediction,
                    merged_symptoms,
                    build_urgent_symptom_note(merged_symptoms),
                )
                if patient_finished_symptom_list and not has_confident_prediction_preview(preview_for_prediction):
                    assistant_reply = (
                        "I can run the screening with the details you gave, but these symptoms can overlap across many conditions.\n\n"
                        f"{assistant_reply}"
                    )
                st.session_state.prediction_symptoms = merged_symptoms
                st.session_state.assistant_trigger_predict = True
                assistant_reply += "\n\nI also opened the full prediction result below."
            else:
                assistant_reply = build_patient_conversation_reply(
                    cleaned_message,
                    [],
                    merged_symptoms,
                    clinical_notes,
                    preview_for_prediction,
                    care_plan_for_prediction,
                    build_urgent_symptom_note(merged_symptoms),
                    allow_predict=False,
                )
                assistant_reply = (
                    "I do not want to guess from too little information yet. "
                    f"{assistant_reply}"
                )
        else:
            next_question = get_next_patient_question(
                merged_symptoms,
                clinical_notes,
                allow_predict=False,
                preview=preview_for_prediction,
            )
            if next_question:
                assistant_reply = (
                    "I need one more detail before running the screening.\n\n"
                    f"{next_question}"
                )
            else:
                assistant_reply = (
                    "I could not run the screening from the current details yet. "
                    "Please add one more clear symptom or duration, then type 'predict now' again."
                )
    else:
        assistant_reply = get_live_or_fallback_chat_reply(
            st.session_state.symptom_chat_history,
            fallback_reply,
            merged_symptoms,
            preview,
            clinical_notes,
            action=action,
        )

    st.session_state.symptom_chat_history.append({"role": "assistant", "content": assistant_reply})
    return ""


def format_chat_message_content(content):
    safe_content = html.escape(strip_live_ai_noise(content))
    safe_content = re.sub(r"\*{3,}", "", safe_content)
    safe_content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe_content)
    safe_content = re.sub(r"__(.+?)__", r"<strong>\1</strong>", safe_content)
    safe_content = re.sub(r"(?m)^\s*\*\s+", "- ", safe_content)
    safe_content = safe_content.replace("*", "")

    important_labels = (
        "Quick answer:",
        "Additional app guidance:",
        "What I understood:",
        "Screening view:",
        "What it may mean in this screening app:",
        "What to do now:",
        "Doctor/tests guidance:",
        "Doctor booking guidance:",
        "Tests and reports:",
        "Lab test guidance:",
        "Care guidance:",
        "Medicine safety:",
        "Warning signs:",
        "Specialist guidance:",
        "Safety alert:",
        "Symptom-wise tips:",
        "Next step in the app:",
        "Chatbot mode information:",
    )
    for label in important_labels:
        escaped_label = html.escape(label)
        safe_content = safe_content.replace(escaped_label, f"<strong>{escaped_label}</strong>")

    return safe_content.replace("\n", "<br>")


def render_assistant_chat_history(messages):
    visible_messages = messages[-ASSISTANT_MAX_HISTORY:]
    if not visible_messages:
        st.markdown(
            (
                '<div class="assistant-chat-history">'
                '<div class="assistant-chat-empty">'
                '<div class="assistant-chat-empty-title">Tell me your symptoms</div>'
                '<div class="assistant-chat-empty-copy">Write naturally, for example: fever and cough since yesterday, no breathing problem.</div>'
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    message_items = []
    for message in visible_messages:
        role = str(message.get("role", "assistant")).strip().lower()
        is_user = role == "user"
        role_class = "user" if is_user else "assistant"
        label = "You" if is_user else "Assistant"
        avatar_label = "YOU" if is_user else "AI"
        cleaned_content = strip_live_ai_noise(message.get("content", ""))
        if not cleaned_content:
            continue
        content = format_chat_message_content(cleaned_content)
        message_items.append(
            f'<div class="assistant-chat-row assistant-chat-row--{role_class}">'
            f'<div class="assistant-chat-avatar assistant-chat-avatar--{role_class}">{avatar_label}</div>'
            f'<div class="assistant-chat-bubble assistant-chat-bubble--{role_class}">'
            f'<div class="assistant-chat-label">{label}</div>'
            f'<div class="assistant-chat-text">{content}</div>'
            "</div>"
            "</div>"
        )

    if not message_items:
        st.markdown(
            (
                '<div class="assistant-chat-history">'
                '<div class="assistant-chat-empty">'
                '<div class="assistant-chat-empty-title">Tell me your symptoms</div>'
                '<div class="assistant-chat-empty-copy">Type a fresh symptom message and I will answer with app guidance.</div>'
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div class='assistant-chat-history'>{''.join(message_items)}</div>",
        unsafe_allow_html=True,
    )


def build_assistant_state_html(detected_symptoms, preview, live_ai_ready):
    if not detected_symptoms:
        return ""
    symptom_chips = []
    for symptom in detected_symptoms[:8]:
        symptom_chips.append(
            f'<span class="assistant-symptom-chip">{html.escape(format_symptom_label(symptom))}</span>'
        )
    if len(detected_symptoms) > 8:
        symptom_chips.append(
            f'<span class="assistant-symptom-chip assistant-symptom-chip--more">+{len(detected_symptoms) - 8} more</span>'
        )

    return f"""
        <div class="assistant-symptom-rack">
            {''.join(symptom_chips)}
        </div>
    """


def reply_needs_more_detail(reply_text):
    cleaned_reply = re.sub(r"\s+", " ", str(reply_text or "")).strip()
    if not cleaned_reply:
        return True
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", cleaned_reply))
    word_count = len(cleaned_reply.split())
    return word_count < 70 or sentence_count < 4


def render_symptom_assistant(model, label_encoder, feature_names):
    detected_symptoms = st.session_state.assistant_detected_symptoms
    clinical_notes = st.session_state.get("assistant_clinical_notes", {}) or {}
    preview = get_prediction_preview(model, label_encoder, feature_names, detected_symptoms)
    assistant_image_b64 = get_image_base64(ASSISTANT_IMAGE_PATH)
    assistant_image_html = ""
    if assistant_image_b64:
        assistant_image_html = (
            f"<img src='data:image/png;base64,{assistant_image_b64}' "
            "class='assistant-bot-image' alt='Health chatbot robot' />"
        )
    else:
        assistant_image_html = "<div class='assistant-launcher-logo'>AI</div>"

    assistant_status_class = "assistant-status-dot assistant-status-dot--live"
    assistant_status_copy = "One question at a time"

    if not st.session_state.assistant_panel_open:
        st.markdown(
            f"""
            <div class="assistant-launcher-card assistant-launcher-card--interactive">
                <div class="assistant-launcher-glow"></div>
                <div class="assistant-launcher-visual">
                    {assistant_image_html}
                </div>
                <div class="assistant-launcher-kicker-row">
                    <span class="{assistant_status_class}"></span>
                    <span class="assistant-launcher-kicker">{html.escape(assistant_status_copy)}</span>
                </div>
                <div class="assistant-launcher-title">AI Symptom Chat</div>
                <p class="assistant-launcher-copy">Describe symptoms in normal words. The assistant asks one short question at a time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Chat", key="open_assistant_panel", type="primary"):
            st.session_state.assistant_panel_open = True
            st.rerun()
        return

    with st.container(border=True):
        header_col, close_col = st.columns([4.1, 1.7], gap="small")
        with header_col:
            st.markdown(
                f"""
                <div class="assistant-panel-header">
                    <div class="assistant-mini-brand">
                        <div class="assistant-panel-avatar">
                            {assistant_image_html}
                        </div>
                        <div>
                            <div class="assistant-panel-kicker">Patient Assistant</div>
                            <div class="assistant-shell-title">AI Symptom Chat</div>
                            <p class="assistant-shell-copy">Tell symptoms in normal words. I will ask one short question at a time.</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with close_col:
            if st.button("Close", key="close_assistant_panel"):
                st.session_state.assistant_panel_open = False
                st.rerun()

        global_warning = None

        render_assistant_chat_history(st.session_state.symptom_chat_history)
        if detected_symptoms:
            st.caption(f"Saved symptoms: {format_symptom_summary(detected_symptoms)}")

        with st.form("symptom_assistant_form", clear_on_submit=True):
            input_col, send_col = st.columns([3.8, 1.2], gap="small")
            with input_col:
                assistant_input = st.text_input(
                    "Describe symptoms",
                    placeholder="Example: I have blocked nose since yesterday, no fever, sneezing yes",
                    label_visibility="collapsed",
                )
            with send_col:
                send_to_assistant = st.form_submit_button("Send", type="primary", use_container_width=True)

        if send_to_assistant:
            global_warning = process_symptom_assistant_message(assistant_input, model, label_encoder, feature_names)
            if not global_warning:
                st.rerun()

        if global_warning:
            st.error(global_warning, icon="⚠️")

        if detected_symptoms:
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                if st.button("Use Symptoms", key="use_assistant_symptoms", type="primary"):
                    st.session_state.prediction_symptoms = detected_symptoms
                    st.rerun()
            with action_col2:
                if st.button("Predict", key="predict_from_assistant"):
                    if is_ready_for_assistant_prediction(detected_symptoms, clinical_notes, preview) and has_confident_prediction_preview(preview):
                        st.session_state.prediction_symptoms = detected_symptoms
                        st.session_state.assistant_trigger_predict = True
                    else:
                        preview_care_plan = None
                        if preview:
                            preview_care_plan = get_disease_care_plan(
                                preview["predicted_disease"],
                                preview["prediction_percent"],
                            )
                        assistant_reply = build_patient_conversation_reply(
                            "",
                            [],
                            detected_symptoms,
                            clinical_notes,
                            preview,
                            preview_care_plan,
                            build_urgent_symptom_note(detected_symptoms),
                            allow_predict=False,
                        )
                        st.session_state.symptom_chat_history.append({"role": "assistant", "content": assistant_reply})
                    st.rerun()
            with action_col3:
                if st.button("Clear Chat", key="clear_assistant_button"):
                    reset_symptom_assistant_state()
                    st.rerun()
        else:
            st.caption("Type a symptom message above to begin.")
            if st.session_state.symptom_chat_history:
                if st.button("Clear Chat", key="clear_assistant_button"):
                    reset_symptom_assistant_state()
                    st.rerun()


PAGE_META = {
    "Home": {
        "eyebrow": "Clinical Screening Platform",
        "title": "From symptoms to next-step care in one workspace",
        "description": "Explore disease prediction, doctor consultation guidance, diagnostics, and follow-up actions through a cleaner workflow.",
        "chips": ["Prediction flow", "Care guidance", "Doctor booking"],
    },
    "Dashboard": {
        "eyebrow": "Interactive Dashboard",
        "title": "Prediction analytics with real-world next steps",
        "description": "Select symptoms, review ranked disease predictions, understand severity, and book the right doctor without leaving the dashboard.",
        "chips": ["Smart charts", "Lab advice", "Appointment booking"],
    },
    "Appointments": {
        "eyebrow": "Consultation Center",
        "title": "Track and manage doctor bookings",
        "description": "Review active consultations, see historical bookings, and cancel appointments from a single place.",
        "chips": ["Active bookings", "Status tracking", "Quick cancellation"],
    },
    "About": {
        "eyebrow": "Project Overview",
        "title": "Educational screening support for symptom-first care",
        "description": "The app is designed for symptom-based screening support and should always be paired with professional medical judgment.",
        "chips": ["ML powered", "Patient friendly", "Doctor first"],
    },
    "Register": {
        "eyebrow": "User Onboarding",
        "title": "Create a secure account to access the full dashboard",
        "description": "Sign up to save your login, move into the dashboard, and manage prediction-driven appointments.",
        "chips": ["Secure access", "Simple sign-up", "Fast start"],
    },
    "Login": {
        "eyebrow": "Secure Access",
        "title": "Sign in and continue your prediction workflow",
        "description": "Jump back into the dashboard, continue screening, and manage doctor consultations with your saved account.",
        "chips": ["Password protection", "Fast login", "Recovery option"],
    },
    "Contact Us": {
        "eyebrow": "Support Desk",
        "title": "Send questions, suggestions, or project feedback",
        "description": "Use the contact form to share improvement requests, report issues, or ask for help with the system.",
        "chips": ["Feedback", "Support", "Project communication"],
    },
}


def inject_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Manrope:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

        :root {
            --bg-1: #f7efe4;
            --bg-2: #eef7f7;
            --ink: #16212d;
            --muted: #5f6b7a;
            --line: rgba(15, 118, 110, 0.14);
            --panel: rgba(255, 255, 255, 0.82);
            --panel-strong: rgba(255, 255, 255, 0.94);
            --teal: #0f766e;
            --teal-soft: #d7f1ee;
            --orange: #ea580c;
            --shadow: 0 20px 45px rgba(22, 33, 45, 0.08);
            --shadow-soft: 0 12px 28px rgba(15, 118, 110, 0.10);
            --font-ui: 'Manrope', sans-serif;
            --font-display: 'Sora', sans-serif;
            --font-data: 'IBM Plex Mono', monospace;
            --desktop-canvas-width: 1180px;
            --mobile-layout-width: 1024px;
            --sidebar-width: 300px;
        }

        html, body, [class*="css"] {
            font-family: var(--font-ui);
            font-size: 16px;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        html {
            overflow-y: scroll;
            scrollbar-gutter: stable both-edges;
            scrollbar-color: #5a63d8 rgba(214, 232, 255, 0.82);
            scrollbar-width: thin;
        }

        body,
        .stApp,
        [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
            scrollbar-gutter: stable both-edges;
        }

        ::-webkit-scrollbar {
            width: 14px;
            height: 14px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(214, 232, 255, 0.82);
            border-radius: 999px;
            border: 3px solid rgba(247, 249, 255, 0.98);
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #6f78ee, #3f46b6);
            border-radius: 999px;
            border: 3px solid rgba(247, 249, 255, 0.98);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.42);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #555fe4, #2f359f);
        }

        ::-webkit-scrollbar-corner {
            background: rgba(214, 232, 255, 0.82);
        }

        .stApp::after {
            content: "";
            position: fixed;
            top: 18vh;
            right: 4px;
            width: 6px;
            height: 64vh;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(111, 120, 238, 0.42), rgba(63, 70, 182, 0.64));
            box-shadow: 0 0 0 4px rgba(214, 232, 255, 0.46);
            pointer-events: none;
            z-index: 999;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 0% 0%, rgba(15, 118, 110, 0.16), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(234, 88, 12, 0.14), transparent 24%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 42%, #f9fbfc 100%);
            color: var(--ink);
        }

        .stApp {
            color: var(--ink);
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 58px !important;
            min-height: 58px !important;
            visibility: visible !important;
            pointer-events: none !important;
        }

        #MainMenu,
        footer,
        [data-testid="stMainMenu"],
        [data-testid="stElementToolbar"],
        [data-testid="stElementToolbarButton"],
        [data-testid="StyledFullScreenButton"],
        [data-testid="stDeployButton"],
        [data-testid="stStatusWidget"],
        .stDeployButton,
        button[title="View fullscreen"],
        button[aria-label="View fullscreen"],
        button[title="More options"],
        button[aria-label="More options"],
        button[title="Deploy"],
        button[aria-label="Deploy"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        header[data-testid="stHeader"] [data-testid="stHeaderActionElements"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        [data-testid="stToolbar"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            background: transparent !important;
        }

        [data-testid="stSidebarHeader"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            align-items: center !important;
            justify-content: flex-end !important;
            min-height: 58px !important;
            pointer-events: auto !important;
            position: sticky !important;
            top: 0 !important;
            z-index: 10001 !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1360px;
        }

        [data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"],
        button[data-testid="stExpandSidebarButton"],
        [aria-label="Open sidebar"],
        [aria-label="Close sidebar"],
        [aria-label="Collapse sidebar"],
        [title="Open sidebar"],
        [title="Close sidebar"],
        [title="Collapse sidebar"] {
            display: inline-flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            align-items: center !important;
            justify-content: center !important;
            width: 44px !important;
            height: 44px !important;
            min-width: 44px !important;
            min-height: 44px !important;
            border-radius: 999px !important;
            background: linear-gradient(145deg, #ffffff, #dbe8ff) !important;
            border: 2px solid rgba(90, 99, 216, 0.36) !important;
            color: #2d3192 !important;
            box-shadow: 0 12px 26px rgba(64, 74, 130, 0.22), 0 0 0 5px rgba(90, 99, 216, 0.12) !important;
            position: fixed !important;
            top: 0.75rem !important;
            left: 0.75rem !important;
            z-index: 10000 !important;
            animation: sidebarArrowPulse 1.8s ease-in-out infinite;
        }

        [data-testid="stSidebarCollapseButton"],
        button[data-testid="stSidebarCollapseButton"] {
            left: calc(var(--sidebar-width) - 4.9rem) !important;
        }

        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stExpandSidebarButton"] *,
        [data-testid="stSidebarCollapsedControl"] *,
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="stSidebarCollapsedControl"] svg {
            visibility: visible !important;
            opacity: 1 !important;
            color: #2d3192 !important;
            fill: currentColor !important;
            pointer-events: auto !important;
        }

        [data-testid="stSidebarCollapsedControl"]:hover,
        button[data-testid="stSidebarCollapsedControl"]:hover,
        [data-testid="stSidebarCollapseButton"]:hover,
        button[data-testid="stSidebarCollapseButton"]:hover,
        [data-testid="stExpandSidebarButton"]:hover,
        button[data-testid="stExpandSidebarButton"]:hover,
        [aria-label="Open sidebar"]:hover,
        [aria-label="Close sidebar"]:hover,
        [aria-label="Collapse sidebar"]:hover {
            transform: translateY(-1px) scale(1.04) !important;
            background: linear-gradient(145deg, #eef3ff, #cbdcff) !important;
            box-shadow: 0 16px 30px rgba(64, 74, 130, 0.28), 0 0 0 7px rgba(90, 99, 216, 0.16) !important;
        }

        @keyframes sidebarArrowPulse {
            0%, 100% {
                box-shadow: 0 12px 26px rgba(64, 74, 130, 0.22), 0 0 0 5px rgba(90, 99, 216, 0.12);
            }
            50% {
                box-shadow: 0 16px 32px rgba(64, 74, 130, 0.30), 0 0 0 9px rgba(90, 99, 216, 0.18);
            }
        }

        h1, h2, h3, h4 {
            color: var(--ink);
            letter-spacing: -0.04em;
            line-height: 1.14;
            font-family: var(--font-display);
        }

        p, label, .stCaption, .stMarkdown {
            color: var(--ink);
            font-family: var(--font-ui);
            font-size: 0.99rem;
            line-height: 1.72;
            letter-spacing: -0.01em;
        }

        div[data-testid="stWidgetLabel"],
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            font-family: var(--font-ui) !important;
            font-size: 0.99rem !important;
            line-height: 1.72 !important;
            letter-spacing: -0.01em;
        }

        div[data-testid="stWidgetLabel"] {
            font-size: 0.93rem !important;
            font-weight: 700 !important;
        }

        .page-hero {
            position: relative;
            overflow: hidden;
            padding: 1.6rem 1.7rem;
            border-radius: 30px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(255, 255, 255, 0.72));
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .topbar-shell {
            padding: 1rem 1.15rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.76));
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            margin-bottom: 0.85rem;
        }

        .topbar-title {
            margin: 0;
            font-size: 1.68rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--ink);
            font-family: var(--font-display);
        }

        .topbar-subtitle {
            margin: 0.28rem 0 0 0;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.68;
            font-family: var(--font-ui);
        }

        .topbar-subtitle--dynamic {
            position: relative;
            display: flex;
            align-items: center;
            gap: 0.62rem;
            width: fit-content;
            max-width: 100%;
            font-weight: 650;
        }

        .topbar-subtitle--dynamic::before {
            content: "";
            width: 0.58rem;
            height: 0.58rem;
            flex: 0 0 0.58rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #19b996, #6d5dfc);
            box-shadow: 0 0 0 0 rgba(25, 185, 150, 0.30), 0 0 18px rgba(25, 185, 150, 0.36);
            animation: topbarSignalPulse 2.4s ease-in-out infinite;
        }

        @keyframes topbarSignalPulse {
            0%, 100% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(25, 185, 150, 0.28), 0 0 18px rgba(25, 185, 150, 0.34);
            }
            50% {
                transform: scale(1.15);
                box-shadow: 0 0 0 8px rgba(25, 185, 150, 0), 0 0 24px rgba(109, 93, 252, 0.34);
            }
        }

        .inline-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            margin-top: 0.75rem;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.09);
            border: 1px solid rgba(15, 118, 110, 0.16);
            color: #0f5f59;
            font-size: 0.78rem;
            font-weight: 700;
            font-family: var(--font-data);
        }

        .page-hero::after {
            content: "";
            position: absolute;
            top: -65px;
            right: -35px;
            width: 210px;
            height: 210px;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.22), transparent 66%);
            pointer-events: none;
        }

        .page-eyebrow {
            margin: 0 0 0.55rem 0;
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.76rem;
            font-weight: 800;
            font-family: var(--font-data);
        }

        .page-title {
            margin: 0;
            font-size: 2.45rem;
            line-height: 1.05;
            max-width: 740px;
            font-family: var(--font-display);
        }

        .page-description {
            margin: 0.7rem 0 0 0;
            max-width: 820px;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.75;
        }

        .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-chip {
            padding: 0.48rem 0.82rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 118, 110, 0.16);
            background: rgba(15, 118, 110, 0.08);
            color: #0e5d57;
            font-size: 0.8rem;
            font-weight: 700;
            font-family: var(--font-data);
        }

        .status-card, .info-card {
            min-height: 100%;
            padding: 1.25rem 1.2rem;
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.78));
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        .status-label, .card-kicker {
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            font-family: var(--font-data);
        }

        .status-title, .info-card h4 {
            margin: 0;
            color: var(--ink);
            font-size: 1.22rem;
            line-height: 1.28;
            font-family: var(--font-display);
            font-weight: 700;
        }

        .status-copy, .info-card p {
            margin: 0.7rem 0 0 0;
            color: var(--muted);
            line-height: 1.72;
            font-size: 1rem;
            font-family: var(--font-ui);
        }

        .section-title {
            margin: 0 0 0.35rem 0;
            color: var(--ink);
            font-size: 1.62rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            font-family: var(--font-display);
        }

        .section-copy {
            margin: 0 0 1rem 0;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.74;
            font-family: var(--font-ui);
        }

        .stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 18px;
            border: 1px solid rgba(15, 118, 110, 0.16);
            background: rgba(255, 255, 255, 0.78);
            color: var(--ink);
            font-weight: 800;
            box-shadow: var(--shadow-soft);
            transition: transform 0.14s ease, box-shadow 0.14s ease, border-color 0.14s ease;
            font-family: var(--font-ui);
            letter-spacing: -0.01em;
            font-size: 0.98rem;
        }

        .stFormSubmitButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 18px;
            border: none;
            background: linear-gradient(135deg, var(--teal) 0%, #1e9387 100%);
            color: #ffffff;
            font-weight: 800;
            box-shadow: var(--shadow-soft);
            transition: transform 0.14s ease, box-shadow 0.14s ease, filter 0.14s ease;
            font-family: var(--font-ui);
            letter-spacing: -0.01em;
            font-size: 0.98rem;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(15, 118, 110, 0.30);
            box-shadow: 0 18px 32px rgba(15, 118, 110, 0.12);
        }

        .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 32px rgba(15, 118, 110, 0.14);
            filter: brightness(1.02);
        }

        .stButton > button[kind="primary"] {
            color: #ffffff;
            border: none;
            background: linear-gradient(135deg, var(--teal) 0%, #1e9387 100%);
        }

        .stLinkButton a {
            border-radius: 18px;
            border: 1px solid rgba(15, 118, 110, 0.16);
            background: rgba(255, 255, 255, 0.84);
            color: var(--ink);
            font-weight: 800;
            box-shadow: var(--shadow-soft);
            transition: transform 0.14s ease, box-shadow 0.14s ease;
            font-family: var(--font-ui);
            font-size: 0.97rem;
        }

        .stLinkButton a:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 32px rgba(15, 118, 110, 0.12);
            color: var(--ink);
            border-color: rgba(15, 118, 110, 0.28);
        }

        div[data-testid="stMetric"] {
            padding: 1rem 1.05rem;
            border-radius: 24px;
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stMetricLabel"] {
            font-family: var(--font-data) !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.76rem;
        }

        div[data-testid="stMetricValue"] {
            font-family: var(--font-display) !important;
            letter-spacing: -0.03em;
            font-size: 1.95rem !important;
        }

        div[data-testid="stForm"] {
            padding: 1rem 1rem 0.4rem 1rem;
            border-radius: 28px;
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }

        div[data-testid="stExpander"] {
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        .assistant-shell-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.9rem;
            margin-bottom: 0.8rem;
            padding: 1rem 1.05rem;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(215, 241, 238, 0.72));
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        .assistant-shell-head--compact {
            margin-bottom: 0.4rem;
            padding: 0.9rem 0.95rem;
        }

        .assistant-mini-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .assistant-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.85rem;
            margin-bottom: 0.3rem;
            padding: 0.9rem 0.95rem;
            border-radius: 26px;
            background:
                radial-gradient(circle at 8% 18%, rgba(15, 118, 110, 0.18), transparent 28%),
                linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(215, 241, 238, 0.76));
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        .assistant-panel-avatar {
            flex: 0 0 auto;
            width: 76px;
        }

        .assistant-panel-kicker {
            margin: 0 0 0.26rem 0;
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.69rem;
            font-weight: 800;
            font-family: var(--font-data);
        }

        .assistant-inline-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.55rem;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(15, 118, 110, 0.10);
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            font-family: var(--font-data);
        }

        .assistant-shell-title {
            margin: 0;
            color: var(--ink);
            font-size: 1.12rem;
            font-weight: 700;
            line-height: 1.2;
            font-family: var(--font-display);
        }

        .assistant-shell-copy {
            margin: 0.28rem 0 0 0;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
            font-family: var(--font-ui);
        }

        .assistant-mode-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            padding: 0.46rem 0.72rem;
            border-radius: 999px;
            background: rgba(234, 88, 12, 0.10);
            border: 1px solid rgba(234, 88, 12, 0.16);
            color: #9a3d0a;
            font-size: 0.78rem;
            font-weight: 800;
            font-family: var(--font-data);
        }

        .assistant-status-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: rgba(95, 107, 122, 0.42);
            box-shadow: 0 0 0 6px rgba(95, 107, 122, 0.10);
        }

        .assistant-status-dot--live {
            background: #0f766e;
            box-shadow: 0 0 0 6px rgba(15, 118, 110, 0.16);
        }

        .assistant-launcher-anchor {
            display: block;
            color: inherit;
            text-decoration: none;
        }

        .assistant-launcher-card {
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 0.72rem;
            padding: 1.25rem 1rem 1.05rem 1rem;
            border-radius: 30px;
            background:
                radial-gradient(circle at 18% 16%, rgba(15, 118, 110, 0.16), transparent 30%),
                radial-gradient(circle at 82% 14%, rgba(234, 88, 12, 0.14), transparent 24%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(240, 248, 249, 0.82));
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        .assistant-launcher-card--interactive {
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }

        .assistant-launcher-anchor:hover .assistant-launcher-card--interactive {
            transform: translateY(-3px);
            border-color: rgba(15, 118, 110, 0.26);
            box-shadow: 0 24px 38px rgba(15, 118, 110, 0.14);
        }

        .assistant-launcher-glow {
            position: absolute;
            inset: auto auto -52px -42px;
            width: 180px;
            height: 180px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.20), transparent 68%);
            pointer-events: none;
        }

        .assistant-launcher-visual {
            position: relative;
            z-index: 1;
            width: 100%;
            display: flex;
            justify-content: center;
        }

        .assistant-bot-image {
            display: block;
            width: min(100%, 190px);
            height: auto;
            filter: drop-shadow(0 18px 26px rgba(15, 118, 110, 0.14));
        }

        .assistant-launcher-logo {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 78px;
            height: 78px;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f766e 0%, #1e9387 100%);
            color: #ffffff;
            font-family: var(--font-data);
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            box-shadow: 0 18px 32px rgba(15, 118, 110, 0.16);
        }

        .assistant-launcher-kicker-row {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 118, 110, 0.12);
        }

        .assistant-launcher-kicker {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-family: var(--font-data);
        }

        .assistant-launcher-title {
            color: var(--ink);
            font-family: var(--font-display);
            font-size: 1.22rem;
            font-weight: 700;
            line-height: 1.15;
        }

        .assistant-launcher-copy {
            margin: 0;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .assistant-launcher-stats {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            width: 100%;
        }

        .assistant-launcher-stats > div {
            min-width: 0;
            padding: 0.62rem 0.66rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 118, 110, 0.12);
            text-align: left;
        }

        .assistant-launcher-stats span {
            display: block;
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-family: var(--font-data);
        }

        .assistant-launcher-stats strong {
            display: block;
            margin-top: 0.26rem;
            color: var(--ink);
            font-size: 0.82rem;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .assistant-launcher-tags {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.45rem;
        }

        .assistant-tag {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.68rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(15, 118, 110, 0.12);
            color: #0f5f59;
            font-size: 0.76rem;
            font-weight: 700;
            font-family: var(--font-data);
        }

        .assistant-quick-title {
            margin: 0;
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            font-weight: 800;
            font-family: var(--font-data);
        }

        .assistant-prompt-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin: 0.85rem 0 0.65rem;
        }

        .assistant-prompt-count {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 700;
            font-family: var(--font-data);
        }

        .assistant-live-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.58rem;
            margin: 0.68rem 0 0.62rem;
        }

        .assistant-live-card {
            min-width: 0;
            padding: 0.72rem 0.76rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 118, 110, 0.12);
            box-shadow: 0 8px 18px rgba(16, 24, 40, 0.04);
        }

        .assistant-live-card--accent {
            background: linear-gradient(180deg, rgba(215, 241, 238, 0.70), rgba(255, 255, 255, 0.88));
        }

        .assistant-live-label {
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.11em;
            font-size: 0.62rem;
            font-weight: 850;
            font-family: var(--font-data);
        }

        .assistant-live-value {
            margin-top: 0.28rem;
            color: var(--ink);
            font-size: 0.86rem;
            line-height: 1.28;
            font-weight: 800;
            overflow-wrap: anywhere;
        }

        .assistant-live-note {
            margin-top: 0.2rem;
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 700;
            font-family: var(--font-data);
        }

        .assistant-symptom-rack {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0 0 0.8rem;
        }

        .assistant-symptom-chip {
            display: inline-flex;
            align-items: center;
            max-width: 100%;
            padding: 0.36rem 0.58rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(15, 118, 110, 0.13);
            color: #0f5f59;
            font-size: 0.72rem;
            font-weight: 750;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }

        .assistant-symptom-chip--more {
            background: rgba(234, 88, 12, 0.10);
            border-color: rgba(234, 88, 12, 0.16);
            color: #9a3d0a;
        }

        .assistant-symptom-chip--empty {
            background: rgba(95, 107, 122, 0.08);
            border-color: rgba(95, 107, 122, 0.14);
            color: var(--muted);
        }

        .assistant-summary-card {
            min-height: 100%;
            padding: 0.95rem 0.95rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.78));
            border: 1px solid rgba(15, 118, 110, 0.12);
            box-shadow: var(--shadow-soft);
        }

        .assistant-summary-card--accent {
            background: linear-gradient(180deg, rgba(215, 241, 238, 0.68), rgba(255, 255, 255, 0.9));
        }

        .assistant-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.85rem;
            margin-bottom: 0.3rem;
        }

        .assistant-summary-label {
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.68rem;
            font-weight: 800;
            font-family: var(--font-data);
        }

        .assistant-summary-value {
            margin-top: 0.42rem;
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.35;
            font-weight: 700;
            font-family: var(--font-display);
        }

        .assistant-summary-copy {
            margin: 0.45rem 0 0 0;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.55;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.74);
            border-radius: 28px;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stChatMessage"] {
            padding: 0.7rem 0.8rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 118, 110, 0.12);
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.05);
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0;
        }

        .assistant-chat-history {
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
            width: 100%;
            min-height: 240px;
            max-height: none;
            overflow-y: visible;
            overflow-x: hidden;
            padding: 0.85rem;
            margin: 0.6rem 0 0.85rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(15, 118, 110, 0.14);
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.05);
        }

        .assistant-chat-row {
            display: flex;
            align-items: flex-end;
            gap: 0.48rem;
            width: 100%;
        }

        .assistant-chat-row--user {
            justify-content: flex-end;
            flex-direction: row-reverse;
        }

        .assistant-chat-row--assistant {
            justify-content: flex-start;
        }

        .assistant-chat-avatar {
            flex: 0 0 auto;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 12px;
            font-family: var(--font-data);
            font-size: 0.62rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            box-shadow: 0 8px 16px rgba(16, 24, 40, 0.06);
        }

        .assistant-chat-avatar--assistant {
            color: #0f5f59;
            background: rgba(215, 241, 238, 0.92);
            border: 1px solid rgba(15, 118, 110, 0.12);
        }

        .assistant-chat-avatar--user {
            color: #ffffff;
            background: linear-gradient(135deg, #0f766e, #145b91);
            border: 1px solid rgba(255, 255, 255, 0.24);
        }

        @keyframes chatBubbleIn {
            0% { opacity: 0; transform: translateY(8px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .assistant-chat-bubble {
            width: 100%;
            max-width: 100%;
            padding: 0.82rem 1rem;
            border-radius: 16px;
            line-height: 1.6;
            overflow-wrap: anywhere;
            word-break: normal;
            box-shadow: 0 4px 14px rgba(16, 24, 40, 0.05);
            animation: chatBubbleIn 0.35s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
        }

        .assistant-chat-bubble--assistant {
            background: linear-gradient(135deg, #ffffff 0%, #f0faf9 100%);
            border: 1px solid rgba(15, 118, 110, 0.12);
            color: var(--ink);
        }

        .assistant-chat-bubble--user {
            background: linear-gradient(135deg, var(--teal) 0%, #0d6b64 100%);
            border: 1px solid rgba(15, 118, 110, 0.26);
            color: #ffffff;
        }

        .assistant-chat-label {
            margin-bottom: 0.22rem;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: inherit;
            opacity: 0.45;
        }

        .assistant-chat-text {
            font-size: 0.93rem;
            line-height: 1.6;
            color: inherit;
            white-space: pre-wrap;
        }

        .assistant-chat-text strong {
            font-weight: 850;
            color: inherit;
        }

        .assistant-chat-bubble--assistant .assistant-chat-text strong {
            color: #0b5f58;
        }

        .assistant-chat-empty {
            margin: auto;
            max-width: 360px;
            color: var(--muted);
            text-align: center;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .assistant-chat-empty-title {
            color: var(--ink);
            font-family: var(--font-display);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.24rem;
        }

        .assistant-chat-empty-copy {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.5;
        }

        .assistant-connection-note {
            margin: 0.05rem 0 0.82rem;
            padding: 0.55rem 0.68rem;
            border-radius: 14px;
            background: rgba(15, 118, 110, 0.07);
            border: 1px solid rgba(15, 118, 110, 0.10);
            color: var(--muted);
            font-family: var(--font-data);
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1.4;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        textarea {
            border-radius: 16px !important;
            border-color: rgba(15, 118, 110, 0.20) !important;
            background: rgba(255, 255, 255, 0.94) !important;
        }

        input,
        textarea,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            background: transparent !important;
            box-shadow: none !important;
            font-family: var(--font-ui) !important;
            font-size: 0.98rem !important;
            line-height: 1.6 !important;
        }

        [data-baseweb="input"] input::placeholder,
        [data-baseweb="base-input"] input::placeholder,
        textarea::placeholder {
            color: #8390a1 !important;
        }

        div[data-baseweb="tag"] {
            border-radius: 999px !important;
            background: rgba(15, 118, 110, 0.10) !important;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            padding: 0.35rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stAlert"] {
            border-radius: 20px;
            border: 1px solid rgba(15, 118, 110, 0.14);
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stPlotlyChart"] {
            padding: 0.3rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(15, 118, 110, 0.12);
            box-shadow: var(--shadow-soft);
        }

        hr {
            border-color: rgba(15, 118, 110, 0.12);
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .page-hero {
                padding: 1.3rem 1.2rem;
                border-radius: 24px;
            }

            .topbar-shell {
                padding: 0.95rem 1rem;
                border-radius: 24px;
            }

            .page-title {
                font-size: 2rem;
            }

            .page-description {
                font-size: 0.98rem;
                line-height: 1.66;
            }
        }

        @media (max-width: 720px) {
            .block-container {
                padding-top: 0.8rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-bottom: 2rem;
            }

            div[data-testid="stHorizontalBlock"] {
                gap: 0.7rem;
            }

            div[data-testid="column"] {
                min-width: 100% !important;
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            .page-hero {
                padding: 1.1rem 1rem;
                border-radius: 22px;
            }

            .topbar-shell {
                padding: 0.9rem 0.95rem;
                border-radius: 20px;
            }

            .topbar-title {
                font-size: 1.38rem;
            }

            .topbar-subtitle {
                font-size: 0.92rem;
            }

            .assistant-shell-head {
                flex-direction: column;
                padding: 0.95rem 0.95rem;
                border-radius: 22px;
            }

            .assistant-mini-brand {
                align-items: flex-start;
            }

            .assistant-panel-header {
                padding: 0.95rem 0.9rem;
                border-radius: 22px;
            }

            .assistant-panel-avatar {
                width: 62px;
            }

            .assistant-inline-status {
                margin-top: 0.45rem;
                white-space: normal;
            }

            .assistant-launcher-card {
                padding: 1.05rem 0.9rem 0.95rem 0.9rem;
                border-radius: 24px;
            }

            .assistant-bot-image {
                width: min(100%, 155px);
            }

            .page-title {
                font-size: 1.74rem;
                line-height: 1.12;
            }

            .page-eyebrow {
                font-size: 0.7rem;
                letter-spacing: 0.14em;
            }

            .page-description {
                margin-top: 0.6rem;
                font-size: 0.96rem;
            }

            .hero-chip-row {
                gap: 0.4rem;
            }

            .hero-chip {
                font-size: 0.74rem;
                padding: 0.42rem 0.7rem;
            }

            .status-card,
            .info-card,
            div[data-testid="stMetric"],
            div[data-testid="stForm"],
            div[data-testid="stExpander"],
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"],
            div[data-testid="stAlert"],
            div[data-testid="stPlotlyChart"] {
                border-radius: 20px;
            }

            .section-title {
                font-size: 1.34rem;
            }

            .section-copy {
                font-size: 0.96rem;
                margin-bottom: 0.85rem;
            }

            .stButton > button,
            .stLinkButton a {
                min-height: 2.85rem;
                font-size: 0.98rem;
                border-radius: 16px;
            }

            div[data-baseweb="base-input"] > div,
            div[data-baseweb="select"] > div,
            textarea {
                border-radius: 14px !important;
            }
        }

        /* Responsive UI refresh for the active s06 dashboard experience. */
        :root {
            --bg-1: #f7efe4;
            --bg-2: #eef7f7;
            --ink: #101828;
            --muted: #667085;
            --line: rgba(15, 118, 110, 0.14);
            --panel: rgba(255, 255, 255, 0.9);
            --panel-strong: rgba(255, 255, 255, 0.96);
            --teal: #0f766e;
            --teal-soft: #e6f5f3;
            --orange: #b45309;
            --blue: #2563eb;
            --amber: #f59e0b;
            --shadow: 0 12px 28px rgba(16, 24, 40, 0.08);
            --shadow-soft: 0 8px 18px rgba(16, 24, 40, 0.06);
            --radius: 8px;
        }

        html,
        body,
        [class*="css"],
        h1,
        h2,
        h3,
        h4,
        p,
        label,
        button,
        input,
        textarea,
        .stMarkdown,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            letter-spacing: 0 !important;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 0% 0%, rgba(15, 118, 110, 0.16), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(234, 88, 12, 0.14), transparent 24%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 42%, #f9fbfc 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            line-height: 1.18;
        }

        .topbar-shell {
            padding: 1rem 1.1rem;
            border-radius: var(--radius);
            background: var(--panel-strong);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            margin-bottom: 0.75rem;
        }

        .topbar-title {
            font-size: 1.45rem;
        }

        .topbar-subtitle {
            max-width: 760px;
            color: var(--muted);
            font-size: 0.96rem;
        }

        .inline-chip,
        .hero-chip,
        .assistant-tag,
        .assistant-mode-chip,
        .assistant-inline-status,
        .assistant-launcher-kicker-row {
            border-radius: 999px;
            background: var(--teal-soft);
            border: 1px solid rgba(15, 118, 110, 0.18);
            color: #0b5f58;
        }

        .page-hero {
            padding: 1.1rem 0 1.25rem;
            margin-bottom: 1.1rem;
            border: 0;
            border-bottom: 1px solid var(--line);
            border-radius: 0;
            background: transparent;
            box-shadow: none;
        }

        .page-hero::after,
        .assistant-launcher-glow {
            display: none;
        }

        .page-eyebrow,
        .status-label,
        .card-kicker,
        .assistant-panel-kicker,
        .assistant-launcher-kicker,
        .assistant-quick-title,
        .assistant-summary-label {
            letter-spacing: 0 !important;
            text-transform: none;
            color: var(--teal);
            font-size: 0.78rem;
        }

        .page-title {
            max-width: 820px;
            font-size: 2.05rem;
            line-height: 1.16;
        }

        .page-description {
            max-width: 820px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.64;
        }

        .hero-chip-row {
            gap: 0.45rem;
        }

        .hero-chip {
            padding: 0.38rem 0.62rem;
            font-size: 0.78rem;
        }

        .status-card,
        .info-card,
        .assistant-shell-head,
        .assistant-panel-header,
        .assistant-launcher-card,
        .assistant-summary-card,
        div[data-testid="stMetric"],
        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stAlert"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stChatMessage"] {
            border-radius: var(--radius) !important;
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            box-shadow: var(--shadow-soft) !important;
        }

        .status-card,
        .info-card,
        .assistant-summary-card {
            padding: 1rem;
        }

        .status-title,
        .info-card h4,
        .assistant-shell-title,
        .assistant-launcher-title,
        .assistant-summary-value {
            font-size: 1.02rem;
            line-height: 1.3;
        }

        .status-copy,
        .info-card p,
        .assistant-shell-copy,
        .assistant-launcher-copy,
        .assistant-summary-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.58;
        }

        .section-title {
            margin-top: 0.4rem;
            font-size: 1.35rem;
            line-height: 1.24;
        }

        .section-copy {
            max-width: 820px;
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.62;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton a {
            width: 100% !important;
            min-height: 2.85rem;
            border-radius: var(--radius) !important;
            border: 1px solid rgba(15, 118, 110, 0.22) !important;
            box-shadow: none !important;
            font-size: 0.94rem;
        }

        div.stButton,
        div[data-testid="stButton"] {
            width: 100% !important;
            min-width: 100% !important;
        }

        div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) button {
            width: 100% !important;
            min-width: 100% !important;
            justify-content: center !important;
        }

        div[data-testid="stPills"],
        div[data-testid="stSegmentedControl"] {
            width: 100% !important;
            overflow: visible !important;
            padding: 0.25rem 0.15rem 0.45rem 0;
            scrollbar-width: none;
        }

        div[data-testid="stPills"]::-webkit-scrollbar,
        div[data-testid="stSegmentedControl"]::-webkit-scrollbar,
        div[data-baseweb="button-group"][role="radiogroup"]::-webkit-scrollbar {
            display: none;
        }

        div[data-testid="stPills"] [role="radiogroup"],
        div[data-testid="stSegmentedControl"] [role="radiogroup"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.55rem !important;
            width: 100% !important;
            min-width: 100% !important;
        }

        div[data-testid="stPills"] label,
        div[data-testid="stSegmentedControl"] label {
            flex: 1 1 118px !important;
            min-width: 108px !important;
            justify-content: center !important;
            border-radius: var(--radius) !important;
            border: 1px solid rgba(15, 118, 110, 0.18) !important;
            background: rgba(255, 255, 255, 0.86) !important;
            box-shadow: var(--shadow-soft) !important;
            color: var(--ink) !important;
            font-weight: 800 !important;
            transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease;
        }

        div[data-testid="stPills"] label:hover,
        div[data-testid="stSegmentedControl"] label:hover {
            transform: translateY(-1px);
            border-color: rgba(15, 118, 110, 0.32) !important;
        }

        div[data-testid="stPills"] label:has(input:checked),
        div[data-testid="stSegmentedControl"] label:has(input:checked) {
            background: var(--teal) !important;
            border-color: var(--teal) !important;
            color: #ffffff !important;
        }

        div[data-testid="stPills"] label:has(input:checked) p,
        div[data-testid="stSegmentedControl"] label:has(input:checked) p {
            color: #ffffff !important;
        }

        div[data-baseweb="button-group"][role="radiogroup"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.55rem !important;
            width: 100% !important;
            overflow: visible !important;
            padding: 0.15rem 0 0.45rem !important;
            scrollbar-width: none;
        }

        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pills"],
        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pillsActive"] {
            flex: 1 1 108px !important;
            width: auto !important;
            min-width: 108px !important;
            max-width: none !important;
            height: 2.75rem !important;
            border-radius: var(--radius) !important;
            justify-content: center !important;
            box-shadow: var(--shadow-soft) !important;
        }

        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pills"] {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(15, 118, 110, 0.18) !important;
            color: var(--ink) !important;
        }

        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pillsActive"] {
            background: var(--teal) !important;
            border: 1px solid var(--teal) !important;
            color: #ffffff !important;
        }

        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pills"] p {
            color: var(--ink) !important;
        }

        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pillsActive"] p {
            color: #ffffff !important;
        }

        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pills"]:hover,
        div[data-baseweb="button-group"][role="radiogroup"] button[data-testid="stBaseButton-pillsActive"]:hover {
            transform: translateY(-1px);
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stLinkButton a:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-soft) !important;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            background: var(--teal) !important;
            border-color: var(--teal) !important;
            color: #ffffff !important;
        }

        .stButton > button[kind="primary"] p,
        .stFormSubmitButton > button[kind="primary"] p,
        div[data-testid="stButton"] > button[kind="primary"] p {
            color: #ffffff !important;
        }

        .stButton > button[kind="secondary"] p,
        div[data-testid="stButton"] > button[kind="secondary"] p {
            color: var(--ink) !important;
        }

        div[data-testid="stMetric"] {
            min-height: 104px;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            text-transform: none;
            font-size: 0.78rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.45rem !important;
        }

        div[data-testid="stForm"] {
            padding: 1rem 1rem 0.55rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        textarea {
            border-radius: var(--radius) !important;
            border-color: var(--line) !important;
            background: #ffffff !important;
        }

        .assistant-launcher-card {
            gap: 0.6rem;
            padding: 1rem;
            text-align: left;
            align-items: flex-start;
        }

        .assistant-launcher-visual {
            justify-content: flex-start;
        }

        .assistant-bot-image {
            width: min(100%, 132px);
        }

        .assistant-launcher-logo {
            width: 58px;
            height: 58px;
            border-radius: var(--radius);
            background: var(--teal);
        }

        .assistant-summary-grid {
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        }

        .stMultiSelect [data-baseweb="tag"],
        div[data-baseweb="tag"] {
            border-radius: 999px !important;
            background: var(--teal-soft) !important;
            color: #0b5f58 !important;
        }

        div[data-testid="stPlotlyChart"] {
            padding: 0.5rem;
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .page-title {
                font-size: 1.72rem;
            }

            .topbar-title {
                font-size: 1.3rem;
            }

            div[data-testid="stMetric"] {
                min-height: auto;
            }
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }

            .page-hero {
                padding-top: 0.75rem;
            }

            .page-title {
                font-size: 1.48rem;
            }

            .page-description,
            .section-copy {
                font-size: 0.93rem;
            }

            .section-title {
                font-size: 1.16rem;
            }

            .stButton > button,
            .stFormSubmitButton > button,
            .stLinkButton a {
                min-height: 2.75rem;
                font-size: 0.92rem;
            }
        }

        /* Dashboard sidebar navigation */
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(230, 245, 243, 0.9)),
                linear-gradient(180deg, #ffffff, #f6fbfb);
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.1rem;
        }

        .sidebar-brand {
            padding: 1rem;
            border-radius: var(--radius);
            background: linear-gradient(135deg, #0f766e, #2563eb);
            color: #ffffff;
            box-shadow: 0 14px 28px rgba(15, 118, 110, 0.16);
            margin-bottom: 0.85rem;
        }

        .sidebar-brand-title {
            margin: 0;
            font-family: var(--font-display);
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.22;
            color: #ffffff;
        }

        .sidebar-brand-copy {
            margin: 0.4rem 0 0 0;
            color: rgba(255, 255, 255, 0.82);
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .sidebar-user-chip {
            display: block;
            padding: 0.75rem 0.85rem;
            margin: 0.65rem 0 0.85rem 0;
            border-radius: var(--radius);
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 118, 110, 0.16);
            color: var(--ink);
            font-weight: 800;
            font-size: 0.86rem;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] {
            gap: 0.45rem;
        }

        section[data-testid="stSidebar"] label {
            min-height: 2.75rem;
            padding: 0.55rem 0.7rem !important;
            border-radius: var(--radius) !important;
            background: rgba(255, 255, 255, 0.78) !important;
            border: 1px solid rgba(15, 118, 110, 0.12) !important;
            box-shadow: var(--shadow-soft);
            transition: transform 0.16s ease, background 0.16s ease, border-color 0.16s ease;
        }

        section[data-testid="stSidebar"] label:hover {
            transform: translateX(2px);
            border-color: rgba(15, 118, 110, 0.28) !important;
        }

        section[data-testid="stSidebar"] label:has(input:checked) {
            background: var(--teal) !important;
            border-color: var(--teal) !important;
        }

        section[data-testid="stSidebar"] label:has(input:checked) p {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        .topbar-shell {
            margin-bottom: 1.05rem;
        }

        /* Fixed button sizing */
        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton a,
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100% !important;
            height: 46px !important;
            min-height: 46px !important;
            max-height: 46px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 0.95rem !important;
            line-height: 1.1 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        .stButton > button p,
        .stFormSubmitButton > button p,
        .stLinkButton a p,
        div[data-testid="stButton"] > button p,
        div[data-testid="stFormSubmitButton"] > button p {
            margin: 0 !important;
            line-height: 1.1 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 0.55rem !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label {
            width: 100% !important;
            height: 46px !important;
            min-height: 46px !important;
            max-height: 46px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            overflow: hidden !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label p {
            margin: 0 !important;
            line-height: 1.1 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        /* Dynamic button polish */
        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton a,
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 14px !important;
            background: linear-gradient(145deg, #ffffff, #f4f6ff) !important;
            border: 1px solid rgba(90, 99, 216, 0.14) !important;
            color: var(--patient-text) !important;
            box-shadow: 0 10px 22px rgba(64, 74, 130, 0.08) !important;
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease, background 0.16s ease !important;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stLinkButton a:hover,
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-2px) !important;
            border-color: rgba(90, 99, 216, 0.32) !important;
            box-shadow: 0 16px 30px rgba(64, 74, 130, 0.16) !important;
            background: linear-gradient(145deg, #ffffff, #eef1ff) !important;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: linear-gradient(145deg, var(--patient-blue), var(--patient-blue-dark)) !important;
            border-color: transparent !important;
            color: #ffffff !important;
            box-shadow: 0 14px 26px rgba(90, 99, 216, 0.28) !important;
        }

        .stButton > button[kind="primary"] p,
        .stFormSubmitButton > button[kind="primary"] p,
        div[data-testid="stButton"] > button[kind="primary"] p,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] p {
            color: #ffffff !important;
        }

        /* Patient dashboard visual system inspired by the provided reference. */
        :root {
            --patient-bg: #eef2fb;
            --patient-surface: #ffffff;
            --patient-card: #ffffff;
            --patient-blue: #5a63d8;
            --patient-blue-dark: #2d3192;
            --patient-lilac: #f1f2ff;
            --patient-cyan: #eaf8ff;
            --patient-yellow: #fff6d8;
            --patient-red: #fff0f0;
            --patient-text: #27304f;
            --patient-muted: #7b83a5;
            --patient-line: rgba(90, 99, 216, 0.12);
            --patient-shadow: 0 22px 55px rgba(64, 74, 130, 0.12);
            --patient-soft-shadow: 0 12px 28px rgba(64, 74, 130, 0.08);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 18% 6%, rgba(90, 99, 216, 0.12), transparent 22%),
                radial-gradient(circle at 84% 12%, rgba(255, 255, 255, 0.72), transparent 12%),
                linear-gradient(180deg, #eef2fb 0%, #f7f9ff 100%) !important;
        }

        .block-container {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            padding-top: 1.1rem !important;
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
            box-sizing: border-box;
        }

        section[data-testid="stSidebar"] {
            background: transparent !important;
            border-right: 0 !important;
            z-index: 20 !important;
            width: var(--sidebar-width) !important;
            min-width: var(--sidebar-width) !important;
            max-width: var(--sidebar-width) !important;
            flex: 0 0 var(--sidebar-width) !important;
            resize: none !important;
            overflow: visible !important;
        }

        section[data-testid="stSidebar"] > div {
            width: calc(var(--sidebar-width) - 1.1rem) !important;
            min-width: calc(var(--sidebar-width) - 1.1rem) !important;
            max-width: calc(var(--sidebar-width) - 1.1rem) !important;
            box-sizing: border-box !important;
            resize: none !important;
            margin: 1rem 0.55rem;
            border-radius: 26px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.8);
            box-shadow: var(--patient-soft-shadow);
            max-height: calc(100vh - 2rem);
            overflow-y: auto;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"],
        section[data-testid="stSidebar"] [data-testid="stSidebarResizer"],
        div[data-testid="stSidebarResizeHandle"],
        div[data-testid="stSidebarResizer"],
        [class*="resizer"],
        [class*="ResizeHandle"] {
            display: none !important;
            pointer-events: none !important;
            width: 0 !important;
            opacity: 0 !important;
        }

        .sidebar-brand {
            background: linear-gradient(145deg, #ffffff, #f4f6ff) !important;
            color: var(--patient-blue-dark) !important;
            border: 1px solid var(--patient-line);
            box-shadow: none !important;
        }

        .sidebar-brand-title {
            color: var(--patient-blue-dark) !important;
        }

        .sidebar-brand-copy {
            color: var(--patient-muted) !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label {
            background: #ffffff !important;
            border-color: transparent !important;
            color: var(--patient-blue-dark) !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: var(--patient-lilac) !important;
            transform: translateX(3px);
        }

        section[data-testid="stSidebar"] label:has(input:checked) {
            background: var(--patient-blue) !important;
            border-color: var(--patient-blue) !important;
            box-shadow: 0 12px 24px rgba(90, 99, 216, 0.22);
        }

        .topbar-shell {
            border-radius: 24px !important;
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.88) !important;
            box-shadow: var(--patient-soft-shadow) !important;
        }

        .patient-dashboard {
            padding: 0.2rem 0 1rem 0;
        }

        .patient-hero-card {
            position: relative;
            min-height: 150px;
            overflow: hidden;
            padding: 1.25rem 1.3rem;
            border-radius: 24px;
            background:
                linear-gradient(120deg, rgba(255, 255, 255, 0.98), rgba(245, 247, 255, 0.94)),
                var(--patient-surface);
            border: 1px solid rgba(255, 255, 255, 0.9);
            box-shadow: var(--patient-shadow);
            margin-bottom: 1rem;
        }

        .patient-hero-card::after {
            content: "";
            position: absolute;
            right: -34px;
            bottom: -42px;
            width: 190px;
            height: 190px;
            border-radius: 999px;
            background: rgba(90, 99, 216, 0.10);
        }

        .patient-hero-copy {
            position: relative;
            max-width: 100%;
            z-index: 1;
        }

        .patient-greeting {
            margin: 0;
            color: var(--patient-text);
            font-family: var(--font-display);
            font-size: 1.38rem;
            line-height: 1.22;
        }

        .patient-greeting span {
            color: #f5b642;
        }

        .patient-hero-text {
            margin: 0.55rem 0 0 0;
            color: var(--patient-muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .patient-read-more {
            display: inline-flex;
            margin-top: 0.65rem;
            color: var(--patient-blue);
            font-weight: 800;
            font-size: 0.86rem;
        }

        .patient-stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-bottom: 0.95rem;
        }

        .patient-stat-card {
            min-height: 112px;
            padding: 1rem;
            border-radius: 18px;
            background: var(--patient-card);
            border: 1px solid rgba(255, 255, 255, 0.9);
            box-shadow: var(--patient-soft-shadow);
        }

        .patient-stat-card.featured {
            background: linear-gradient(145deg, #eef2ff, #e5edff);
            color: var(--patient-text);
        }

        .patient-stat-kicker {
            margin: 0;
            color: var(--patient-muted);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .patient-stat-card.featured .patient-stat-kicker,
        .patient-stat-card.featured .patient-stat-sub,
        .patient-stat-card.featured .patient-stat-value {
            color: var(--patient-text);
        }

        .patient-stat-card.stat-blue {
            background: linear-gradient(145deg, #dfeeff, #d6e8ff);
        }

        .patient-stat-card.stat-mint {
            background: linear-gradient(145deg, #dcf8ef, #d2f0e8);
        }

        .patient-stat-card.stat-yellow {
            background: linear-gradient(145deg, #fff0bf, #ffe8a3);
        }

        .patient-stat-card.stat-rose {
            background: linear-gradient(145deg, #ffdfe7, #ffd3df);
        }

        .patient-stat-value {
            margin: 0.38rem 0 0 0;
            color: var(--patient-text);
            font-family: var(--font-display);
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.12;
        }

        .patient-stat-sub {
            margin: 0.4rem 0 0 0;
            color: var(--patient-muted);
            font-size: 0.75rem;
            line-height: 1.35;
        }

        .patient-card-grid {
            display: grid;
            grid-template-columns: 0.72fr 1.28fr;
            gap: 0.95rem;
        }

        .patient-mini-panel,
        .patient-chart-panel,
        .patient-profile-panel {
            border-radius: 24px;
            background: var(--patient-card);
            border: 1px solid rgba(255, 255, 255, 0.9);
            box-shadow: var(--patient-soft-shadow);
        }

        .patient-mini-panel {
            padding: 1rem;
            margin-bottom: 0.95rem;
        }

        .patient-plot-title {
            margin: 0 0 0.45rem 0;
            color: var(--patient-text);
            font-family: var(--font-display);
            font-size: 0.94rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .patient-plot-caption {
            margin: -0.18rem 0 0.55rem 0;
            color: var(--patient-muted);
            font-size: 0.74rem;
            line-height: 1.35;
        }

        .patient-mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.95rem;
        }

        .patient-mini-label {
            margin: 0;
            color: var(--patient-text);
            font-weight: 800;
            font-size: 0.88rem;
        }

        .patient-ring-row {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-top: 0.75rem;
        }

        .patient-ring {
            width: 70px;
            height: 70px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            font-weight: 900;
            color: var(--patient-blue);
            background: conic-gradient(var(--patient-blue) 0 42%, #e8ecff 42% 100%);
        }

        .patient-ring.gold {
            color: #e0a90e;
            background: conic-gradient(#f5c84b 0 61%, #fff4ca 61% 100%);
        }

        .patient-chart-panel {
            padding: 1rem 1rem 0.35rem 1rem;
        }

        .patient-chart-heading {
            margin: 0 0 0.6rem 0;
            padding: 1rem 1rem 0 1rem;
            color: var(--patient-text);
            font-weight: 800;
            font-size: 0.9rem;
        }

        .patient-profile-panel {
            min-height: 100%;
            padding: 1.05rem;
            background: linear-gradient(180deg, #dbeeff, #cce4ff);
            color: var(--patient-text);
        }

        .patient-avatar {
            width: 86px;
            height: 86px;
            border-radius: 999px;
            margin: 0 auto 0.8rem auto;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(226, 241, 255, 0.82));
            border: 3px solid rgba(255, 255, 255, 0.92);
            box-shadow: 0 14px 28px rgba(63, 104, 176, 0.22);
            overflow: hidden;
        }

        .patient-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scale(1.03);
        }

        .patient-avatar span {
            color: #2f65b9;
            font-family: var(--font-display);
            font-size: 1.45rem;
            font-weight: 900;
        }

        .patient-profile-name {
            margin: 0;
            color: var(--patient-text);
            text-align: center;
            font-family: var(--font-display);
            font-size: 1.05rem;
        }

        .patient-profile-role {
            margin: 0.25rem 0 1rem 0;
            text-align: center;
            color: var(--patient-muted);
            font-size: 0.78rem;
        }

        .patient-profile-stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            margin-bottom: 1rem;
        }

        .patient-profile-chip {
            padding: 0.65rem 0.35rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.72);
            text-align: center;
        }

        .patient-profile-chip strong {
            display: block;
            color: var(--patient-text);
            font-size: 0.82rem;
        }

        .patient-profile-chip span {
            color: var(--patient-muted);
            font-size: 0.66rem;
        }

        .patient-calendar-title,
        .patient-plan-title {
            margin: 0.95rem 0 0.6rem 0;
            color: var(--patient-text);
            font-size: 0.86rem;
            font-weight: 800;
        }

        .patient-calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 0.28rem;
        }

        .patient-calendar-weekday {
            height: 20px;
            display: grid;
            place-items: center;
            color: var(--patient-muted);
            font-family: var(--font-data);
            font-size: 0.58rem;
            font-weight: 900;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .patient-calendar-day {
            height: 24px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: var(--patient-muted);
            font-size: 0.66rem;
        }

        .patient-calendar-day.empty {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            pointer-events: none;
        }

        .patient-calendar-day.active {
            background: var(--patient-blue);
            color: #ffffff;
            font-weight: 900;
        }

        .patient-appointment-item {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            padding: 0.74rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.55rem;
        }

        .patient-appointment-item strong {
            color: var(--patient-text);
            font-size: 0.78rem;
        }

        .patient-appointment-item span {
            color: var(--patient-muted);
            font-size: 0.7rem;
        }

        .doctor-action-link,
        .doctor-action-card {
            width: 100%;
            height: 100%;
            box-sizing: border-box;
            min-height: 92px;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 1.05rem 1.08rem;
            border-radius: 18px;
            text-decoration: none !important;
            border: 1px solid rgba(255, 255, 255, 0.88);
            background: linear-gradient(145deg, #ffffff, #eef5ff);
            box-shadow: 0 14px 28px rgba(64, 74, 130, 0.10);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
            cursor: pointer;
        }

        .doctor-action-link:hover,
        .doctor-action-card:hover {
            transform: translateY(-2px);
            border-color: rgba(90, 99, 216, 0.28);
            box-shadow: 0 20px 34px rgba(64, 74, 130, 0.16);
            text-decoration: none !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.doctor-action-card),
        div[data-testid="stHorizontalBlock"]:has(.doctor-action-link) {
            gap: 1.15rem !important;
            margin: 0.95rem 0 1.45rem 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.doctor-action-card) div[data-testid="column"],
        div[data-testid="stHorizontalBlock"]:has(.doctor-action-link) div[data-testid="column"] {
            min-width: 0 !important;
        }

        .doctor-action-dot {
            width: 15px;
            height: 15px;
            flex: 0 0 auto;
            border-radius: 999px;
            background: #5a63d8;
            box-shadow: 0 0 0 7px rgba(90, 99, 216, 0.14);
        }

        .doctor-action-copy {
            display: grid;
            gap: 0.18rem;
        }

        .doctor-action-copy strong {
            color: var(--patient-text);
            font-size: 0.94rem;
            line-height: 1.15;
            font-family: var(--font-display);
        }

        .doctor-action-copy em {
            color: var(--patient-muted);
            font-style: normal;
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .doctor-action-mint {
            background: linear-gradient(145deg, #ffffff, #ddf7ef);
        }

        .doctor-action-mint .doctor-action-dot {
            background: #0f9f7a;
            box-shadow: 0 0 0 7px rgba(15, 159, 122, 0.14);
        }

        .doctor-action-sky {
            background: linear-gradient(145deg, #ffffff, #dfeeff);
        }

        .doctor-action-sky .doctor-action-dot {
            background: #2f7ee6;
            box-shadow: 0 0 0 7px rgba(47, 126, 230, 0.14);
        }

        .doctor-action-gold {
            background: linear-gradient(145deg, #ffffff, #fff0bf);
        }

        .doctor-action-gold .doctor-action-dot {
            background: #e0a90e;
            box-shadow: 0 0 0 7px rgba(224, 169, 14, 0.16);
        }

        .doctor-action-rose {
            background: linear-gradient(145deg, #ffffff, #ffe1e9);
        }

        .doctor-action-rose .doctor-action-dot {
            background: #e85d82;
            box-shadow: 0 0 0 7px rgba(232, 93, 130, 0.14);
        }

        .city-suggestion-panel {
            margin: 0.35rem 0 0.8rem 0;
            padding: 0.75rem;
            border-radius: 18px;
            background: linear-gradient(145deg, #ffffff, #eef6ff);
            border: 1px solid rgba(90, 99, 216, 0.12);
            box-shadow: var(--patient-soft-shadow);
        }

        .city-suggestion-title {
            margin: 0 0 0.5rem 0;
            color: var(--patient-text);
            font-weight: 800;
            font-size: 0.82rem;
        }

        .city-suggestion-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .city-suggestion-pill {
            max-width: 100%;
            padding: 0.38rem 0.58rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(90, 99, 216, 0.14);
            color: var(--patient-blue-dark);
            font-weight: 800;
            font-size: 0.72rem;
            line-height: 1.25;
            white-space: normal;
        }

        .provider-title {
            margin: 0.9rem 0 0.5rem 0;
            color: var(--patient-text);
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 0.98rem;
        }

        .provider-directory-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.72rem;
            margin-bottom: 0.85rem;
        }

        .provider-card-link {
            display: block;
            height: 100%;
            color: inherit !important;
            text-decoration: none !important;
        }

        .provider-card {
            min-height: 198px;
            padding: 0.86rem;
            border-radius: 18px;
            background: linear-gradient(145deg, #ffffff, #eef8ff);
            border: 1px solid rgba(90, 99, 216, 0.13);
            box-shadow: var(--patient-soft-shadow);
            display: flex;
            flex-direction: column;
            gap: 0.46rem;
        }

        .provider-card--clickable {
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
            cursor: pointer;
        }

        .provider-card-link:hover .provider-card--clickable {
            transform: translateY(-2px);
            border-color: rgba(15, 159, 122, 0.26);
            box-shadow: 0 18px 34px rgba(38, 52, 96, 0.14);
        }

        .provider-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.55rem;
        }

        .provider-card-top strong {
            color: #0f9f7a;
            font-family: var(--font-data);
            font-size: 0.82rem;
        }

        .provider-chip {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            padding: 0.28rem 0.52rem;
            border-radius: 999px;
            background: rgba(90, 99, 216, 0.10);
            color: var(--patient-blue-dark);
            font-weight: 800;
            font-size: 0.68rem;
        }

        .provider-card h4 {
            margin: 0;
            color: var(--patient-text);
            font-family: var(--font-display);
            font-size: 0.94rem;
            line-height: 1.2;
        }

        .provider-address {
            margin: 0;
            color: var(--patient-muted);
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .provider-meta,
        .provider-contact {
            display: flex;
            flex-wrap: wrap;
            gap: 0.34rem;
        }

        .provider-meta span,
        .provider-contact span,
        .provider-contact a {
            padding: 0.28rem 0.46rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 159, 122, 0.12);
            color: #356171;
            font-size: 0.67rem;
            font-weight: 700;
            text-decoration: none !important;
        }

        .provider-map-cta {
            margin-top: auto;
            width: fit-content;
            padding: 0.38rem 0.62rem;
            border-radius: 999px;
            background: rgba(15, 159, 122, 0.10);
            border: 1px solid rgba(15, 159, 122, 0.14);
            color: #226052;
            font-family: var(--font-data);
            font-size: 0.72rem;
            font-weight: 850;
        }

        .provider-empty {
            padding: 0.9rem;
            border-radius: 18px;
            background: linear-gradient(145deg, #ffffff, #f2f7ff);
            border: 1px dashed rgba(90, 99, 216, 0.20);
            color: var(--patient-muted);
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .provider-source-note {
            padding: 0.7rem 0.85rem;
            border-radius: 16px;
            background: rgba(221, 247, 239, 0.76);
            color: #226052;
            border: 1px solid rgba(15, 159, 122, 0.12);
            font-size: 0.74rem;
            font-weight: 700;
            margin: 0.35rem 0 0.75rem 0;
        }

        .remedy-table {
            width: 100%;
            overflow: hidden;
            margin-bottom: 1rem;
            border-radius: 20px;
            background: #ffffff;
            border: 1px solid rgba(90, 99, 216, 0.13);
            box-shadow: var(--patient-soft-shadow);
        }

        .remedy-summary-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.2rem 0 0.85rem;
        }

        .remedy-summary-card {
            padding: 0.78rem 0.86rem;
            border-radius: 16px;
            background: linear-gradient(145deg, #ffffff, #eef8ff);
            border: 1px solid rgba(90, 99, 216, 0.12);
            box-shadow: 0 10px 22px rgba(38, 52, 96, 0.06);
        }

        .remedy-summary-label {
            color: var(--patient-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-family: var(--font-data);
            font-size: 0.66rem;
            font-weight: 850;
        }

        .remedy-summary-value {
            margin-top: 0.24rem;
            color: var(--patient-text);
            font-family: var(--font-display);
            font-size: 1rem;
            font-weight: 850;
            line-height: 1.25;
        }

        .remedy-care-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }

        .remedy-care-table thead {
            background: linear-gradient(135deg, rgba(90, 99, 216, 0.12), rgba(15, 159, 122, 0.10));
        }

        .remedy-care-table th {
            padding: 0.78rem 0.82rem;
            color: var(--patient-blue-dark);
            text-align: left;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-family: var(--font-data);
            font-size: 0.67rem;
            font-weight: 900;
            border-bottom: 1px solid rgba(90, 99, 216, 0.12);
        }

        .remedy-care-table td {
            padding: 0.86rem 0.82rem;
            vertical-align: top;
            border-bottom: 1px solid rgba(90, 99, 216, 0.09);
        }

        .remedy-care-table tr:last-child td {
            border-bottom: 0;
        }

        .remedy-care-table col:nth-child(1) {
            width: 21%;
        }

        .remedy-care-table col:nth-child(2) {
            width: 39%;
        }

        .remedy-care-table col:nth-child(3) {
            width: 19%;
        }

        .remedy-care-table col:nth-child(4) {
            width: 21%;
        }

        .remedy-symptom-cell {
            display: flex;
            flex-direction: column;
            gap: 0.34rem;
            min-width: 0;
        }

        .remedy-symptom-name {
            color: var(--patient-text);
            font-family: var(--font-display);
            font-weight: 850;
            font-size: 0.92rem;
            line-height: 1.25;
        }

        .remedy-symptom-tag {
            width: fit-content;
            padding: 0.26rem 0.5rem;
            border-radius: 999px;
            background: rgba(15, 159, 122, 0.09);
            border: 1px solid rgba(15, 159, 122, 0.14);
            color: #226052;
            font-family: var(--font-data);
            font-size: 0.66rem;
            font-weight: 800;
        }

        .remedy-step-list {
            display: grid;
            gap: 0.42rem;
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .remedy-step {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            align-items: start;
            gap: 0.46rem;
            color: #234253;
            font-size: 0.78rem;
            line-height: 1.38;
            font-weight: 700;
        }

        .remedy-step-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.28rem;
            height: 1.28rem;
            border-radius: 999px;
            background: rgba(90, 99, 216, 0.10);
            color: var(--patient-blue-dark);
            font-family: var(--font-data);
            font-size: 0.62rem;
            font-weight: 900;
        }

        .remedy-purpose,
        .remedy-safety {
            color: var(--patient-muted);
            font-size: 0.76rem;
            line-height: 1.42;
            font-weight: 700;
        }

        .remedy-purpose strong,
        .remedy-safety strong {
            display: block;
            margin-bottom: 0.24rem;
            color: var(--patient-text);
            font-family: var(--font-display);
            font-size: 0.78rem;
            font-weight: 850;
        }

        .remedy-safety {
            padding: 0.52rem 0.6rem;
            border-radius: 14px;
            background: rgba(255, 247, 237, 0.78);
            border: 1px solid rgba(234, 88, 12, 0.12);
        }

        .remedy-table-note {
            margin: 0.68rem 0 1rem;
            padding: 0.66rem 0.78rem;
            border-radius: 16px;
            background: rgba(221, 247, 239, 0.76);
            border: 1px solid rgba(15, 159, 122, 0.12);
            color: #226052;
            font-size: 0.76rem;
            line-height: 1.45;
            font-weight: 750;
        }

        .remedy-row {
            display: grid;
            grid-template-columns: minmax(150px, 0.45fr) minmax(0, 1fr);
            gap: 0.7rem;
            align-items: start;
            padding: 0.8rem;
            border-radius: 18px;
            background: linear-gradient(145deg, #ffffff, #eef8ff);
            border: 1px solid rgba(90, 99, 216, 0.12);
            box-shadow: var(--patient-soft-shadow);
        }

        .remedy-symptom {
            color: var(--patient-text);
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 0.9rem;
        }

        .remedy-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }

        .remedy-pill {
            padding: 0.36rem 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(15, 159, 122, 0.14);
            color: #226052;
            font-size: 0.73rem;
            font-weight: 700;
        }

        .remedy-mobile-label {
            display: none;
        }

        .remedy-legacy-hidden {
            display: none;
        }

        .remedy-table-wrap {
            margin-bottom: 1rem;
        }

        .remedy-row {
            display: grid;
            grid-template-columns: minmax(150px, 0.45fr) minmax(0, 1fr);
            gap: 0.7rem;
            align-items: start;
            padding: 0.8rem;
            border-radius: 18px;
            background: linear-gradient(145deg, #ffffff, #eef8ff);
            border: 1px solid rgba(90, 99, 216, 0.12);
            box-shadow: var(--patient-soft-shadow);
        }

        .remedy-symptom {
            color: var(--patient-text);
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 0.9rem;
        }

        .remedy-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }

        .remedy-pill {
            padding: 0.36rem 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(15, 159, 122, 0.14);
            color: #226052;
            font-size: 0.73rem;
            font-weight: 700;
        }

        .assistant-panel-header {
            align-items: center !important;
            padding: 0.62rem !important;
            border-radius: 14px !important;
            margin-bottom: 0.2rem !important;
        }

        .assistant-mini-brand {
            align-items: center !important;
            gap: 0.55rem !important;
        }

        .assistant-panel-avatar {
            width: 42px !important;
            min-width: 42px !important;
        }

        .assistant-summary-grid {
            grid-template-columns: 1fr !important;
            gap: 0.65rem !important;
        }

        .assistant-live-strip {
            grid-template-columns: 1fr !important;
            gap: 0.52rem !important;
        }

        .assistant-live-card {
            padding: 0.64rem 0.68rem !important;
            border-radius: 14px !important;
        }

        .assistant-launcher-stats {
            grid-template-columns: 1fr !important;
        }

        .assistant-summary-card {
            min-height: auto !important;
            padding: 0.85rem !important;
        }

        .assistant-shell-title,
        .assistant-launcher-title {
            font-size: 0.96rem !important;
            line-height: 1.2 !important;
        }

        .assistant-shell-copy,
        .assistant-launcher-copy {
            font-size: 0.78rem !important;
            line-height: 1.32 !important;
            margin-top: 0.16rem !important;
        }

        .assistant-panel-kicker,
        .assistant-quick-title {
            font-size: 0.68rem !important;
        }

        .assistant-inline-status {
            margin-top: 0.35rem !important;
            padding: 0.26rem 0.48rem !important;
            font-size: 0.68rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) {
            padding: 0.65rem !important;
        }

        button[kind="secondary"],
        button[kind="primary"] {
            min-height: 42px !important;
            height: auto !important;
            white-space: normal !important;
        }

        button[kind="secondary"] p,
        button[kind="primary"] p {
            white-space: normal !important;
            line-height: 1.22 !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton a,
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            height: auto !important;
            min-height: 46px !important;
            white-space: normal !important;
            overflow: visible !important;
            align-items: center !important;
        }

        .stButton > button p,
        .stFormSubmitButton > button p,
        .stLinkButton a p,
        div[data-testid="stButton"] > button p,
        div[data-testid="stFormSubmitButton"] > button p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.22 !important;
        }

        div[data-testid="stChatMessage"] {
            padding: 0.55rem !important;
            border-radius: 14px !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }

        .assistant-panel-header,
        .assistant-launcher-card,
        .assistant-mini-brand,
        .assistant-shell-title,
        .assistant-shell-copy,
        .assistant-launcher-title,
        .assistant-launcher-copy,
        .assistant-inline-status,
        .assistant-launcher-kicker-row,
        .assistant-tag {
            max-width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
            overflow-wrap: anywhere !important;
        }

        .assistant-mini-brand {
            width: 100% !important;
        }

        .assistant-panel-avatar {
            flex: 0 0 auto !important;
        }

        .assistant-bot-image {
            max-width: 100% !important;
            object-fit: contain !important;
        }

        .assistant-launcher-tags {
            width: 100% !important;
            align-items: center !important;
        }

        .assistant-quick-title {
            margin-top: 0.7rem !important;
            margin-bottom: 0.55rem !important;
        }

        .assistant-prompt-bar {
            margin-top: 0.72rem !important;
            margin-bottom: 0.52rem !important;
        }

        .assistant-prompt-bar .assistant-quick-title {
            margin: 0 !important;
        }

        .assistant-symptom-rack {
            margin-bottom: 0.66rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) div[data-testid="stHorizontalBlock"] {
            gap: 0.5rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) div[data-testid="stChatMessage"] {
            max-width: 100% !important;
            overflow-wrap: anywhere !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) [data-testid="stMarkdownContainer"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) [data-testid="stMarkdownContainer"] p,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) [data-testid="stMarkdownContainer"] li {
            max-width: 100% !important;
            overflow-wrap: anywhere !important;
            word-break: normal !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) div[data-testid="stForm"] {
            padding: 0.75rem 0.75rem 0.35rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) input {
            min-height: 42px !important;
            font-size: 0.92rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) button,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) button p,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) button,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) button p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.22 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) button,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) button {
            height: auto !important;
            min-height: 46px !important;
            padding: 0.55rem 0.7rem !important;
        }

        @media (min-width: 1025px) {
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) {
                position: sticky !important;
                top: 1rem !important;
            }

            .assistant-panel-header {
                padding: 0.8rem !important;
            }

            .assistant-panel-avatar {
                width: 54px !important;
                min-width: 54px !important;
            }

            .assistant-shell-title,
            .assistant-launcher-title {
                font-size: 1.02rem !important;
            }

            .assistant-shell-copy,
            .assistant-launcher-copy {
                font-size: 0.86rem !important;
                line-height: 1.45 !important;
            }
        }

        @media (max-width: 720px) {
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-launcher-card) {
                border-radius: 18px !important;
            }

            div[data-testid="stHorizontalBlock"]:has(.doctor-action-card),
            div[data-testid="stHorizontalBlock"]:has(.doctor-action-link) {
                gap: 0.9rem !important;
                margin: 0.85rem 0 1.25rem 0 !important;
            }

            div[data-testid="stHorizontalBlock"]:has(.doctor-action-card) div[data-testid="column"],
            div[data-testid="stHorizontalBlock"]:has(.doctor-action-link) div[data-testid="column"] {
                margin-bottom: 0.8rem !important;
            }

            .doctor-action-card,
            .doctor-action-link {
                min-height: 88px !important;
                padding: 1rem !important;
            }

            .assistant-panel-header {
                padding: 0.74rem !important;
                align-items: flex-start !important;
            }

            .assistant-mini-brand {
                align-items: flex-start !important;
                gap: 0.65rem !important;
            }

            .assistant-panel-avatar {
                width: 44px !important;
                min-width: 44px !important;
            }

            .assistant-inline-status {
                display: flex !important;
                width: fit-content !important;
                max-width: 100% !important;
            }

            .assistant-launcher-card {
                padding: 0.9rem !important;
                gap: 0.52rem !important;
            }

            .assistant-launcher-tags {
                justify-content: flex-start !important;
                gap: 0.38rem !important;
            }

            .assistant-tag {
                flex: 1 1 72px !important;
                justify-content: center !important;
                padding: 0.34rem 0.46rem !important;
                font-size: 0.7rem !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.assistant-panel-header) div[data-testid="column"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }

            .assistant-chat-history {
                min-height: 260px !important;
                max-height: none !important;
                overflow-y: visible !important;
                padding: 0.65rem !important;
            }

            .assistant-chat-avatar {
                width: 30px !important;
                height: 30px !important;
                border-radius: 10px !important;
                font-size: 0.56rem !important;
            }

            .assistant-chat-bubble {
                max-width: 96% !important;
                padding: 0.68rem 0.74rem !important;
            }

            .assistant-chat-text {
                font-size: 0.86rem !important;
                line-height: 1.5 !important;
            }
        }

        @media (max-width: 430px) {
            .assistant-panel-avatar {
                width: 38px !important;
                min-width: 38px !important;
            }

            .assistant-shell-title,
            .assistant-launcher-title {
                font-size: 0.92rem !important;
            }

            .assistant-shell-copy,
            .assistant-launcher-copy {
                font-size: 0.76rem !important;
                line-height: 1.36 !important;
            }

            .assistant-inline-status {
                font-size: 0.66rem !important;
            }
        }

        @media (min-width: 1025px) {
            body,
            .stApp,
            [data-testid="stAppViewContainer"] {
                overflow-x: auto;
            }

            .block-container {
                width: min(var(--desktop-canvas-width), calc(100vw - var(--sidebar-width) - 4.5rem)) !important;
                min-width: 0 !important;
                max-width: min(var(--desktop-canvas-width), calc(100vw - var(--sidebar-width) - 4.5rem)) !important;
                margin-left: 1rem !important;
            }
        }

        @media (max-width: 1024px), (hover: none) and (pointer: coarse) {
            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"] {
                max-width: 100vw !important;
                overflow-x: hidden !important;
            }

            .stApp::after {
                display: none !important;
            }

            [data-testid="stAppViewContainer"] {
                display: block !important;
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
            }

            [data-testid="stMain"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                margin-left: 0 !important;
            }

            section[data-testid="stSidebar"] {
                position: fixed !important;
                inset: 0 auto 0 0 !important;
                width: min(86vw, 320px) !important;
                min-width: 0 !important;
                max-width: min(86vw, 320px) !important;
                height: 100dvh !important;
                max-height: 100dvh !important;
                z-index: 9999 !important;
                background: transparent !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] > div {
                width: calc(100% - 1.1rem) !important;
                min-width: 0 !important;
                max-width: calc(100% - 1.1rem) !important;
                height: calc(100dvh - 1.1rem) !important;
                max-height: calc(100dvh - 1.1rem) !important;
                margin: 0.55rem !important;
                padding: 0.7rem !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
                border-radius: 24px !important;
                box-sizing: border-box !important;
            }

            [data-testid="stExpandSidebarButton"],
            button[data-testid="stExpandSidebarButton"],
            [data-testid="stSidebarCollapsedControl"],
            button[data-testid="stSidebarCollapsedControl"] {
                top: 0.75rem !important;
                left: 0.75rem !important;
            }

            [data-testid="stSidebarCollapseButton"],
            button[data-testid="stSidebarCollapseButton"] {
                top: 0.85rem !important;
                left: calc(min(86vw, 320px) - 4.95rem) !important;
            }

            .block-container {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 0.75rem !important;
            }

            .topbar-shell {
                margin-top: 0 !important;
            }

            section[data-testid="stSidebar"] [role="radiogroup"] {
                display: flex !important;
                flex-direction: column !important;
                gap: 0.55rem !important;
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label {
                min-width: 0 !important;
                width: 100% !important;
                transform: none !important;
            }

            .patient-stat-grid,
            .patient-card-grid,
            .patient-mini-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .patient-hero-card {
                min-height: auto;
                padding: 1rem;
            }

            .patient-profile-panel {
                margin-top: 0.2rem;
            }
        }

        @media (max-width: 520px) {
            .patient-stat-grid,
            .patient-card-grid,
            .patient-mini-grid {
                grid-template-columns: 1fr;
            }

            .patient-hero-card,
            .patient-profile-panel,
            .patient-chart-panel,
            .patient-mini-panel {
                border-radius: 18px;
            }

            .patient-greeting {
                font-size: 1.14rem;
            }

            .remedy-summary-strip {
                grid-template-columns: 1fr;
            }

            .remedy-table {
                border-radius: 18px;
                background: transparent;
                border: 0;
                box-shadow: none;
                overflow: visible;
            }

            .remedy-care-table,
            .remedy-care-table tbody,
            .remedy-care-table tr,
            .remedy-care-table td {
                display: block;
                width: 100%;
            }

            .remedy-care-table colgroup,
            .remedy-care-table thead {
                display: none;
            }

            .remedy-care-table tr {
                margin-bottom: 0.78rem;
                border-radius: 18px;
                background: linear-gradient(145deg, #ffffff, #eef8ff);
                border: 1px solid rgba(90, 99, 216, 0.12);
                box-shadow: var(--patient-soft-shadow);
                overflow: hidden;
            }

            .remedy-care-table td {
                padding: 0.72rem 0.82rem;
                border-bottom: 1px solid rgba(90, 99, 216, 0.08);
            }

            .remedy-care-table td:last-child {
                border-bottom: 0;
            }

            .remedy-care-table td::before {
                content: attr(data-label);
                display: block;
                margin-bottom: 0.32rem;
                color: var(--patient-blue-dark);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-family: var(--font-data);
                font-size: 0.64rem;
                font-weight: 900;
            }

            .remedy-row {
                grid-template-columns: 1fr;
            }

            .provider-directory-grid {
                grid-template-columns: 1fr;
            }
        }
        
        /* Advanced Dynamic UI/UX for Text Inputs (Login/Register) */
        div[data-testid="stTextInput"] > div[data-baseweb="input"],
        div[data-testid="stTextInput"] > div > div[data-baseweb="input"],
        div[data-baseweb="input"] {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border: 2px solid rgba(15, 118, 110, 0.15) !important;
            border-radius: 12px !important;
            transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03) !important;
            overflow: visible !important;
        }

        div[data-testid="stTextInput"] > div[data-baseweb="input"]:focus-within,
        div[data-testid="stTextInput"] > div > div[data-baseweb="input"]:focus-within,
        div[data-baseweb="input"]:focus-within {
            border: 2px solid var(--teal) !important;
            box-shadow: 0 8px 24px rgba(15, 118, 110, 0.15) !important;
            background-color: #ffffff !important;
            transform: translateY(-2px) !important;
        }

        div[data-testid="stTextInput"] input {
            color: var(--ink) !important;
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            padding: 12px 14px !important;
        }
        div[data-testid="stTextInput"] input::placeholder,
        div[data-baseweb="input"] input::placeholder {
            color: #a0aec0 !important;
            font-weight: 400 !important;
            opacity: 1 !important;
        }
        
        div[data-testid="stTextInput"] label {
            font-weight: 500 !important;
            color: var(--muted) !important;
            margin-bottom: 6px !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.3px !important;
        }
        
        /* Advanced smooth transition for sidebar */
        section[data-testid="stSidebar"] {
            transition: all 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
            box-shadow: 2px 0 20px rgba(0,0,0,0.1) !important;
        }
        [data-testid="stSidebarNav"] {
            transition: opacity 0.4s ease-in-out !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    inject_theme_control_styles()
    if get_active_ui_theme() == "Dark":
        inject_dark_theme_styles()
    inject_sidebar_text_cleanup_styles()


def inject_sidebar_text_cleanup_styles():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] *,
        section[data-testid="stSidebar"] [role="radiogroup"] label p,
        section[data-testid="stSidebar"] [role="radiogroup"] label p *,
        section[data-testid="stSidebar"] [role="radiogroup"] label span:not([data-testid="stIconMaterial"]) {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
            box-shadow: none !important;
            -webkit-box-shadow: none !important;
            outline: 0 !important;
            border: 0 !important;
            text-shadow: none !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"]::selection,
        section[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] *::selection,
        section[data-testid="stSidebar"] [role="radiogroup"] label p::selection,
        section[data-testid="stSidebar"] [role="radiogroup"] label span::selection {
            background: transparent !important;
            color: inherit !important;
            -webkit-text-fill-color: currentColor !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label:focus,
        section[data-testid="stSidebar"] [role="radiogroup"] label:focus-visible,
        section[data-testid="stSidebar"] [role="radiogroup"] label:focus-within {
            outline: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(eyebrow, title, description, chips=None):
    chips_html = ""
    if chips:
        chips_html = "<div class='hero-chip-row'>" + "".join(
            f"<span class='hero-chip'>{html.escape(str(chip))}</span>" for chip in chips
        ) + "</div>"

    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-eyebrow">{html.escape(str(eyebrow))}</div>
            <h2 class="page-title">{html.escape(str(title))}</h2>
            <p class="page-description">{html.escape(str(description))}</p>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title, description, kicker="Feature"):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-kicker">{html.escape(str(kicker))}</div>
            <h4>{html.escape(str(title))}</h4>
            <p>{html.escape(str(description))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(title, description, label="Current Status"):
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{html.escape(str(label))}</div>
            <h4 class="status-title">{html.escape(str(title))}</h4>
            <p class="status-copy">{html.escape(str(description))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_doctor_cards(doctors, per_row=2):
    for start_idx in range(0, len(doctors), per_row):
        doctor_chunk = doctors[start_idx : start_idx + per_row]
        doctor_cols = st.columns(len(doctor_chunk))
        for col, doctor in zip(doctor_cols, doctor_chunk):
            with col:
                render_info_card(
                    doctor["name"],
                    (
                        f"{', '.join(doctor['specialties'])}. "
                        f"{doctor['hospital']}. "
                        f"{doctor['experience_years']} years experience. "
                        f"Fee: INR {doctor['fee_inr']}."
                    ),
                    kicker="Recommended Doctor",
                )


def render_section_intro(title, description):
    st.markdown(f"<div class='section-title'>{html.escape(str(title))}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-copy'>{html.escape(str(description))}</p>", unsafe_allow_html=True)


def apply_warm_compact_plot_style(fig, height=215):
    dark_mode = get_active_ui_theme() == "Dark"
    text_color = "#eef6ff" if dark_mode else "#27304f"
    muted_color = "#9fb0c3" if dark_mode else "#7b83a5"
    grid_color = "rgba(136,160,181,0.18)" if dark_mode else "rgba(90,99,216,0.10)"
    axis_line_color = "rgba(136,160,181,0.20)" if dark_mode else "rgba(90,99,216,0.12)"
    hover_bg = "#111923" if dark_mode else "#27304f"
    plot_bg = "rgba(10,15,22,0.18)" if dark_mode else "rgba(255,255,255,0)"
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=12, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        font=dict(family="Manrope, sans-serif", size=11, color=text_color),
        showlegend=False,
        dragmode=False,
        hoverlabel=dict(bgcolor=hover_bg, font_size=11, font_color="#ffffff"),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        tickfont=dict(size=10, color=muted_color),
        linecolor=axis_line_color,
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zeroline=False,
        fixedrange=True,
        tickfont=dict(size=10, color=muted_color),
    )
    return fig


def render_small_plot_card(title, caption, fig):
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="patient-plot-title">{html.escape(str(title))}</div>
            <p class="patient-plot-caption">{html.escape(str(caption))}</p>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False,
                "doubleClick": False,
            },
        )


def normalize_disease_name(disease_name):
    return " ".join(str(disease_name).strip().split()).lower()


def build_test_list(*items):
    return [{"Test": test_name, "Why It May Be Needed": reason} for test_name, reason in items]


def build_care_profile(
    severity,
    specialist,
    doctor_required,
    doctor_urgency,
    blood_report_required,
    lab_test_required,
    note,
    home_remedies=None,
    medicines=None,
    lab_tests=None,
    warning_signs=None,
):
    return {
        "severity": severity,
        "specialist": specialist,
        "doctor_required": doctor_required,
        "doctor_urgency": doctor_urgency,
        "blood_report_required": blood_report_required,
        "lab_test_required": lab_test_required,
        "note": note,
        "home_remedies": home_remedies or ["Rest", "Hydration", "Balanced light diet"],
        "medicines": medicines or ["Take medicines only after medical advice"],
        "lab_tests": lab_tests or [],
        "warning_signs": warning_signs
        or [
            "Breathing difficulty",
            "Chest pain",
            "Persistent vomiting",
            "Confusion, fainting, or seizures",
            "Bleeding or severe weakness",
        ],
    }


CATEGORY_CARE_DB = {
    "mild_self_care": build_care_profile(
        severity="Mild",
        specialist="General Physician",
        doctor_required="Usually not",
        doctor_urgency="Home care is usually enough if symptoms stay mild.",
        blood_report_required="No",
        lab_test_required="No",
        note="Lab test is not required in routine mild cases. Consult a doctor if fever, breathing trouble, dehydration, or weakness increases.",
        home_remedies=["Rest", "Drink plenty of fluids", "Use a light diet", "Monitor symptoms for 24-48 hours"],
        medicines=["Use only simple symptom relief if appropriate", "Avoid self-starting antibiotics"],
        warning_signs=["High fever lasting more than 3 days", "Breathing trouble", "Poor oral intake", "Drowsiness", "New rash or rapid worsening"],
    ),
    "skin_mild": build_care_profile(
        severity="Mild",
        specialist="Dermatologist",
        doctor_required="Only if not improving",
        doctor_urgency="Book a skin consultation if the rash spreads, becomes painful, or does not improve.",
        blood_report_required="No",
        lab_test_required="No",
        note="Lab test is not required for most mild skin conditions. A doctor review is useful if lesions spread or keep recurring.",
        home_remedies=["Keep skin clean and dry", "Avoid scratching", "Use clean towels and clothing"],
        medicines=["Use creams or tablets only after medical advice", "Avoid steroid creams without a prescription"],
        warning_signs=["Rapidly spreading rash", "Pus or severe pain", "Fever", "Swelling of face or lips", "Skin peeling"],
    ),
    "orthopedic_nonemergency": build_care_profile(
        severity="Mild to Moderate",
        specialist="Orthopedic Doctor",
        doctor_required="Recommended",
        doctor_urgency="Schedule a consultation if pain limits movement or keeps returning.",
        blood_report_required="No",
        lab_test_required="No",
        note="Lab test is not required in most routine joint or spine problems. Imaging or physiotherapy review may be more useful if symptoms persist.",
        home_remedies=["Rest the affected area", "Use posture support", "Gentle stretching if comfortable", "Apply warm compress"],
        medicines=["Use pain medicines only as advised", "Do not overuse anti-inflammatory tablets"],
        warning_signs=["Severe swelling", "Numbness", "Weakness in arm or leg", "Inability to walk", "Loss of bladder or bowel control"],
    ),
    "neuro_nonemergency": build_care_profile(
        severity="Moderate",
        specialist="Neurologist / ENT Specialist",
        doctor_required="Recommended",
        doctor_urgency="Consult a doctor if attacks are frequent, severe, or disabling.",
        blood_report_required="No",
        lab_test_required="No",
        note="Lab test is not required in most routine migraine or positional vertigo cases, but medical review is useful if symptoms are frequent or unusual.",
        home_remedies=["Rest in a quiet room", "Stay hydrated", "Avoid known triggers", "Rise slowly from bed or chairs"],
        medicines=["Use doctor-approved symptom relief only", "Avoid self-medicating repeatedly"],
        warning_signs=["Sudden severe headache", "Double vision", "New weakness", "Repeated vomiting", "Loss of consciousness"],
    ),
    "digestive_nonemergency": build_care_profile(
        severity="Moderate",
        specialist="General Physician / Gastroenterologist",
        doctor_required="Recommended if persistent",
        doctor_urgency="Consult a doctor if pain, acidity, bleeding, or vomiting keeps recurring.",
        blood_report_required="No",
        lab_test_required="No",
        note="Lab test is not required for many mild digestive conditions. If symptoms persist or there is bleeding, a doctor may suggest further evaluation.",
        home_remedies=["Eat light meals", "Avoid spicy and oily foods", "Drink water in small sips", "Do not lie down immediately after meals"],
        medicines=["Use antacids or symptom relief only if appropriate", "Avoid painkillers that irritate the stomach"],
        warning_signs=["Black stool", "Blood in vomit", "Severe dehydration", "Persistent abdominal pain", "Unexplained weight loss"],
    ),
    "skin_chronic": build_care_profile(
        severity="Moderate",
        specialist="Dermatologist",
        doctor_required="Yes",
        doctor_urgency="Book a dermatology visit if flare-ups are frequent or large skin areas are involved.",
        blood_report_required="Depends",
        lab_test_required="Depends",
        note="Routine lab tests are often not required, but a dermatologist may order tests before long-term treatment.",
        home_remedies=["Keep skin moisturized", "Avoid harsh soaps", "Track triggers", "Manage stress"],
        medicines=["Use ointments only as prescribed", "Do not abruptly stop prescription creams"],
        warning_signs=["Painful cracked skin", "Fever", "Rapid worsening", "Joint swelling", "Signs of skin infection"],
    ),
}


DISEASE_CATEGORY_MAP = {
    normalize_disease_name("Acne"): "skin_mild",
    normalize_disease_name("Allergy"): "mild_self_care",
    normalize_disease_name("Arthritis"): "orthopedic_nonemergency",
    normalize_disease_name("Cervical spondylosis"): "orthopedic_nonemergency",
    normalize_disease_name("Chicken pox"): "mild_self_care",
    normalize_disease_name("Common Cold"): "mild_self_care",
    normalize_disease_name("Dimorphic hemmorhoids(piles)"): "digestive_nonemergency",
    normalize_disease_name("Fungal infection"): "skin_mild",
    normalize_disease_name("GERD"): "digestive_nonemergency",
    normalize_disease_name("(vertigo) Paroymsal  Positional Vertigo"): "neuro_nonemergency",
    normalize_disease_name("Impetigo"): "skin_mild",
    normalize_disease_name("Migraine"): "neuro_nonemergency",
    normalize_disease_name("Osteoarthristis"): "orthopedic_nonemergency",
    normalize_disease_name("Peptic ulcer diseae"): "digestive_nonemergency",
    normalize_disease_name("Psoriasis"): "skin_chronic",
}


DISEASE_CARE_DB = {
    normalize_disease_name("Dengue"): build_care_profile(
        severity="Serious",
        specialist="General Physician / Internal Medicine",
        doctor_required="Yes",
        doctor_urgency="Doctor review is needed the same day, especially for fever, rash, bleeding, or weakness.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="This condition often needs confirmatory blood testing and monitoring of platelet count and hydration status.",
        home_remedies=["Drink ORS and water", "Take proper rest", "Use light diet"],
        medicines=["Paracetamol if advised", "Avoid aspirin and ibuprofen unless a doctor says otherwise"],
        lab_tests=build_test_list(
            ("Complete blood count (CBC) with platelet count", "Checks platelet fall and infection pattern."),
            ("Hematocrit", "Helps detect plasma leakage and dehydration."),
            ("Dengue NS1 antigen", "Useful for early confirmation."),
            ("Dengue IgM / IgG", "Supports confirmation in later stages."),
            ("Liver function test", "Looks for liver involvement in moderate to severe cases."),
        ),
        warning_signs=["Bleeding gums or nose", "Severe abdominal pain", "Repeated vomiting", "Extreme weakness", "Black stool"],
    ),
    normalize_disease_name("Malaria"): build_care_profile(
        severity="Serious",
        specialist="General Physician / Infectious Disease Specialist",
        doctor_required="Yes",
        doctor_urgency="Consult a doctor the same day because confirmation and prescription treatment are important.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Malaria treatment should follow confirmatory testing and medical supervision.",
        home_remedies=["Hydration", "Rest", "Monitor fever pattern"],
        medicines=["Antimalarial medicines only by doctor prescription"],
        lab_tests=build_test_list(
            ("Peripheral blood smear", "Looks for malaria parasites directly."),
            ("Rapid malaria antigen test", "Provides faster confirmation."),
            ("Complete blood count (CBC)", "Checks anemia and platelet changes."),
            ("Liver and kidney function tests", "Assesses organ stress in moderate or severe disease."),
        ),
        warning_signs=["Confusion", "Breathing trouble", "Severe chills", "Yellow eyes", "Reduced urine output"],
    ),
    normalize_disease_name("Typhoid"): build_care_profile(
        severity="Serious",
        specialist="General Physician / Gastroenterologist",
        doctor_required="Yes",
        doctor_urgency="Book a doctor visit promptly for confirmation and antibiotics.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Typhoid commonly needs lab confirmation before treatment is finalized.",
        home_remedies=["Hydration", "Soft diet", "Complete rest"],
        medicines=["Antibiotics only by doctor prescription"],
        lab_tests=build_test_list(
            ("Complete blood count (CBC)", "Checks infection severity and dehydration impact."),
            ("Blood culture", "Best confirms the typhoid organism."),
            ("Widal or Typhidot test", "May support the diagnosis where used locally."),
            ("Stool culture", "Can help detect infection in some cases."),
            ("Liver function test", "Useful if illness is prolonged or severe."),
        ),
        warning_signs=["Persistent high fever", "Severe weakness", "Confusion", "Abdominal swelling", "Blood in stool"],
    ),
    normalize_disease_name("Pneumonia"): build_care_profile(
        severity="Serious",
        specialist="Pulmonologist / General Physician",
        doctor_required="Yes",
        doctor_urgency="Prompt medical review is needed, especially if there is cough, fever, chest pain, or breathing difficulty.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="This prediction should be clinically confirmed because pneumonia can worsen quickly.",
        home_remedies=["Rest", "Warm fluids", "Steam inhalation only if comfortable"],
        medicines=["Antibiotics only if prescribed", "Fever medicines as advised"],
        lab_tests=build_test_list(
            ("Complete blood count (CBC)", "Checks infection severity."),
            ("C-reactive protein (CRP)", "Helps assess inflammation."),
            ("Chest X-ray", "Confirms lung involvement."),
            ("Sputum culture", "Helps identify the germ in selected cases."),
            ("Pulse oximetry", "Checks oxygen level and need for urgent care."),
        ),
        warning_signs=["Shortness of breath", "Blue lips", "Confusion", "High fever", "Low oxygen level"],
    ),
    normalize_disease_name("Bronchial Asthma"): build_care_profile(
        severity="Moderate to Serious",
        specialist="Pulmonologist",
        doctor_required="Yes",
        doctor_urgency="Consult a doctor soon, and seek urgent care if breathing becomes difficult or inhaler relief is poor.",
        blood_report_required="Depends",
        lab_test_required="Yes",
        note="Breathing symptoms should be clinically reviewed. Testing helps assess severity and long-term control.",
        home_remedies=["Avoid dust and smoke", "Use clean air", "Follow breathing exercises if previously advised"],
        medicines=["Use prescribed rescue inhaler", "Use controller inhaler only as directed"],
        lab_tests=build_test_list(
            ("Spirometry", "Measures airflow limitation."),
            ("Peak flow measurement", "Tracks airway narrowing over time."),
            ("Pulse oximetry", "Checks oxygen status during attacks."),
            ("CBC with eosinophils", "May support allergy-related evaluation."),
            ("Chest X-ray", "Used when symptoms are atypical or severe."),
        ),
        warning_signs=["Unable to speak full sentences", "Bluish lips", "Severe wheezing", "Fast breathing", "Poor relief after inhaler use"],
    ),
    normalize_disease_name("Heart attack"): build_care_profile(
        severity="Emergency",
        specialist="Emergency Physician / Cardiologist",
        doctor_required="Emergency",
        doctor_urgency="Go to the emergency department immediately. Do not wait at home.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="This is a medical emergency and the app should never be used as a delay before emergency care.",
        home_remedies=["No home remedy delay"],
        medicines=["Emergency treatment only under hospital care"],
        lab_tests=build_test_list(
            ("ECG", "Looks for acute heart muscle injury."),
            ("Troponin blood test", "Confirms heart muscle damage."),
            ("CK-MB", "Supports cardiac injury assessment in selected cases."),
            ("Echocardiogram", "Checks pumping function and complications."),
            ("Coronary evaluation", "Helps plan urgent treatment when needed."),
        ),
        warning_signs=["Chest pressure", "Sweating", "Jaw or arm pain", "Shortness of breath", "Collapse"],
    ),
    normalize_disease_name("Paralysis (brain hemorrhage)"): build_care_profile(
        severity="Emergency",
        specialist="Emergency Physician / Neurologist",
        doctor_required="Emergency",
        doctor_urgency="Go to emergency care immediately because stroke-like symptoms are time-sensitive.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Urgent hospital evaluation is required. This prediction should trigger emergency assessment, not home monitoring.",
        home_remedies=["Do not delay emergency care"],
        medicines=["Emergency treatment only under hospital care"],
        lab_tests=build_test_list(
            ("CT scan or MRI brain", "Confirms bleeding and location."),
            ("Complete blood count (CBC)", "Checks blood count before treatment."),
            ("Coagulation profile", "Checks bleeding and clotting status."),
            ("Blood sugar", "Rules out glucose-related stroke mimic."),
            ("Blood pressure monitoring", "Guides emergency treatment."),
        ),
        warning_signs=["Face drooping", "Arm weakness", "Speech difficulty", "Sudden confusion", "Sudden severe headache"],
    ),
    normalize_disease_name("Diabetes "): build_care_profile(
        severity="Moderate",
        specialist="General Physician / Endocrinologist",
        doctor_required="Yes",
        doctor_urgency="Book a doctor consultation to confirm the diagnosis and start a management plan.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood reports are important because diabetes needs confirmation and long-term monitoring.",
        home_remedies=["Limit sugary drinks", "Stay hydrated", "Follow a balanced meal plan", "Track symptoms"],
        medicines=["Start medicines only after doctor advice"],
        lab_tests=build_test_list(
            ("Fasting blood sugar", "Checks baseline glucose level."),
            ("Postprandial blood sugar", "Measures glucose after meals."),
            ("HbA1c", "Shows average sugar control over about 3 months."),
            ("Urine sugar and ketones", "Screens for uncontrolled diabetes."),
            ("Kidney function test", "Checks complications and medicine safety."),
        ),
        warning_signs=["Confusion", "Vomiting", "Severe thirst", "Very frequent urination", "Rapid breathing"],
    ),
    normalize_disease_name("Hypertension "): build_care_profile(
        severity="Moderate",
        specialist="General Physician / Cardiologist",
        doctor_required="Yes",
        doctor_urgency="Doctor confirmation is needed because long-term monitoring matters even if you feel well.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood tests and heart-kidney evaluation are commonly used when high blood pressure is suspected.",
        home_remedies=["Reduce salt intake", "Stay active if able", "Sleep well", "Track blood pressure readings"],
        medicines=["Take blood pressure medicines only as prescribed"],
        lab_tests=build_test_list(
            ("Blood pressure monitoring", "Confirms whether readings stay high."),
            ("Kidney function test", "Checks whether kidneys are affected."),
            ("Lipid profile", "Assesses cardiovascular risk."),
            ("Urine routine / albumin", "Screens for kidney impact."),
            ("ECG", "Looks for heart strain."),
        ),
        warning_signs=["Very severe headache", "Chest pain", "Breathing trouble", "Vision changes", "Neurological weakness"],
    ),
    normalize_disease_name("Hyperthyroidism"): build_care_profile(
        severity="Moderate",
        specialist="Endocrinologist",
        doctor_required="Yes",
        doctor_urgency="Book a medical review because confirmation needs thyroid testing.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Thyroid symptoms need blood tests to confirm the exact disorder and decide treatment.",
        home_remedies=["Avoid stimulants if they worsen symptoms", "Stay hydrated", "Track palpitations and weight change"],
        medicines=["Start thyroid medicines only after medical advice"],
        lab_tests=build_test_list(
            ("TSH", "Main screening test for thyroid imbalance."),
            ("Free T4", "Measures active thyroid hormone."),
            ("Free T3", "Helps assess overactive thyroid."),
            ("Thyroid antibody tests", "May identify autoimmune thyroid disease."),
            ("ECG", "Useful when palpitations are present."),
        ),
        warning_signs=["Fast heartbeat", "Chest pain", "Severe weight loss", "Tremors", "Breathlessness"],
    ),
    normalize_disease_name("Hypothyroidism"): build_care_profile(
        severity="Moderate",
        specialist="Endocrinologist",
        doctor_required="Yes",
        doctor_urgency="Doctor consultation is advised because thyroid disorders need blood confirmation.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood reports help confirm hypothyroidism and guide dosing.",
        home_remedies=["Take adequate rest", "Eat balanced meals", "Track fatigue and weight change"],
        medicines=["Use thyroid medicine only after doctor confirmation"],
        lab_tests=build_test_list(
            ("TSH", "Main screening test for low thyroid function."),
            ("Free T4", "Confirms the extent of thyroid hormone deficiency."),
            ("Thyroid antibody tests", "May help identify autoimmune causes."),
            ("Lipid profile", "Checks cholesterol changes related to low thyroid."),
        ),
        warning_signs=["Extreme drowsiness", "Swelling", "Slow pulse", "Confusion", "Breathing difficulty"],
    ),
    normalize_disease_name("Hypoglycemia"): build_care_profile(
        severity="Serious",
        specialist="General Physician / Endocrinologist",
        doctor_required="Yes",
        doctor_urgency="Prompt review is recommended because low sugar can become dangerous quickly.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood glucose confirmation is important, especially if symptoms are recurrent or severe.",
        home_remedies=["Take quick sugar only if you are fully alert", "Eat a follow-up snack once stable", "Do not stay alone if symptoms are severe"],
        medicines=["Medicine adjustment must be done by a doctor"],
        lab_tests=build_test_list(
            ("Random blood sugar", "Confirms low glucose at the time of symptoms."),
            ("HbA1c", "Checks overall sugar control if diabetes is present."),
            ("Insulin and C-peptide", "Used in recurrent or unexplained episodes."),
            ("Kidney and liver function tests", "Looks for causes that can worsen low sugar."),
        ),
        warning_signs=["Sweating with confusion", "Fainting", "Seizure", "Inability to eat", "Repeated low-sugar episodes"],
    ),
    normalize_disease_name("Urinary tract infection"): build_care_profile(
        severity="Moderate",
        specialist="General Physician / Urologist",
        doctor_required="Yes",
        doctor_urgency="Consult a doctor because urine testing guides the right treatment.",
        blood_report_required="Depends",
        lab_test_required="Yes",
        note="Urine tests are usually required, especially if there is burning urination, fever, or back pain.",
        home_remedies=["Drink more water", "Do not hold urine", "Maintain hygiene"],
        medicines=["Antibiotics only by doctor prescription"],
        lab_tests=build_test_list(
            ("Urine routine and microscopy", "Looks for pus cells, bacteria, and blood."),
            ("Urine culture", "Identifies the germ and best antibiotic."),
            ("Complete blood count (CBC)", "Checks if the infection is spreading in feverish cases."),
            ("Kidney function test", "Useful if infection is severe or recurrent."),
        ),
        warning_signs=["Fever", "Back or flank pain", "Vomiting", "Blood in urine", "Reduced urine output"],
    ),
    normalize_disease_name("Tuberculosis"): build_care_profile(
        severity="Serious",
        specialist="Chest Physician / Infectious Disease Specialist",
        doctor_required="Yes",
        doctor_urgency="Prompt doctor review is needed because confirmation and long-term treatment are required.",
        blood_report_required="Depends",
        lab_test_required="Yes",
        note="Tuberculosis needs confirmatory testing and a supervised treatment plan.",
        home_remedies=["Rest", "Nutritious diet", "Use cough hygiene", "Avoid close exposure to others until evaluated"],
        medicines=["Anti-tuberculosis medicines only under doctor supervision"],
        lab_tests=build_test_list(
            ("Sputum AFB test", "Checks for tuberculosis bacteria."),
            ("GeneXpert / CBNAAT", "Confirms TB and screens drug resistance."),
            ("Chest X-ray", "Assesses lung involvement."),
            ("Complete blood count (CBC)", "Provides supportive information."),
            ("ESR", "Helps assess inflammation in some settings."),
        ),
        warning_signs=["Coughing blood", "Weight loss", "Persistent fever", "Breathing difficulty", "Severe weakness"],
    ),
    normalize_disease_name("AIDS"): build_care_profile(
        severity="Serious",
        specialist="Infectious Disease Specialist",
        doctor_required="Yes",
        doctor_urgency="Doctor consultation is required for confirmation, staging, and treatment planning.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="This condition requires formal testing, confirmation, and specialist follow-up.",
        home_remedies=["Good nutrition", "Hydration", "Avoid missing medical follow-up"],
        medicines=["Treatment must be prescribed by an HIV specialist"],
        lab_tests=build_test_list(
            ("HIV antigen / antibody test", "Confirms HIV infection."),
            ("CD4 count", "Shows immune system strength."),
            ("HIV viral load", "Measures active virus in the blood."),
            ("Complete blood count (CBC)", "Checks baseline blood counts."),
            ("Liver and kidney function tests", "Guides treatment safety."),
        ),
        warning_signs=["Severe weight loss", "Persistent fever", "Shortness of breath", "Severe diarrhea", "Confusion"],
    ),
    normalize_disease_name("Alcoholic hepatitis"): build_care_profile(
        severity="Serious",
        specialist="Gastroenterologist / Hepatologist",
        doctor_required="Yes",
        doctor_urgency="Doctor review is needed promptly, especially with jaundice, vomiting, abdominal swelling, or confusion.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Liver disease should be medically evaluated because blood reports and imaging guide severity assessment.",
        home_remedies=["Avoid alcohol completely", "Use a soft balanced diet", "Stay hydrated"],
        medicines=["Use medicines only after liver specialist advice"],
        lab_tests=build_test_list(
            ("Liver function test", "Measures bilirubin and liver enzyme injury."),
            ("Complete blood count (CBC)", "Checks infection, anemia, and platelet count."),
            ("Prothrombin time / INR", "Shows liver-related clotting risk."),
            ("Ultrasound abdomen", "Assesses liver size and complications."),
            ("Kidney function test", "Important in advanced liver disease."),
        ),
        warning_signs=["Yellow eyes", "Abdominal swelling", "Confusion", "Bleeding", "Vomiting blood"],
    ),
    normalize_disease_name("Chronic cholestasis"): build_care_profile(
        severity="Serious",
        specialist="Gastroenterologist / Hepatologist",
        doctor_required="Yes",
        doctor_urgency="Consult a doctor because bile-flow problems usually need medical evaluation and testing.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood tests and imaging are usually required to identify the cause of cholestasis.",
        home_remedies=["Eat light meals", "Avoid alcohol", "Stay hydrated"],
        medicines=["Start medicines only after medical advice"],
        lab_tests=build_test_list(
            ("Liver function test", "Checks bilirubin and cholestatic enzymes."),
            ("Ultrasound abdomen", "Looks for obstruction or liver disease."),
            ("Complete blood count (CBC)", "Checks infection and overall status."),
            ("Prothrombin time / INR", "Assesses liver-related clotting function."),
            ("Viral hepatitis markers", "Rules out infectious causes when needed."),
        ),
        warning_signs=["Yellow eyes", "Dark urine", "Severe itching", "Pale stool", "Abdominal swelling"],
    ),
    normalize_disease_name("Jaundice"): build_care_profile(
        severity="Serious",
        specialist="General Physician / Gastroenterologist",
        doctor_required="Yes",
        doctor_urgency="Prompt medical review is needed because jaundice has many important causes that need testing.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood report and liver evaluation are usually required when jaundice is suspected.",
        home_remedies=["Drink fluids", "Avoid alcohol", "Use light meals", "Rest well"],
        medicines=["Use medicines only after medical advice"],
        lab_tests=build_test_list(
            ("Liver function test", "Measures bilirubin and liver injury."),
            ("Complete blood count (CBC)", "Looks for infection or anemia."),
            ("Viral hepatitis panel", "Checks common infectious causes."),
            ("Ultrasound abdomen", "Helps find obstruction or liver enlargement."),
            ("Prothrombin time / INR", "Assesses liver severity."),
        ),
        warning_signs=["Confusion", "Bleeding", "Abdominal swelling", "Severe vomiting", "Reduced appetite with worsening weakness"],
    ),
    normalize_disease_name("hepatitis A"): build_care_profile(
        severity="Serious",
        specialist="General Physician / Gastroenterologist",
        doctor_required="Yes",
        doctor_urgency="Doctor consultation is recommended for confirmation and liver function monitoring.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Hepatitis symptoms need liver-related blood testing and follow-up.",
        home_remedies=["Rest", "Hydration", "Avoid alcohol", "Use simple meals"],
        medicines=["Avoid liver-toxic medicines unless prescribed"],
        lab_tests=build_test_list(
            ("Liver function test", "Measures liver inflammation and bilirubin."),
            ("Hepatitis A IgM", "Confirms recent hepatitis A infection."),
            ("Complete blood count (CBC)", "Provides baseline health status."),
            ("Prothrombin time / INR", "Assesses liver severity when needed."),
        ),
        warning_signs=["Yellow eyes", "Persistent vomiting", "Confusion", "Bleeding", "Severe abdominal pain"],
    ),
    normalize_disease_name("Hepatitis B"): build_care_profile(
        severity="Serious",
        specialist="Gastroenterologist / Hepatologist",
        doctor_required="Yes",
        doctor_urgency="Doctor consultation is needed for confirmation and liver assessment.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood reports are required to confirm hepatitis B and check liver involvement.",
        home_remedies=["Rest", "Hydration", "Avoid alcohol", "Use a balanced light diet"],
        medicines=["Treatment decisions must be made by a specialist"],
        lab_tests=build_test_list(
            ("HBsAg and hepatitis B markers", "Confirms hepatitis B infection."),
            ("Liver function test", "Checks the extent of liver injury."),
            ("HBV DNA viral load", "Measures active viral replication."),
            ("Complete blood count (CBC)", "Provides baseline blood counts."),
            ("Ultrasound abdomen", "Screens liver structure and complications."),
        ),
        warning_signs=["Yellow eyes", "Bleeding", "Abdominal swelling", "Confusion", "Severe weakness"],
    ),
    normalize_disease_name("Hepatitis C"): build_care_profile(
        severity="Serious",
        specialist="Gastroenterologist / Hepatologist",
        doctor_required="Yes",
        doctor_urgency="Doctor consultation is needed for confirmation and treatment planning.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Hepatitis C needs confirmatory blood tests and specialist follow-up.",
        home_remedies=["Avoid alcohol", "Stay hydrated", "Use balanced meals"],
        medicines=["Antiviral treatment must be prescribed by a specialist"],
        lab_tests=build_test_list(
            ("Hepatitis C antibody test", "Screens for hepatitis C exposure."),
            ("HCV RNA", "Confirms active infection."),
            ("Liver function test", "Checks liver inflammation and damage."),
            ("Complete blood count (CBC)", "Provides baseline health information."),
            ("Ultrasound abdomen", "Looks for chronic liver changes."),
        ),
        warning_signs=["Yellow eyes", "Abdominal swelling", "Bleeding", "Confusion", "Severe fatigue"],
    ),
    normalize_disease_name("Hepatitis D"): build_care_profile(
        severity="Serious",
        specialist="Gastroenterologist / Hepatologist",
        doctor_required="Yes",
        doctor_urgency="Prompt specialist review is needed because hepatitis D can worsen liver disease.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood tests are required to confirm hepatitis D and assess liver function.",
        home_remedies=["Avoid alcohol", "Rest", "Balanced light diet"],
        medicines=["Use medicines only after specialist review"],
        lab_tests=build_test_list(
            ("Hepatitis D antibody / RNA test", "Confirms hepatitis D infection."),
            ("Hepatitis B markers", "Hepatitis D is linked with hepatitis B."),
            ("Liver function test", "Assesses liver injury."),
            ("Complete blood count (CBC)", "Provides baseline blood status."),
            ("Prothrombin time / INR", "Checks liver-related clotting status."),
        ),
        warning_signs=["Yellow eyes", "Abdominal swelling", "Bleeding", "Confusion", "Severe weakness"],
    ),
    normalize_disease_name("Hepatitis E"): build_care_profile(
        severity="Serious",
        specialist="General Physician / Gastroenterologist",
        doctor_required="Yes",
        doctor_urgency="Doctor consultation is advised for confirmation and hydration/liver monitoring.",
        blood_report_required="Yes",
        lab_test_required="Yes",
        note="Blood reports are commonly used to confirm hepatitis E and assess severity.",
        home_remedies=["Rest", "Hydration", "Avoid alcohol", "Use light meals"],
        medicines=["Avoid liver-toxic medicines unless prescribed"],
        lab_tests=build_test_list(
            ("Liver function test", "Checks bilirubin and liver enzyme elevation."),
            ("Hepatitis E IgM", "Confirms recent hepatitis E infection."),
            ("Complete blood count (CBC)", "Provides general health status."),
            ("Prothrombin time / INR", "Assesses liver severity when needed."),
        ),
        warning_signs=["Yellow eyes", "Persistent vomiting", "Confusion", "Bleeding", "Severe weakness"],
    ),
    normalize_disease_name("Gastroenteritis"): build_care_profile(
        severity="Moderate",
        specialist="General Physician",
        doctor_required="Recommended",
        doctor_urgency="Consult a doctor if vomiting, diarrhea, fever, or dehydration is moderate to severe.",
        blood_report_required="Depends",
        lab_test_required="Depends",
        note="Lab test is not required in many mild cases, but stool or blood tests may be needed if dehydration, fever, or prolonged illness is present.",
        home_remedies=["Drink ORS", "Use light foods", "Rest", "Avoid oily and spicy food"],
        medicines=["Use anti-vomiting or anti-diarrheal medicines only if advised"],
        lab_tests=build_test_list(
            ("Stool routine", "May help if diarrhea is severe or prolonged."),
            ("Complete blood count (CBC)", "Checks infection severity and dehydration effect."),
            ("Electrolyte panel", "Looks for dehydration-related imbalance."),
        ),
        warning_signs=["Very low urine output", "Persistent vomiting", "High fever", "Blood in stool", "Severe weakness"],
    ),
    normalize_disease_name("Drug Reaction"): build_care_profile(
        severity="Moderate",
        specialist="General Physician / Dermatologist",
        doctor_required="Yes",
        doctor_urgency="Doctor review is recommended because some drug reactions can worsen quickly.",
        blood_report_required="Depends",
        lab_test_required="Depends",
        note="Many mild drug rashes do not need lab testing, but severe reactions may require blood tests and urgent care.",
        home_remedies=["Stop only the suspected medicine if a doctor has already told you to do so", "Avoid scratching", "Stay hydrated"],
        medicines=["Use anti-allergy treatment only if appropriate and safe", "Seek medical advice before restarting medicines"],
        lab_tests=build_test_list(
            ("Complete blood count (CBC)", "Used when the reaction is widespread, feverish, or severe."),
            ("Liver function test", "Looks for drug-related liver involvement."),
            ("Kidney function test", "Checks for severe systemic drug reaction."),
        ),
        warning_signs=["Lip or tongue swelling", "Breathing trouble", "Skin peeling", "High fever", "Blisters"],
    ),
    normalize_disease_name("Varicose veins"): build_care_profile(
        severity="Moderate",
        specialist="Vascular Surgeon",
        doctor_required="Recommended",
        doctor_urgency="Doctor consultation is useful if there is leg swelling, pain, or skin changes.",
        blood_report_required="No",
        lab_test_required="Depends",
        note="Routine blood reports are not usually required. A vascular assessment or Doppler scan may be advised if symptoms are troublesome.",
        home_remedies=["Elevate legs", "Walk regularly", "Avoid long standing", "Use compression stockings if advised"],
        medicines=["Use pain relief only if needed and safe"],
        lab_tests=build_test_list(
            ("Venous Doppler ultrasound", "Checks blood flow and valve problems in leg veins."),
        ),
        warning_signs=["Sudden leg swelling", "Severe pain", "Skin ulcer", "Red hot vein", "Breathlessness"],
    ),
}


def get_disease_care_plan(predicted_disease, confidence):
    normalized_name = normalize_disease_name(predicted_disease)
    default_profile = build_care_profile(
        severity="Moderate",
        specialist="General Physician",
        doctor_required="Recommended",
        doctor_urgency="Book a doctor review if symptoms continue or worsen.",
        blood_report_required="Depends",
        lab_test_required="Depends",
        note="Use this prediction as screening support only. A doctor should confirm the diagnosis and decide whether reports or tests are needed.",
    )

    category_key = DISEASE_CATEGORY_MAP.get(normalized_name)
    category_profile = CATEGORY_CARE_DB.get(category_key, {})
    disease_profile = DISEASE_CARE_DB.get(normalized_name, {})

    care_plan = {**default_profile, **category_profile, **disease_profile}

    if confidence < 50:
        care_plan["note"] = (
            "The prediction may overlap with other diseases. "
            + care_plan["note"]
            + " If symptoms do not match the prediction, please get a doctor review."
        )
        if care_plan["doctor_required"] == "Usually not":
            care_plan["doctor_required"] = "Recommended if symptoms continue"

    if care_plan["lab_test_required"] == "Yes" and not care_plan["lab_tests"]:
        care_plan["lab_tests"] = build_test_list(
            ("Complete blood count (CBC)", "Checks infection, inflammation, and anemia."),
            ("Basic metabolic panel", "Assesses hydration, kidney function, and electrolytes."),
        )

    return care_plan


def appointment_row_to_dict(row):
    try:
        symptoms = json.loads(row["symptoms_json"] or "[]")
    except json.JSONDecodeError:
        symptoms = []

    return {
        "appointment_id": row["appointment_id"],
        "username": row["username"],
        "patient_name": row["patient_name"],
        "predicted_disease": row["predicted_disease"],
        "specialist": row["specialist"],
        "doctor_name": row["doctor_name"],
        "hospital": row["hospital"],
        "consultation_mode": row["consultation_mode"],
        "appointment_date": row["appointment_date"],
        "appointment_slot": row["appointment_slot"],
        "status": row["status"],
        "booked_at": row["booked_at"],
        "symptoms": symptoms,
        "reason": row["reason"],
    }


def load_appointments_from_mysql():
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT appointment_id, username, patient_name, predicted_disease, specialist,
                   doctor_name, hospital, consultation_mode, appointment_date,
                   appointment_slot, status, booked_at, symptoms_json, reason
            FROM appointments
            ORDER BY appointment_date DESC, appointment_slot DESC, booked_at DESC
            """
        )
        return [appointment_row_to_dict(row) for row in cursor.fetchall()]
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def save_appointments_to_mysql(appointments):
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        for appointment in appointments:
            appointment_id = str(appointment.get("appointment_id", "")).strip()
            username = str(appointment.get("username", "")).strip().lower()
            if not appointment_id or not username:
                continue
            cursor.execute(
                """
                INSERT INTO appointments (
                    appointment_id, username, patient_name, predicted_disease, specialist,
                    doctor_name, hospital, consultation_mode, appointment_date,
                    appointment_slot, status, booked_at, symptoms_json, reason, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON DUPLICATE KEY UPDATE
                    username = VALUES(username),
                    patient_name = VALUES(patient_name),
                    predicted_disease = VALUES(predicted_disease),
                    specialist = VALUES(specialist),
                    doctor_name = VALUES(doctor_name),
                    hospital = VALUES(hospital),
                    consultation_mode = VALUES(consultation_mode),
                    appointment_date = VALUES(appointment_date),
                    appointment_slot = VALUES(appointment_slot),
                    status = VALUES(status),
                    booked_at = VALUES(booked_at),
                    symptoms_json = VALUES(symptoms_json),
                    reason = VALUES(reason),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    appointment_id,
                    username,
                    str(appointment.get("patient_name", "")).strip(),
                    str(appointment.get("predicted_disease", "")).strip(),
                    str(appointment.get("specialist", "")).strip(),
                    str(appointment.get("doctor_name", "")).strip(),
                    str(appointment.get("hospital", "")).strip(),
                    str(appointment.get("consultation_mode", "In-person")).strip(),
                    str(appointment.get("appointment_date", "")).strip(),
                    str(appointment.get("appointment_slot", "")).strip(),
                    str(appointment.get("status", "Booked")).strip() or "Booked",
                    str(appointment.get("booked_at", "")).strip(),
                    json.dumps(appointment.get("symptoms", [])),
                    str(appointment.get("reason", "")).strip(),
                ),
            )
        conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def load_appointments():
    ensure_database_ready()
    return load_appointments_from_mysql()


def save_appointments(appointments):
    ensure_database_ready()
    save_appointments_to_mysql(appointments)


def save_appointment(appointment):
    save_appointments([appointment])


def is_appointment_slot_taken_in_mysql(doctor_name, appointment_date, appointment_slot):
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1
            FROM appointments
            WHERE lower(trim(doctor_name)) = lower(trim(%s))
              AND appointment_date = %s
              AND lower(trim(appointment_slot)) = lower(trim(%s))
              AND status = 'Booked'
            LIMIT 1
            """,
            (doctor_name, appointment_date, appointment_slot),
        )
        return cursor.fetchone() is not None
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def is_appointment_slot_taken(doctor_name, appointment_date, appointment_slot):
    ensure_database_ready()
    return is_appointment_slot_taken_in_mysql(doctor_name, appointment_date, appointment_slot)


def cancel_appointment_for_user_in_mysql(appointment_id, username):
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE appointments
            SET status = 'Cancelled',
                updated_at = CURRENT_TIMESTAMP
            WHERE appointment_id = %s
              AND username = %s
              AND status = 'Booked'
            """,
            (appointment_id, username),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def cancel_appointment_for_user(appointment_id, username):
    ensure_database_ready()
    return cancel_appointment_for_user_in_mysql(appointment_id, username)


def split_specialists(specialist_text):
    return [part.strip() for part in str(specialist_text).split("/") if part.strip()]


def build_appointment_id():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"APT-{timestamp}"


def build_lab_appointment_id():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"LAB-{timestamp}"


LAB_TEST_PRICE_RULES = (
    ("ct scan", 3500),
    ("mri", 6500),
    ("blood culture", 1400),
    ("troponin", 1200),
    ("dengue igm", 1100),
    ("dengue ns1", 900),
    ("typhidot", 900),
    ("sputum culture", 900),
    ("coagulation", 900),
    ("spirometry", 900),
    ("liver function", 800),
    ("kidney function", 750),
    ("c reactive", 700),
    ("crp", 700),
    ("rapid malaria", 650),
    ("widal", 600),
    ("chest x ray", 600),
    ("hba1c", 500),
    ("complete blood count", 450),
    ("cbc", 450),
    ("platelet", 350),
    ("urine", 250),
    ("blood sugar", 150),
    ("glucose", 150),
    ("pulse oximetry", 150),
    ("blood pressure", 100),
)


def get_lab_test_price(test_name):
    normalized_name = normalize_text_for_matching(test_name)
    for keyword, price in LAB_TEST_PRICE_RULES:
        if keyword in normalized_name:
            return float(price)
    return 500.0


def normalize_lab_test_item(item):
    test_name = clean_provider_text(item.get("Test") if isinstance(item, dict) else item)
    if not test_name:
        return {}
    return {
        "Test": test_name,
        "Why It May Be Needed": clean_provider_text(
            item.get("Why It May Be Needed", "") if isinstance(item, dict) else "",
            "Supports confirmation or severity review.",
        ),
        "Estimated Price": get_lab_test_price(test_name),
    }


def build_lab_test_catalog(care_plan):
    if not care_plan:
        return []

    catalog = []
    seen_tests = set()
    for item in care_plan.get("lab_tests", []) or []:
        normalized_item = normalize_lab_test_item(item)
        if not normalized_item:
            continue
        key = normalize_text_for_matching(normalized_item["Test"])
        if key in seen_tests:
            continue
        seen_tests.add(key)
        catalog.append(normalized_item)

    if not catalog and lab_tests_are_required(care_plan):
        for item in build_test_list(
            ("Complete blood count (CBC)", "General screening for infection, anemia, and inflammation."),
            ("Basic metabolic panel", "Reviews hydration, kidney function, electrolytes, and sugar balance."),
        ):
            normalized_item = normalize_lab_test_item(item)
            key = normalize_text_for_matching(normalized_item["Test"])
            if key not in seen_tests:
                seen_tests.add(key)
                catalog.append(normalized_item)
    return catalog


def lab_testing_needed(care_plan):
    if not care_plan:
        return False

    lab_required = lab_requirement_status(care_plan.get("lab_test_required", ""))
    blood_required = lab_requirement_status(care_plan.get("blood_report_required", ""))
    return bool(care_plan.get("lab_tests")) or lab_required in {"yes", "depends"} or blood_required in {"yes", "depends"}


def format_lab_cart_rows(items):
    rows = []
    for item in items or []:
        rows.append(
            {
                "Lab Test": item.get("Test", ""),
                "Why It May Be Needed": item.get("Why It May Be Needed", ""),
                "Estimated Price": f"INR {float(item.get('Estimated Price', 0) or 0):,.0f}",
            }
        )
    return pd.DataFrame(rows)


def calculate_lab_cart_total(items):
    return float(sum(float(item.get("Estimated Price", 0) or 0) for item in items or []))


def lab_appointment_row_to_dict(row):
    try:
        lab_tests = json.loads(row["lab_tests_json"] or "[]")
    except json.JSONDecodeError:
        lab_tests = []
    try:
        symptoms = json.loads(row["symptoms_json"] or "[]")
    except json.JSONDecodeError:
        symptoms = []

    return {
        "lab_appointment_id": row["lab_appointment_id"],
        "username": row["username"],
        "patient_name": row["patient_name"],
        "predicted_disease": row["predicted_disease"],
        "lab_name": row["lab_name"],
        "lab_tests": lab_tests,
        "total_amount": float(row["total_amount"] or 0),
        "payment_method": row["payment_method"],
        "payment_status": row["payment_status"],
        "appointment_date": row["appointment_date"],
        "appointment_slot": row["appointment_slot"],
        "status": row["status"],
        "booked_at": row["booked_at"],
        "symptoms": symptoms,
        "payment_reference": row["payment_reference"],
    }


def load_lab_appointments_from_mysql():
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT lab_appointment_id, username, patient_name, predicted_disease,
                   lab_name, lab_tests_json, total_amount, payment_method, payment_status,
                   appointment_date, appointment_slot, status, booked_at, symptoms_json, payment_reference
            FROM lab_appointments
            ORDER BY appointment_date DESC, appointment_slot DESC, booked_at DESC
            """
        )
        return [lab_appointment_row_to_dict(row) for row in cursor.fetchall()]
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def save_lab_appointments_to_mysql(appointments):
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        for appointment in appointments:
            appointment_id = str(appointment.get("lab_appointment_id", "")).strip()
            username = str(appointment.get("username", "")).strip().lower()
            if not appointment_id or not username:
                continue
            cursor.execute(
                """
                INSERT INTO lab_appointments (
                    lab_appointment_id, username, patient_name, predicted_disease,
                    lab_name, lab_tests_json, total_amount, payment_method, payment_status,
                    appointment_date, appointment_slot, status, booked_at, symptoms_json,
                    payment_reference, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON DUPLICATE KEY UPDATE
                    username = VALUES(username),
                    patient_name = VALUES(patient_name),
                    predicted_disease = VALUES(predicted_disease),
                    lab_name = VALUES(lab_name),
                    lab_tests_json = VALUES(lab_tests_json),
                    total_amount = VALUES(total_amount),
                    payment_method = VALUES(payment_method),
                    payment_status = VALUES(payment_status),
                    appointment_date = VALUES(appointment_date),
                    appointment_slot = VALUES(appointment_slot),
                    status = VALUES(status),
                    booked_at = VALUES(booked_at),
                    symptoms_json = VALUES(symptoms_json),
                    payment_reference = VALUES(payment_reference),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    appointment_id,
                    username,
                    str(appointment.get("patient_name", "")).strip(),
                    str(appointment.get("predicted_disease", "")).strip(),
                    str(appointment.get("lab_name", "")).strip(),
                    json.dumps(appointment.get("lab_tests", [])),
                    float(appointment.get("total_amount", 0) or 0),
                    str(appointment.get("payment_method", "Cash")).strip(),
                    str(appointment.get("payment_status", "Pending")).strip(),
                    str(appointment.get("appointment_date", "")).strip(),
                    str(appointment.get("appointment_slot", "")).strip(),
                    str(appointment.get("status", "Booked")).strip() or "Booked",
                    str(appointment.get("booked_at", "")).strip(),
                    json.dumps(appointment.get("symptoms", [])),
                    str(appointment.get("payment_reference", "")).strip(),
                ),
            )
        conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def load_lab_appointments():
    ensure_database_ready()
    return load_lab_appointments_from_mysql()


def save_lab_appointments(appointments):
    ensure_database_ready()
    save_lab_appointments_to_mysql(appointments)


def save_lab_appointment(appointment):
    save_lab_appointments([appointment])


def is_lab_appointment_slot_taken_in_mysql(lab_name, appointment_date, appointment_slot):
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1
            FROM lab_appointments
            WHERE lower(trim(lab_name)) = lower(trim(%s))
              AND appointment_date = %s
              AND lower(trim(appointment_slot)) = lower(trim(%s))
              AND status = 'Booked'
            LIMIT 1
            """,
            (lab_name, appointment_date, appointment_slot),
        )
        return cursor.fetchone() is not None
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def is_lab_appointment_slot_taken(lab_name, appointment_date, appointment_slot):
    ensure_database_ready()
    return is_lab_appointment_slot_taken_in_mysql(lab_name, appointment_date, appointment_slot)


def cancel_lab_appointment_for_user_in_mysql(appointment_id, username):
    conn = get_mysql_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE lab_appointments
            SET status = 'Cancelled',
                updated_at = CURRENT_TIMESTAMP
            WHERE lab_appointment_id = %s
              AND username = %s
              AND status = 'Booked'
            """,
            (appointment_id, username),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def cancel_lab_appointment_for_user(appointment_id, username):
    ensure_database_ready()
    return cancel_lab_appointment_for_user_in_mysql(appointment_id, username)


def format_lab_appointment_rows(appointments):
    if not appointments:
        return pd.DataFrame()

    rows = []
    for appointment in appointments:
        test_names = [
            str(item.get("Test", "")).strip()
            for item in appointment.get("lab_tests", [])
            if str(item.get("Test", "")).strip()
        ]
        rows.append(
            {
                "Lab Appointment ID": appointment["lab_appointment_id"],
                "Date": appointment["appointment_date"],
                "Time": appointment["appointment_slot"],
                "Lab": appointment["lab_name"],
                "Tests": ", ".join(test_names),
                "Amount": f"INR {float(appointment.get('total_amount', 0) or 0):,.0f}",
                "Payment": appointment["payment_status"],
                "Method": appointment["payment_method"],
                "Predicted Disease": appointment["predicted_disease"],
                "Status": appointment["status"],
                "Booked On": appointment["booked_at"],
            }
        )

    return pd.DataFrame(rows)


def format_appointment_rows(appointments):
    if not appointments:
        return pd.DataFrame()

    rows = []
    for appointment in appointments:
        rows.append(
            {
                "Appointment ID": appointment["appointment_id"],
                "Date": appointment["appointment_date"],
                "Time": appointment["appointment_slot"],
                "Doctor": appointment["doctor_name"],
                "Specialty": appointment["specialist"],
                "Mode": appointment["consultation_mode"],
                "Hospital": appointment["hospital"],
                "Predicted Disease": appointment["predicted_disease"],
                "Status": appointment["status"],
                "Booked On": appointment["booked_at"],
            }
        )

    return pd.DataFrame(rows)


def render_themed_dataframe(dataframe, min_width=820):
    if get_active_ui_theme() != "Dark":
        st.dataframe(dataframe, width="stretch")
        return

    if dataframe is None or dataframe.empty:
        st.info("No rows available.")
        return

    safe_df = dataframe.fillna("").astype(str)
    header_cells = "".join(f"<th>{html.escape(str(column))}</th>" for column in safe_df.columns)
    body_rows = []
    for _, row in safe_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        body_rows.append(f"<tr>{cells}</tr>")

    st.markdown(
        f"""
        <div class="dashboard-dark-table-wrap">
            <table class="dashboard-dark-table" style="min-width: {int(min_width)}px;">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lab_test_booking_section(care_plan, predicted_disease, selected_symptoms, show_catalog=True):
    render_section_intro(
        "Lab Test Booking",
        "Select suggested tests, add them to cart, choose payment, and save the lab appointment.",
    )

    st.session_state.setdefault("lab_test_cart", [])
    st.session_state.setdefault("lab_booking_notice", "")

    if not lab_testing_needed(care_plan):
        st.session_state.lab_test_cart = []
        st.session_state.lab_booking_notice = ""
        st.session_state.lab_test_cart_context = ""
        st.success("Lab test booking is not required for this result unless symptoms persist, worsen, or a doctor advises testing.")
        return

    catalog = build_lab_test_catalog(care_plan)
    if not catalog:
        st.session_state.lab_test_cart = []
        st.info("No specific lab test is available for this result. Please discuss testing with a doctor before booking.")
        return

    catalog_fingerprint = "|".join(
        normalize_text_for_matching(item.get("Test", ""))
        for item in catalog
    )
    lab_cart_context = (
        f"{normalize_disease_name(predicted_disease)}|"
        f"{care_plan.get('blood_report_required', '')}|"
        f"{care_plan.get('lab_test_required', '')}|"
        f"{catalog_fingerprint}"
    )
    if st.session_state.get("lab_test_cart_context") != lab_cart_context:
        st.session_state.lab_test_cart = []
        st.session_state.lab_booking_notice = ""
        st.session_state.lab_test_cart_context = lab_cart_context

    if st.session_state.lab_booking_notice:
        st.success(st.session_state.lab_booking_notice)

    if show_catalog:
        catalog_df = format_lab_cart_rows(catalog).reset_index(drop=True)
        render_themed_dataframe(catalog_df, min_width=780)
        st.caption("Test names come from the care plan for this prediction. Prices are estimates, so confirm final charges with the diagnostic center.")

    label_to_item = {
        f"{item['Test']} - INR {float(item.get('Estimated Price', 0) or 0):,.0f}": item
        for item in catalog
    }
    selected_labels = st.multiselect(
        "Select lab tests",
        list(label_to_item.keys()),
        key="lab_test_selector",
    )

    add_col, all_col, clear_col = st.columns(3)
    with add_col:
        if st.button("Add Selected Tests", type="primary", key="add_selected_lab_tests"):
            if not selected_labels:
                st.warning("Please select at least one lab test.")
            else:
                existing = {
                    normalize_text_for_matching(item.get("Test", ""))
                    for item in st.session_state.lab_test_cart
                }
                added_count = 0
                for label in selected_labels:
                    item = label_to_item[label]
                    item_key = normalize_text_for_matching(item["Test"])
                    if item_key in existing:
                        continue
                    st.session_state.lab_test_cart.append(item)
                    existing.add(item_key)
                    added_count += 1
                if added_count:
                    st.success(f"Added {added_count} test(s) to cart.")
                else:
                    st.info("Selected tests are already in the cart.")
    with all_col:
        if st.button("Add All Suggested", key="add_all_lab_tests"):
            st.session_state.lab_test_cart = [dict(item) for item in catalog]
            st.success("All suggested tests are in the cart.")
    with clear_col:
        if st.button("Clear Cart", key="clear_lab_test_cart"):
            st.session_state.lab_test_cart = []
            st.info("Lab test cart cleared.")

    catalog_by_key = {
        normalize_text_for_matching(item.get("Test", "")): item
        for item in catalog
    }
    cart_items = []
    seen_cart_items = set()
    for item in st.session_state.lab_test_cart:
        item_key = normalize_text_for_matching(item.get("Test", ""))
        if item_key not in catalog_by_key or item_key in seen_cart_items:
            continue
        cart_items.append(dict(catalog_by_key[item_key]))
        seen_cart_items.add(item_key)
    st.session_state.lab_test_cart = cart_items
    cart_total = calculate_lab_cart_total(cart_items)

    cart_col1, cart_col2, cart_col3 = st.columns(3)
    with cart_col1:
        st.metric("Cart Tests", len(cart_items))
    with cart_col2:
        st.metric("Estimated Total", f"INR {cart_total:,.0f}")
    with cart_col3:
        st.metric("Cart Status", "Ready" if cart_items else "Empty")

    if cart_items:
        render_themed_dataframe(format_lab_cart_rows(cart_items), min_width=780)
    else:
        st.caption("Add at least one lab test to continue booking.")

    booking_col1, booking_col2 = st.columns(2)
    with booking_col1:
        lab_name = st.text_input(
            "Diagnostic Center / Lab",
            placeholder="Example: City Diagnostic Lab",
            key="lab_booking_lab_name",
        )
        lab_date = st.date_input(
            "Lab Appointment Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            key="lab_booking_date",
        )
    with booking_col2:
        lab_slot = st.text_input(
            "Lab Appointment Time",
            placeholder="Example: 8:30 AM",
            key="lab_booking_slot",
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Cash", "Online"],
            key="lab_payment_method",
        )

    payment_reference = ""
    if payment_method == "Online":
        payment_reference = st.text_input(
            "Online Payment Reference",
            placeholder="Enter UPI/card transaction reference",
            key="lab_payment_reference",
        )
        st.caption("Online payment reference is saved with the booking.")
    else:
        st.caption("Cash payment will be marked as pending until paid at the lab.")

    if st.button("Save Payment And Book Lab Appointment", type="primary", key="confirm_lab_test_booking"):
        missing_fields = []
        if not cart_items:
            missing_fields.append("at least one lab test")
        if not lab_name.strip():
            missing_fields.append("diagnostic center or lab name")
        if not lab_slot.strip():
            missing_fields.append("appointment time")
        if payment_method == "Online" and not payment_reference.strip():
            missing_fields.append("online payment reference")

        if missing_fields:
            st.error(f"Please enter {', '.join(missing_fields)}.")
            return

        booking_tests = [dict(item) for item in cart_items]
        booking_total = calculate_lab_cart_total(booking_tests)
        appointment_date_value = lab_date.isoformat()
        if is_lab_appointment_slot_taken(lab_name.strip(), appointment_date_value, lab_slot.strip()):
            st.error("That lab time slot has already been booked. Please choose another slot.")
            return

        current_user = st.session_state.current_user or {}
        username = current_user.get("username", "").strip().lower()
        patient_name = f"{current_user.get('first_name', '').strip()} {current_user.get('last_name', '').strip()}".strip()
        lab_appointment_id = build_lab_appointment_id()
        payment_status = "Online reference saved" if payment_method == "Online" else "Cash pending at lab"

        save_lab_appointment(
            {
                "lab_appointment_id": lab_appointment_id,
                "username": username,
                "patient_name": patient_name,
                "predicted_disease": predicted_disease,
                "lab_name": lab_name.strip(),
                "lab_tests": booking_tests,
                "total_amount": booking_total,
                "payment_method": payment_method,
                "payment_status": payment_status,
                "appointment_date": appointment_date_value,
                "appointment_slot": lab_slot.strip(),
                "status": "Booked",
                "booked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "symptoms": [format_symptom_label(symptom) for symptom in selected_symptoms],
                "payment_reference": payment_reference.strip(),
            }
        )
        st.session_state.lab_test_cart = []
        st.session_state.lab_booking_notice = (
            f"Lab appointment {lab_appointment_id} booked at {lab_name.strip()} "
            f"on {appointment_date_value} at {lab_slot.strip()}."
        )
        st.success(st.session_state.lab_booking_notice)

    current_user = st.session_state.current_user or {}
    username = current_user.get("username", "").strip().lower()
    active_lab_appointments = [
        item
        for item in load_lab_appointments()
        if item.get("username", "").strip().lower() == username and item.get("status") == "Booked"
    ]
    if active_lab_appointments:
        render_section_intro(
            "Your Active Lab Tests",
            "Recently booked lab appointments for the signed-in user.",
        )
        recent_lab_bookings = sorted(
            active_lab_appointments,
            key=lambda item: (item.get("appointment_date", ""), item.get("appointment_slot", "")),
            reverse=True,
        )[:5]
        render_themed_dataframe(format_lab_appointment_rows(recent_lab_bookings), min_width=920)


def render_appointments_page():
    render_section_intro(
        "Your Care Timeline",
        "Review doctor visits and lab test appointments in one place, then cancel bookings cleanly if plans change.",
    )
    current_user = st.session_state.current_user or {}
    username = current_user.get("username", "").strip().lower()

    all_appointments = load_appointments()
    user_appointments = [item for item in all_appointments if item.get("username", "").strip().lower() == username]
    all_lab_appointments = load_lab_appointments()
    user_lab_appointments = [
        item for item in all_lab_appointments if item.get("username", "").strip().lower() == username
    ]

    if not user_appointments and not user_lab_appointments:
        st.info("No appointments booked yet. Make a prediction first, then book a doctor or lab test from the dashboard.")
        empty_col1, empty_col2 = st.columns(2)
        with empty_col1:
            if st.button("Go To Dashboard", key="appointments_go_dashboard", type="primary"):
                st.session_state.page = "Dashboard"
                st.rerun()
        with empty_col2:
            render_info_card(
                "How Booking Works",
                "Once a disease is predicted, the dashboard recommends specialists, lab tests when needed, and saves each booking here.",
                kicker="Empty State",
            )
        return

    user_appointments = sorted(
        user_appointments,
        key=lambda item: (item.get("appointment_date", ""), item.get("appointment_slot", ""), item.get("booked_at", "")),
        reverse=True,
    )
    user_lab_appointments = sorted(
        user_lab_appointments,
        key=lambda item: (item.get("appointment_date", ""), item.get("appointment_slot", ""), item.get("booked_at", "")),
        reverse=True,
    )

    active_appointments = [item for item in user_appointments if item.get("status") == "Booked"]
    history_appointments = [item for item in user_appointments if item.get("status") != "Booked"]
    active_lab_appointments = [item for item in user_lab_appointments if item.get("status") == "Booked"]
    history_lab_appointments = [item for item in user_lab_appointments if item.get("status") != "Booked"]

    render_section_intro(
        "Booking Center",
        "Active doctor visits, lab tests, and past booking history are separated for easier tracking.",
    )
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        st.metric("Doctor Active", len(active_appointments))
    with summary_col2:
        st.metric("Lab Active", len(active_lab_appointments))
    with summary_col3:
        st.metric("History", len(history_appointments) + len(history_lab_appointments))
    with summary_col4:
        st.metric("Total", len(user_appointments) + len(user_lab_appointments))

    doctor_tab, lab_tab, history_tab = st.tabs(["Doctor Bookings", "Lab Tests", "History"])
    with doctor_tab:
        if active_appointments:
            render_themed_dataframe(format_appointment_rows(active_appointments))
        else:
            st.info("No active doctor bookings right now.")

    with lab_tab:
        if active_lab_appointments:
            render_themed_dataframe(format_lab_appointment_rows(active_lab_appointments), min_width=920)
        else:
            st.info("No active lab test bookings right now.")

    with history_tab:
        if history_appointments:
            st.markdown("**Doctor appointment history**")
            render_themed_dataframe(format_appointment_rows(history_appointments))
        if history_lab_appointments:
            st.markdown("**Lab appointment history**")
            render_themed_dataframe(format_lab_appointment_rows(history_lab_appointments), min_width=920)
        if not history_appointments and not history_lab_appointments:
            st.info("No past or cancelled appointments yet.")

    if active_appointments:
        cancel_options = {
            f"{item['appointment_id']} | {item['doctor_name']} | {item['appointment_date']} | {item['appointment_slot']}": item[
                "appointment_id"
            ]
            for item in active_appointments
        }

        render_section_intro(
            "Cancel Doctor Appointment",
            "Choose an active doctor booking if you need to release the appointment slot.",
        )
        selected_cancel_label = st.selectbox("Choose doctor appointment", list(cancel_options.keys()), key="cancel_appointment")
        if st.button("Cancel Selected Doctor Appointment", type="secondary", key="cancel_appointment_button"):
            appointment_id = cancel_options[selected_cancel_label]
            if cancel_appointment_for_user(appointment_id, username):
                st.success(f"Appointment {appointment_id} has been cancelled.")
                st.rerun()
            else:
                st.error("Unable to cancel the selected appointment.")
    else:
        st.caption("There are no active doctor bookings available for cancellation.")

    if active_lab_appointments:
        lab_cancel_options = {
            f"{item['lab_appointment_id']} | {item['lab_name']} | {item['appointment_date']} | {item['appointment_slot']}": item[
                "lab_appointment_id"
            ]
            for item in active_lab_appointments
        }

        render_section_intro(
            "Cancel Lab Appointment",
            "Choose an active lab test booking if you need to release the slot.",
        )
        selected_lab_cancel_label = st.selectbox(
            "Choose lab appointment",
            list(lab_cancel_options.keys()),
            key="cancel_lab_appointment",
        )
        if st.button("Cancel Selected Lab Appointment", type="secondary", key="cancel_lab_appointment_button"):
            lab_appointment_id = lab_cancel_options[selected_lab_cancel_label]
            if cancel_lab_appointment_for_user(lab_appointment_id, username):
                st.success(f"Lab appointment {lab_appointment_id} has been cancelled.")
                st.rerun()
            else:
                st.error("Unable to cancel the selected lab appointment.")
    else:
        st.caption("There are no active lab bookings available for cancellation.")


def render_lab_diagnostic_center_search(care_plan, predicted_disease):
    render_section_intro(
        "Diagnostic Center Search",
        "Enter a city or area to find nearby diagnostic labs before booking your selected tests.",
    )

    location_text = st.text_input(
        "Enter city or area for diagnostic labs",
        placeholder="Type any Indian city or area, example: ahm, naroda, pune",
        key="lab_search_location",
    )
    typed_location = location_text.strip()
    maps_key_token = get_map_provider_cache_token()
    city_suggestions = get_city_suggestions(location_text)
    selected_city = ""
    if len(normalize_location_query(location_text)) >= 2:
        if city_suggestions:
            city_preview_html = "".join(
                f"<span class='city-suggestion-pill'>{html.escape(city)}</span>"
                for city in city_suggestions
            )
            st.markdown(
                f"""
                <div class="city-suggestion-panel">
                    <p class="city-suggestion-title">Matching city and area suggestions</p>
                    <div class="city-suggestion-list">{city_preview_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            location_options = []
            if typed_location:
                location_options.append(typed_location)
            for suggestion in city_suggestions:
                if normalize_location_query(suggestion) not in {
                    normalize_location_query(option) for option in location_options
                }:
                    location_options.append(suggestion)
            selected_city = st.selectbox(
                "Location used for lab search",
                location_options,
                key="lab_city_suggestion",
            )
        else:
            st.caption("No saved match found yet. You can still search any Indian city or area with the text you typed.")
    else:
        st.caption("Type at least 2 letters to show India-wide city and area suggestions.")

    search_location = selected_city or typed_location
    current_lab_context = (
        f"{normalize_location_query(search_location)}|"
        f"{normalize_disease_name(predicted_disease)}|"
        f"{care_plan.get('lab_test_required', '')}|"
        f"{care_plan.get('blood_report_required', '')}|"
        f"{maps_key_token}"
    )
    if st.session_state.lab_provider_location_fingerprint != current_lab_context:
        previous_lab_was_selected = bool(
            st.session_state.selected_lab_provider
            or st.session_state.selected_lab_provider_context
            or st.session_state.lab_provider_autofill_notice
        )
        st.session_state.lab_provider_search_results = None
        st.session_state.lab_provider_search_context = ""
        st.session_state.selected_lab_provider = None
        st.session_state.selected_lab_provider_context = ""
        st.session_state.lab_provider_autofill_notice = ""
        st.session_state.pop("selected_lab_provider_for_booking", None)
        if previous_lab_was_selected:
            st.session_state.lab_booking_lab_name = ""
        st.session_state.lab_provider_location_fingerprint = current_lab_context

    if st.button("Find Diagnostic Labs", type="primary", width="stretch", key="search_diagnostic_labs_button"):
        if len(normalize_location_query(search_location)) < 2:
            st.warning("Please enter a city or area before searching diagnostic labs.")
        else:
            with st.spinner(f"Finding diagnostic labs near {search_location}..."):
                st.session_state.lab_provider_search_results = search_nearby_healthcare(
                    search_location,
                    "Diagnostic Lab",
                    predicted_disease,
                    "labs",
                    maps_key_token,
                )
                st.session_state.lab_provider_search_context = current_lab_context

    lab_results = None
    if (
        st.session_state.lab_provider_search_results
        and st.session_state.lab_provider_search_context == current_lab_context
    ):
        lab_results = st.session_state.lab_provider_search_results

    if lab_results:
        origin = lab_results.get("origin") or {}
        if origin:
            radius_copy = format_search_radius(lab_results.get("search_radius_m"))
            radius_sentence = f" Search radius: about {html.escape(radius_copy)}." if radius_copy else ""
            st.markdown(
                (
                    '<div class="provider-source-note">'
                    f"Showing diagnostic lab listings near {html.escape(clean_provider_text(origin.get('display_name'), search_location))}. "
                    f"Source: {html.escape(lab_results.get('source', 'public map listings'))}."
                    f"{radius_sentence}"
                    " Click a lab card to open its map location."
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        if lab_results.get("error"):
            st.warning(lab_results["error"])

        labs = lab_results.get("labs", [])
        render_provider_results(
            "Nearby Diagnostic Labs",
            labs,
            "No named diagnostic lab listing was found for this area. Try a nearby larger city or enter the lab name manually.",
        )

        if labs:
            lab_options = {
                f"{idx + 1}. {format_provider_option(provider)}": provider
                for idx, provider in enumerate(labs)
            }
            selected_lab_label = st.selectbox(
                "Select a diagnostic lab for booking",
                list(lab_options.keys()),
                key="selected_lab_provider_for_booking",
            )
            if st.button("Use Selected Diagnostic Lab", type="primary", key="use_lab_provider_recommendation"):
                selected_lab = lab_options[selected_lab_label]
                selected_lab_name = clean_provider_text(selected_lab.get("name"), "selected diagnostic lab")
                st.session_state.lab_booking_lab_name = selected_lab_name
                st.session_state.selected_lab_provider = selected_lab
                st.session_state.selected_lab_provider_context = current_lab_context
                st.session_state.lab_provider_autofill_notice = (
                    f"Added {selected_lab_name} to the lab booking form."
                )
        else:
            st.info("No named diagnostic lab listing was found. You can still enter the diagnostic center manually.")
    elif len(normalize_location_query(search_location)) >= 2:
        st.caption("Click Find Diagnostic Labs to load nearby labs inside the Lab Tests page.")

    if st.session_state.lab_provider_autofill_notice and st.session_state.selected_lab_provider_context == current_lab_context:
        st.success(st.session_state.lab_provider_autofill_notice)


def render_lab_tests_page():
    render_section_intro(
        "Lab Tests",
        "Review tests from the latest screening result, then select tests, payment method, and appointment slot.",
    )

    prediction_context = st.session_state.get("prediction_context")
    if not prediction_context:
        st.info("Run a prediction first. If lab tests are needed, the prescribed tests will appear here.")
        if st.button("Go To Dashboard", key="lab_tests_go_dashboard_empty", type="primary"):
            st.session_state.page = "Dashboard"
            st.rerun()
        return

    if prediction_context.get("guarded"):
        st.warning("The latest screening needs more symptom detail before lab tests can be suggested.")
        st.caption(prediction_context.get("guard_message", "Please add clearer symptoms, then run prediction again."))
        if st.button("Go To Dashboard", key="lab_tests_go_dashboard_guarded", type="primary"):
            st.session_state.page = "Dashboard"
            st.rerun()
        return

    predicted_disease = str(prediction_context.get("predicted_disease", "")).strip()
    selected_symptoms = list(prediction_context.get("selected_symptoms", []))
    confidence = float(prediction_context.get("confidence", 0.0) or 0.0)
    care_plan = get_disease_care_plan(predicted_disease, confidence)
    catalog = build_lab_test_catalog(care_plan)

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        st.metric("Predicted Disease", predicted_disease or "Not available")
    with summary_col2:
        st.metric("Blood Report", care_plan["blood_report_required"])
    with summary_col3:
        st.metric("Lab Tests", care_plan["lab_test_required"])
    with summary_col4:
        st.metric("Suggested Tests", len(catalog))

    st.caption("These tests come from the latest screening care plan. A qualified doctor or diagnostic center should confirm the final test order.")

    if lab_testing_needed(care_plan):
        render_lab_diagnostic_center_search(care_plan, predicted_disease)

    if catalog:
        render_section_intro(
            "Prescribed Tests From Latest Screening",
            "Only the tests connected to the current prediction are shown here.",
        )
        render_themed_dataframe(format_lab_cart_rows(catalog), min_width=780)
        st.caption("Prices are estimates. Confirm the final test order and charges with the diagnostic center.")
        render_lab_test_booking_section(care_plan, predicted_disease, selected_symptoms, show_catalog=False)
    elif lab_testing_needed(care_plan):
        st.info("Testing may be needed, but this screening result does not prescribe a specific test list. Please discuss the right test with a doctor.")
    else:
        st.success("No lab test is usually needed for this result unless symptoms persist, worsen, or a doctor advises testing.")


def render_navbar():
    nav_options = ["Home", "Login", "Register", "About", "Contact Us"]
    if st.session_state.logged_in:
        nav_options = ["Dashboard", "Lab Tests", "Appointments", "Contact Us", "Home", "About"]

    if st.session_state.page not in nav_options:
        st.session_state.page = nav_options[0]

    signed_in_chip = ""
    subtitle = "AI symptom radar for faster insights, smarter care routes, and confident booking."
    if st.session_state.logged_in:
        name = st.session_state.current_user["first_name"] + " " + st.session_state.current_user["last_name"]
        signed_in_chip = f"<div class='inline-chip'>Signed in as {html.escape(name)}</div>"

    button_label_map = {
        "Home": "Home",
        "Dashboard": "Dashboard",
        "Lab Tests": "Lab Tests",
        "Appointments": "Appointments",
        "About": "About Us",
        "Register": "Register",
        "Login": "Login",
        "Contact Us": "Contact Us",
        "Logout": "Logout",
    }

    nav_label_to_page = {button_label_map.get(page_name, page_name): page_name for page_name in nav_options}
    nav_labels = [button_label_map.get(page_name, page_name) for page_name in nav_options]
    if st.session_state.logged_in:
        nav_labels.append("Logout")

    current_nav_label = button_label_map.get(st.session_state.page, st.session_state.page)
    if st.session_state.get("main_nav_choice") not in nav_labels:
        st.session_state.main_nav_choice = current_nav_label

    def sync_sidebar_navigation():
        selected_label = st.session_state.get("main_nav_choice", current_nav_label)
        if selected_label == "Logout":
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.prediction_context = None
            st.session_state.prediction_symptoms = []
            reset_symptom_assistant_state(clear_connection=True)
            st.session_state.page = "Home"
            st.session_state.main_nav_choice = button_label_map["Home"]
            st.session_state.close_sidebar_after_nav = True
            return

        selected_page_name = nav_label_to_page.get(selected_label)
        if selected_page_name:
            st.session_state.page = selected_page_name
            st.session_state.close_sidebar_after_nav = True

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Disease Prediction System</div>
            <p class="sidebar-brand-copy">Patient dashboard for symptoms, care guidance, and booking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.logged_in:
        name = st.session_state.current_user["first_name"] + " " + st.session_state.current_user["last_name"]
        st.sidebar.markdown(
            f"<span class='sidebar-user-chip'>Signed in: {html.escape(name)}</span>",
            unsafe_allow_html=True,
        )

    selected_nav_label = st.sidebar.radio(
        "Navigation",
        nav_labels,
        key="main_nav_choice",
        label_visibility="collapsed",
        on_change=sync_sidebar_navigation,
    )

    render_theme_toggle()

    st.markdown(
        f"""
        <div class="topbar-shell">
            <div class="topbar-title">Disease Prediction System</div>
            <p class="topbar-subtitle topbar-subtitle--dynamic">{html.escape(subtitle)}</p>
            {signed_in_chip}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_nav_label == "Logout":
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.prediction_context = None
        st.session_state.prediction_symptoms = []
        reset_symptom_assistant_state(clear_connection=True)
        st.session_state.page = "Home"
        st.session_state.main_nav_choice = button_label_map["Home"]
        st.session_state.close_sidebar_after_nav = True
        st.rerun()

    selected_page = nav_label_to_page.get(selected_nav_label, st.session_state.page)
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.session_state.close_sidebar_after_nav = True

    st.divider()
    return st.session_state.page


def close_sidebar_after_navigation():
    components.html(
        """
        <script>
        (() => {
            const parentDoc = window.parent.document;

            const buttonMeta = (button) => {
                const aria = (button?.getAttribute('aria-label') || '').toLowerCase();
                const title = (button?.getAttribute('title') || '').toLowerCase();
                const testid = (button?.getAttribute('data-testid') || '').toLowerCase();
                const kind = (button?.getAttribute('kind') || '').toLowerCase();
                return { aria, title, testid, kind };
            };

            const isCloseButton = (button) => {
                const { aria, title, testid, kind } = buttonMeta(button);
                return (
                    aria.includes('close sidebar') ||
                    aria.includes('collapse sidebar') ||
                    title.includes('close sidebar') ||
                    title.includes('collapse sidebar') ||
                    testid.includes('sidebarcollapsebutton') ||
                    (kind === 'header' && !aria.includes('expand') && !aria.includes('open'))
                );
            };

            const closeSidebar = () => {
                const closeButton = Array.from(parentDoc.querySelectorAll('button')).find(isCloseButton);
                if (closeButton) {
                    closeButton.click();
                }
            };

            [360, 720, 1080].forEach((delay) => window.setTimeout(closeSidebar, delay));
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_home(model_metadata, label_encoder, feature_names, X_train):
    current_user = st.session_state.current_user or {}
    patient_name = (
        f"{current_user.get('first_name', '').strip()} {current_user.get('last_name', '').strip()}".strip()
        if st.session_state.logged_in
        else "Patient"
    )
    patient_avatar_b64 = get_image_base64(PATIENT_AVATAR_PATH)
    avatar_initials = "".join(part[:1] for part in patient_name.split()[:2]).upper() or "P"
    patient_avatar_html = (
        f"<img src='data:image/png;base64,{patient_avatar_b64}' alt='patient avatar' />"
        if patient_avatar_b64
        else f"<span>{html.escape(avatar_initials)}</span>"
    )

    username = current_user.get("username", "").strip().lower()
    active_appointments = [
        item
        for item in load_appointments()
        if item.get("username", "").strip().lower() == username and item.get("status") == "Booked"
    ]
    active_count = len(active_appointments)

    main_col, profile_col = st.columns([2.25, 0.9], gap="large")
    with main_col:
        st.markdown(
            f"""
            <div class="patient-dashboard">
                <div class="patient-hero-card">
                    <div class="patient-hero-copy">
                        <h2 class="patient-greeting">Hello, <span>{html.escape(patient_name)}</span></h2>
                        <p class="patient-hero-text">Track symptoms, check prediction guidance, and keep doctor follow-up organized in one calm dashboard.</p>
                        <span class="patient-read-more">Start from the sidebar</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="patient-stat-grid">
                <div class="patient-stat-card stat-blue">
                    <p class="patient-stat-kicker">Diseases</p>
                    <div class="patient-stat-value">{len(label_encoder.classes_)}</div>
                    <p class="patient-stat-sub">Prediction classes</p>
                </div>
                <div class="patient-stat-card stat-mint">
                    <p class="patient-stat-kicker">Symptoms</p>
                    <div class="patient-stat-value">{len(feature_names)}</div>
                    <p class="patient-stat-sub">Inputs available</p>
                </div>
                <div class="patient-stat-card stat-yellow">
                    <p class="patient-stat-kicker">Accuracy</p>
                    <div class="patient-stat-value">{model_metadata['accuracy']:.1%}</div>
                    <p class="patient-stat-sub">Saved model score</p>
                </div>
                <div class="patient-stat-card stat-rose">
                    <p class="patient-stat-kicker">Bookings</p>
                    <div class="patient-stat-value">{active_count}</div>
                    <p class="patient-stat-sub">Active appointments</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        flow_df = pd.DataFrame(
            {
                "Step": ["Capture", "Predict", "Guide", "Book"],
                "Score": [72, 86, 78, min(92, 48 + active_count * 12)],
            }
        )
        fig_flow = go.Figure(
            go.Scatter(
                x=flow_df["Step"],
                y=flow_df["Score"],
                mode="lines+markers",
                line=dict(color="#ea7a35", width=4, shape="spline"),
                marker=dict(size=11, color=["#5a63d8", "#22c7a9", "#f5b642", "#ff7f9c"], line=dict(width=3, color="#ffffff")),
                fill="tozeroy",
                fillcolor="rgba(234, 122, 53, 0.12)",
                hovertemplate="%{x}: %{y}%<extra></extra>",
            )
        )
        fig_flow.update_yaxes(range=[0, 100], ticksuffix="%")
        apply_warm_compact_plot_style(fig_flow, height=190)

        render_small_plot_card("Care Flow", "A warm step trend from symptoms to booking.", fig_flow)

    with profile_col:
        today = date.today()
        first_weekday, days_in_month = calendar.monthrange(today.year, today.month)
        weekday_headers = "".join(
            f"<div class='patient-calendar-weekday'>{html.escape(day)}</div>"
            for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        )
        leading_blanks = "".join("<div class='patient-calendar-day empty'></div>" for _ in range(first_weekday))
        calendar_days = "".join(
            f"<div class='patient-calendar-day{' active' if day == today.day else ''}'>{day}</div>"
            for day in range(1, days_in_month + 1)
        )
        if active_appointments:
            appointment_items = "".join(
                "<div class='patient-appointment-item'>"
                f"<div><strong>{html.escape(item.get('doctor_name', 'Doctor'))}</strong><br>"
                f"<span>{html.escape(item.get('hospital', 'Clinic'))}</span></div>"
                f"<span>{html.escape(item.get('appointment_slot', 'Time'))}</span>"
                "</div>"
                for item in active_appointments[:3]
            )
        else:
            appointment_items = (
                "<div class='patient-appointment-item'>"
                "<div><strong>No active booking</strong><br>"
                "<span>Run prediction and save a doctor visit.</span></div>"
                "</div>"
            )

        profile_panel_html = (
            "<div class='patient-profile-panel'>"
            f"<div class='patient-avatar'>{patient_avatar_html}</div>"
            f"<h3 class='patient-profile-name'>{html.escape(patient_name)}</h3>"
            "<p class='patient-profile-role'>Patient dashboard</p>"
            "<div class='patient-profile-stats'>"
            f"<div class='patient-profile-chip'><strong>{len(label_encoder.classes_)}</strong><span>diseases</span></div>"
            f"<div class='patient-profile-chip'><strong>{len(feature_names)}</strong><span>symptoms</span></div>"
            f"<div class='patient-profile-chip'><strong>{active_count}</strong><span>bookings</span></div>"
            "</div>"
            f"<div class='patient-calendar-title'>{today.strftime('%B %Y')}</div>"
            f"<div class='patient-calendar-grid'>{weekday_headers}{leading_blanks}{calendar_days}</div>"
            "<div class='patient-plan-title'>Appointments</div>"
            f"{appointment_items}"
            "</div>"
        )
        st.markdown(profile_panel_html, unsafe_allow_html=True)


def render_about():
    render_section_intro(
        "About The System",
        "This project is designed as an educational screening support interface with a more guided, modern clinical workflow.",
    )
    about_col1, about_col2, about_col3 = st.columns(3)
    with about_col1:
        render_info_card(
            "Why It Exists",
            "The system helps users explore likely diseases from symptoms using a trained machine learning model.",
            kicker="Purpose",
        )
    with about_col2:
        render_info_card(
            "Care Path",
            "Prediction results connect to severity, tests, doctor selection, and appointment history.",
            kicker="Flow",
        )
    with about_col3:
        render_info_card(
            "Important Safety Note",
            "Predictions are not a medical diagnosis. Final confirmation should always come from a qualified doctor.",
            kicker="Safety",
        )

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        render_status_card(
            "Database persistence enabled",
            "Users and appointments are stored directly in MySQL.",
            label="Production Foundation",
        )
    with status_col2:
        render_status_card(
            "Clinical and privacy review still required",
            "Before real patient use, validate medical content, consent, privacy, backups, and deployment security.",
            label="Launch Gate",
        )


def render_register():
    render_section_intro(
        "Create Your Account",
        "Use a simple registration flow to access the dashboard, prediction results, and appointment management tools.",
    )

    form_col, info_col = st.columns([1.2, 0.8])

    with form_col:
        with st.form("register_form", clear_on_submit=True):
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account")

    with info_col:
        render_info_card(
            "What You Unlock",
            "Registered users can access prediction, care guidance, doctor booking, and appointment history.",
            kicker="Account Benefit",
        )
        render_info_card(
            "Quick Tip",
            "Use a memorable username and a strong password to return to your appointments later.",
            kicker="Best Practice",
        )

    if submitted:
        if not all([first_name.strip(), last_name.strip(), username.strip(), password, confirm_password]):
            st.error("Please fill all fields.")
            return

        if password != confirm_password:
            st.error("Password and Confirm Password do not match.")
            return

        users = load_users()
        user_key = username.strip().lower()

        if user_key in users:
            st.error("Username already exists.")
            return

        users[user_key] = {
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "username": username.strip(),
            "password_hash": hash_password(password),
        }
        save_users(users)
        st.success("Account created successfully. You can now login.")


def render_login():
    render_section_intro(
        "Sign In",
        "Access the dashboard to continue prediction, diagnostics review, and doctor booking.",
    )

    login_tab, reset_tab = st.tabs(["Sign In", "Forgot Password"])

    with login_tab:
        form_col, help_col = st.columns([1.15, 0.85])
        with form_col:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In")

        with help_col:
            render_info_card(
                "Account Access",
                "Sign in to use prediction, doctor booking, and appointment history.",
                kicker="Login",
            )

        if submitted:
            users = load_users()
            user_key = username.strip().lower()

            if user_key not in users:
                st.error("Invalid username or password.")
                return

            if not verify_password(password, users[user_key]["password_hash"]):
                st.error("Invalid username or password.")
                return

            # Upgrade legacy hash on successful login.
            if not users[user_key]["password_hash"].startswith("pbkdf2_sha256$"):
                users[user_key]["password_hash"] = hash_password(password)
                save_users(users)

            st.session_state.logged_in = True
            st.session_state.current_user = users[user_key]
            st.session_state.page = "Dashboard"
            st.success("Login successful. Redirecting to Dashboard...")
            st.rerun()

    with reset_tab:
        render_info_card(
            "Password Reset",
            "Enter your username and set a new password.",
            kicker="Recovery",
        )
        with st.form("reset_password_form", clear_on_submit=True):
            username = st.text_input("Username", key="reset_username")
            new_password = st.text_input("New Password", type="password", key="reset_new_password")
            confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
            submitted_reset = st.form_submit_button("Reset Password")

    if "submitted_reset" in locals() and submitted_reset:
        if not all([username.strip(), new_password, confirm_password]):
            st.error("Please fill all fields.")
            return

        if new_password != confirm_password:
            st.error("New Password and Confirm Password do not match.")
            return

        users = load_users()
        user_key = username.strip().lower()

        if user_key not in users:
            st.error("User not found.")
            return

        if len(new_password) < 8:
            st.error("New password must be at least 8 characters.")
            return

        users[user_key]["password_hash"] = hash_password(new_password)
        save_users(users)
        st.success("Password reset successful. Please sign in with your new password.")


def render_contact():
    render_section_intro(
        "Contact The Project Team",
        "Use this form for project questions, prediction workflow suggestions, or support requests.",
    )

    form_col, side_col = st.columns([1.15, 0.85])

    with form_col:
        with st.form("contact_form", clear_on_submit=True):
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message", height=120)
            submitted = st.form_submit_button("Send")

    with side_col:
        render_info_card(
            "Useful Topics",
            "Share questions about prediction flow, booking, reports, model output, or account access.",
            kicker="Examples",
        )

    if submitted:
        if not all([name.strip(), email.strip(), message.strip()]):
            st.error("Please complete all contact details.")
        else:
            st.success("Thanks for contacting us. We received your message.")


def render_prediction(model_metadata, preprocessing_data, label_encoder, feature_names):
    model = model_metadata["model"]
    symptom_remedies_df = load_symptom_remedies()

    render_section_intro(
        "Predict",
        "Choose symptoms, run screening, then review care, reports, charts, and booking.",
    )
    st.caption("The latest prediction stays visible while booking controls remain available.")

    workspace_chat_col, workspace_control_col = st.columns([1.3, 0.9], gap="large")

    with workspace_chat_col:
        render_symptom_assistant(model, label_encoder, feature_names)

    with workspace_control_col:
        render_section_intro(
            "Symptoms",
            "Use the chatbot for guided symptom capture or choose symptoms manually.",
        )
        render_info_card(
            "Manual Symptom Picker",
            "Select symptoms yourself, or let the chatbot detect symptoms and send them here for you.",
            kicker="Input Flow",
        )
        selected_symptoms_input = st.multiselect(
            "Choose symptoms",
            feature_names,
            max_selections=20,
            key="prediction_symptoms",
            format_func=format_symptom_label,
        )
        predict_button = st.button("Run Prediction", type="primary", width="stretch")

        if st.session_state.assistant_detected_symptoms:
            st.caption("The chatbot has already collected symptoms. Open it from the left to load or predict from them instantly.")

    st.divider()
    if st.session_state.assistant_trigger_predict:
        predict_button = True
        st.session_state.assistant_trigger_predict = False

    if predict_button:
        if len(selected_symptoms_input) == 0:
            st.error("Please select at least one symptom.")
            return

        prediction_result = predict_from_symptom_list(model, label_encoder, feature_names, selected_symptoms_input)
        guard_notes = st.session_state.get("assistant_clinical_notes", {}) or {}
        if should_guard_nonspecific_prediction(prediction_result, selected_symptoms_input, guard_notes):
            st.session_state.prediction_context = {
                "guarded": True,
                "selected_symptoms": list(selected_symptoms_input),
                "predicted_disease": prediction_result["predicted_disease"],
                "confidence": float(prediction_result["prediction_percent"]),
                "guard_message": build_nonspecific_prediction_guard_reply(
                    prediction_result,
                    selected_symptoms_input,
                    guard_notes,
                ),
            }
        else:
            st.session_state.prediction_context = {
                "guarded": False,
                "selected_symptoms": list(selected_symptoms_input),
                "prediction": int(prediction_result["prediction"]),
                "probabilities": prediction_result["probabilities"].tolist(),
                "class_labels": np.asarray(prediction_result["class_labels"]).astype(int).tolist(),
                "predicted_disease": prediction_result["predicted_disease"],
                "confidence": float(prediction_result["prediction_percent"]),
            }

    prediction_context = st.session_state.prediction_context
    if not prediction_context:
        return

    if prediction_context.get("guarded"):
        render_section_intro(
            "Prediction Needs More Detail",
            "The selected symptoms are too nonspecific for a reliable disease label.",
        )
        st.warning(prediction_context.get("guard_message", "Please add more specific symptoms before running prediction."))
        st.caption("The previous emergency-style prediction was suppressed because the saved symptom pattern does not support it.")
        return

    selected_symptoms = prediction_context["selected_symptoms"]
    prediction = int(prediction_context["prediction"])
    probabilities = np.array(prediction_context["probabilities"], dtype=float)
    class_labels = np.array(prediction_context["class_labels"], dtype=int)
    predicted_disease = prediction_context["predicted_disease"]
    confidence = float(prediction_context["confidence"])
    top_probability_df = build_top_prediction_dataframe(label_encoder, class_labels, probabilities)
    care_plan = get_disease_care_plan(predicted_disease, confidence)

    if set(selected_symptoms_input) != set(selected_symptoms):
        st.caption("The result below is based on your last prediction. Click Run Prediction again to refresh it with the current symptom selection.")

    render_section_intro(
        "Prediction Result",
        "Primary result, selected symptom count, and next-step care guidance.",
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Disease", predicted_disease)
    with col2:
        st.metric("Symptoms Selected", len(selected_symptoms))
    with col3:
        st.metric("Care Level", care_plan["severity"])
    with col4:
        st.metric("Suggested Doctor", care_plan["specialist"])

    st.caption("Medical confirmation still requires a qualified clinician. Use this as screening support, not a final diagnosis.")
    external_evidence = get_external_prediction_evidence(predicted_disease, tuple(selected_symptoms))
    render_external_evidence_panel(external_evidence)

    render_section_intro(
        "Care Guide",
        "Severity, doctor need, and care guidance.",
    )
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Severity", care_plan["severity"])
    with metric_col2:
        st.metric("Doctor Consult", care_plan["doctor_required"])
    with metric_col3:
        st.metric("Suggested Doctor", care_plan["specialist"])

    if care_plan["severity"] == "Emergency":
        st.error("Emergency scenario detected. Please seek immediate hospital care.")
    elif "Serious" in care_plan["severity"]:
        st.warning("This predicted disease may need prompt doctor review and confirmatory testing.")
    elif care_plan["severity"] == "Mild":
        st.success("This looks like a lighter condition in many routine cases. Home care may be enough if symptoms stay mild.")
    else:
        st.info("This prediction should be correlated with a doctor review if symptoms persist or become more intense.")

    nav_col1, nav_col2, nav_col3 = st.columns(3, gap="large")
    with nav_col1:
        render_action_card(
            "Connect with Doctor",
            f"{care_plan['specialist']} recommendations appear inside this dashboard.",
            "mint",
        )
    with nav_col2:
        render_action_card(
            "Find Hospital",
            "Nearby hospital listings use public maps first, with optional fallback providers.",
            "sky",
        )
    with nav_col3:
        render_action_card(
            "Track Appointments",
            "Saved doctor and lab bookings appear in the Appointments page.",
            "gold",
        )

    render_section_intro(
        "Doctor Booking",
        "Find a doctor near you, then save the appointment details here.",
    )
    if care_plan["severity"] == "Emergency":
        st.error("Online booking is disabled for emergency predictions. Please go to the nearest emergency hospital immediately.")
    else:
        location_text = st.text_input(
            "Enter your city or area",
            placeholder="Type any Indian city or area, example: ahm, gota, madhapur",
            key="doctor_search_location",
        )
        typed_location = location_text.strip()
        maps_key_token = get_map_provider_cache_token()
        city_suggestions = get_city_suggestions(location_text)
        selected_city = ""
        if len(normalize_location_query(location_text)) >= 2:
            if city_suggestions:
                city_preview_html = "".join(
                    f"<span class='city-suggestion-pill'>{html.escape(city)}</span>"
                    for city in city_suggestions
                )
                st.markdown(
                    f"""
                    <div class="city-suggestion-panel">
                        <p class="city-suggestion-title">Matching city and area suggestions</p>
                        <div class="city-suggestion-list">{city_preview_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                location_options = []
                if typed_location:
                    location_options.append(typed_location)
                for suggestion in city_suggestions:
                    if normalize_location_query(suggestion) not in {
                        normalize_location_query(option) for option in location_options
                    }:
                        location_options.append(suggestion)
                selected_city = st.selectbox(
                    "Location used for search",
                    location_options,
                    key="doctor_city_suggestion",
                )
            else:
                st.caption("No saved match found yet. You can still search any Indian city or area with the text you typed.")
        else:
            st.caption("Type at least 2 letters to show India-wide city and area suggestions.")

        search_location = selected_city or typed_location
        current_provider_context = (
            f"{normalize_location_query(search_location)}|{care_plan['specialist']}|{predicted_disease}|{maps_key_token}"
        )
        if st.session_state.provider_location_fingerprint != current_provider_context:
            previous_provider_was_selected = bool(
                st.session_state.selected_booking_provider
                or st.session_state.selected_booking_provider_context
                or st.session_state.provider_autofill_notice
            )
            st.session_state.provider_search_results = None
            st.session_state.provider_search_context = ""
            st.session_state.selected_booking_provider = None
            st.session_state.selected_booking_provider_context = ""
            st.session_state.provider_autofill_notice = ""
            st.session_state.pop("selected_provider_for_booking", None)
            if previous_provider_was_selected:
                st.session_state.booking_real_doctor_name = ""
                st.session_state.booking_real_hospital = ""
            st.session_state.provider_location_fingerprint = current_provider_context

        search_col1, search_col2 = st.columns(2)
        with search_col1:
            search_doctors_clicked = st.button(
                "Search Doctors",
                type="primary",
                width="stretch",
                key="search_local_doctors_button",
            )
        with search_col2:
            search_hospitals_clicked = st.button(
                "Find Hospitals",
                width="stretch",
                key="search_local_hospitals_button",
            )

        if search_doctors_clicked:
            if len(normalize_location_query(search_location)) < 2:
                st.warning("Please enter a city or area before searching.")
            else:
                with st.spinner(f"Finding {care_plan['specialist']} care near {search_location}..."):
                    st.session_state.provider_search_results = search_nearby_healthcare(
                        search_location,
                        care_plan["specialist"],
                        predicted_disease,
                        "doctors",
                        maps_key_token,
                    )
                    st.session_state.provider_search_context = current_provider_context

        if search_hospitals_clicked:
            if len(normalize_location_query(search_location)) < 2:
                st.warning("Please enter a city or area before searching.")
            else:
                with st.spinner(f"Finding hospitals near {search_location}..."):
                    st.session_state.provider_search_results = search_nearby_healthcare(
                        search_location,
                        care_plan["specialist"],
                        predicted_disease,
                        "hospitals",
                        maps_key_token,
                    )
                    st.session_state.provider_search_context = current_provider_context

        provider_results = None
        if (
            st.session_state.provider_search_results
            and st.session_state.provider_search_context == current_provider_context
        ):
            provider_results = st.session_state.provider_search_results

        if provider_results:
            origin = provider_results.get("origin") or {}
            if origin:
                radius_copy = format_search_radius(provider_results.get("search_radius_m"))
                radius_sentence = f" Search radius: about {html.escape(radius_copy)}." if radius_copy else ""
                care_focus = provider_results.get("care_focus") or get_provider_care_focus(care_plan["specialist"], predicted_disease)
                st.markdown(
                    (
                        '<div class="provider-source-note">'
                        f"Showing {html.escape(care_focus.lower())} listings near {html.escape(clean_provider_text(origin.get('display_name'), search_location))}. "
                        f"Source: {html.escape(provider_results.get('source', 'public map listings'))}."
                        f"{radius_sentence}"
                        " Click a card to open its map location."
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            if provider_results.get("error"):
                st.warning(provider_results["error"])

            if provider_results.get("search_mode") == "hospitals":
                render_provider_results(
                    f"Recommended Hospitals for {predicted_disease}",
                    provider_results.get("hospitals", []),
                    "No named hospital listing was found for this area. Try a nearby larger city or enter the hospital manually.",
                )
                render_provider_results(
                    f"Nearby Doctors and Clinics for {care_plan['specialist']}",
                    provider_results.get("doctors", []),
                    "No named doctor or clinic listing was found for this area. Try a nearby area or city name.",
                )
            else:
                render_provider_results(
                    f"Recommended Doctors and Clinics for {care_plan['specialist']}",
                    provider_results.get("doctors", []),
                    "No named doctor or clinic listing was found for this area. Try a nearby area or city name.",
                )
                render_provider_results(
                    f"Recommended Hospitals for {predicted_disease}",
                    provider_results.get("hospitals", []),
                    "No named hospital listing was found for this area. Try a nearby larger city or enter the hospital manually.",
                )

            all_recommendations = provider_results.get("doctors", []) + provider_results.get("hospitals", [])
            if all_recommendations:
                provider_options = {
                    f"{idx + 1}. {format_provider_option(provider)}": provider
                    for idx, provider in enumerate(all_recommendations)
                }
                selected_provider_label = st.selectbox(
                    "Select a recommendation for booking",
                    list(provider_options.keys()),
                    key="selected_provider_for_booking",
                )
                if st.button("Use Selected Recommendation", type="primary", key="use_provider_recommendation"):
                    selected_provider = provider_options[selected_provider_label]
                    doctor_autofill, hospital_autofill = build_booking_fields_from_provider(
                        selected_provider,
                        care_plan["specialist"],
                    )
                    st.session_state.booking_real_doctor_name = doctor_autofill
                    st.session_state.booking_real_hospital = hospital_autofill
                    st.session_state.selected_booking_provider = selected_provider
                    st.session_state.selected_booking_provider_context = current_provider_context
                    selected_provider_name = clean_provider_text(selected_provider.get("name"), "selected provider")
                    st.session_state.provider_autofill_notice = (
                        f"Added booking details from {selected_provider_name}. Fill any remaining blank field manually."
                    )
            else:
                st.info("No named map listing was found. You can still enter doctor and hospital details manually.")
        elif len(normalize_location_query(search_location)) >= 2:
            st.caption("Click Search Doctors or Find Hospitals to load recommendations inside the dashboard.")

        active_booking_provider = get_active_booking_provider(current_provider_context)
        if st.session_state.provider_autofill_notice and active_booking_provider:
            st.success(st.session_state.provider_autofill_notice)

        st.info("Map listings may show clinic or hospital names when individual doctor names are not published. Selected listings are accepted as booking provider details.")

        selected_doctor_name = st.text_input(
            "Doctor",
            placeholder="Enter doctor name",
            key="booking_real_doctor_name",
        )
        selected_hospital = st.text_input(
            "Hospital or Clinic",
            placeholder="Enter hospital, clinic, or telemedicine platform",
            key="booking_real_hospital",
        )
        booking_mode = st.selectbox(
            "Visit Type",
            ["In-person", "Video", "Phone"],
            key="booking_mode_choice",
        )
        booking_date = st.date_input(
            "Preferred Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            key="booking_date_choice",
        )
        booking_slot = st.text_input(
            "Appointment Time",
            placeholder="Example: 10:30 AM",
            key="booking_slot_choice",
        )
        default_note = (
            f"Predicted disease: {predicted_disease}. "
            f"Symptoms: {', '.join(format_symptom_label(symptom) for symptom in selected_symptoms)}."
        )
        booking_reason = st.text_area(
            "Reason for Consultation",
            value=default_note,
            height=110,
            key="booking_reason_note",
        )

        if care_plan["doctor_required"] == "Usually not":
            st.info("Doctor consultation is optional for this likely mild case, but you can still book if you want medical confirmation.")
        else:
            st.warning(f"Recommended specialist: {care_plan['specialist']}")

        if st.button("Save Appointment", type="primary", key="book_appointment_button"):
            booking_doctor_name = selected_doctor_name.strip()
            booking_hospital_name = selected_hospital.strip()
            if active_booking_provider:
                doctor_autofill, hospital_autofill = build_booking_fields_from_provider(
                    active_booking_provider,
                    care_plan["specialist"],
                )
                booking_doctor_name = booking_doctor_name or doctor_autofill
                booking_hospital_name = booking_hospital_name or hospital_autofill

            missing_fields = []
            if not booking_doctor_name:
                missing_fields.append("doctor")
            if not booking_hospital_name:
                missing_fields.append("hospital or clinic name")
            if not booking_slot.strip():
                missing_fields.append("appointment time")

            if missing_fields:
                st.error(f"Please enter {', '.join(missing_fields)}.")
                return

            current_user = st.session_state.current_user or {}
            username = current_user.get("username", "").strip().lower()
            patient_name = f"{current_user.get('first_name', '').strip()} {current_user.get('last_name', '').strip()}".strip()
            appointment_date_value = booking_date.isoformat()

            slot_taken = is_appointment_slot_taken(
                booking_doctor_name,
                appointment_date_value,
                booking_slot.strip(),
            )
            if slot_taken:
                st.error("That time slot has already been booked. Please choose another slot.")
            else:
                appointment_id = build_appointment_id()
                save_appointment(
                    {
                        "appointment_id": appointment_id,
                        "username": username,
                        "patient_name": patient_name,
                        "predicted_disease": predicted_disease,
                        "specialist": care_plan["specialist"],
                        "doctor_name": booking_doctor_name,
                        "hospital": booking_hospital_name,
                        "consultation_mode": booking_mode,
                        "appointment_date": appointment_date_value,
                        "appointment_slot": booking_slot.strip(),
                        "status": "Booked",
                        "booked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "symptoms": [format_symptom_label(symptom) for symptom in selected_symptoms],
                        "reason": booking_reason.strip(),
                    }
                )
                st.success(
                    f"Appointment saved with {booking_doctor_name} on {appointment_date_value} at {booking_slot.strip()}. Appointment ID: {appointment_id}"
                )

        current_user = st.session_state.current_user or {}
        username = current_user.get("username", "").strip().lower()
        user_appointments = [
            item for item in load_appointments() if item.get("username", "").strip().lower() == username and item.get("status") == "Booked"
        ]
        if user_appointments:
            render_section_intro(
                "Your Active Bookings",
                "Recent open appointments for the signed-in user.",
            )
            recent_bookings = sorted(
                user_appointments,
                key=lambda item: (item.get("appointment_date", ""), item.get("appointment_slot", "")),
                reverse=True,
            )[:5]
            render_themed_dataframe(format_appointment_rows(recent_bookings))

    render_section_intro(
        "Prediction Charts",
        "Compact ranked views for the most likely conditions.",
    )
    fig_ranked_bar = px.bar(
        top_probability_df,
        x="Model Score",
        y="Disease",
        orientation="h",
        color="Model Score",
        color_continuous_scale=["#d6f3ee", "#8dd8c8", "#4d9f92", "#273d52"],
    )
    fig_ranked_bar.update_traces(
        marker_line_width=0,
        hovertemplate="%{y}<extra></extra>",
    )
    fig_ranked_bar.update_yaxes(autorange="reversed")
    fig_ranked_bar.update_xaxes(
        range=[0, max(top_probability_df["Model Score"].max() * 1.15, 5)],
        title_text="Relative score",
        showticklabels=False,
    )
    fig_ranked_bar.update_layout(coloraxis_showscale=False)
    apply_warm_compact_plot_style(fig_ranked_bar, height=260)

    fig_probability_scatter = go.Figure(
        go.Scatter(
            x=list(range(1, len(top_probability_df) + 1)),
            y=top_probability_df["Model Score"],
            mode="lines+markers",
            line=dict(color="#22c7a9", width=4, shape="spline"),
            marker=dict(
                size=13,
                color=["#5a63d8", "#22c7a9", "#f5b642", "#ff7f9c", "#9aa2ff"][: len(top_probability_df)],
                line=dict(color="#ffffff", width=3),
            ),
            text=top_probability_df["Disease"],
            hovertemplate="Rank %{x}<br>%{text}<extra></extra>",
        )
    )
    fig_probability_scatter.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, len(top_probability_df) + 1)),
        title_text="Rank",
    )
    fig_probability_scatter.update_yaxes(title_text="Relative score", showticklabels=False)
    apply_warm_compact_plot_style(fig_probability_scatter, height=260)

    chart_bar_col, chart_scatter_col = st.columns(2, gap="large")
    with chart_bar_col:
        render_small_plot_card("Most Likely Conditions", "Visual comparison of possible conditions.", fig_ranked_bar)
    with chart_scatter_col:
        render_small_plot_card("Condition Ranking", "Relative ordering among possible conditions.", fig_probability_scatter)

    render_section_intro(
        "Symptom-wise Home Remedies",
        "Matched care tips appear as light responsive rows for the symptoms selected in this prediction.",
    )
    selected_symptoms_lower = [s.lower() for s in selected_symptoms]
    matched = symptom_remedies_df[symptom_remedies_df["symptom"].isin(selected_symptoms_lower)]
    if matched.empty:
        st.info("No symptom-wise remedy found in dataset for selected symptoms.")
    else:
        remedy_records = []
        safety_note = "Use these as supportive steps. Seek medical care if symptoms worsen, persist, or feel unusual."
        if care_plan["severity"] == "Emergency":
            safety_note = "Do not delay emergency care. Use home steps only while arranging urgent help."
        elif care_plan["doctor_required"] == "Yes":
            safety_note = "Use as supportive care while arranging the recommended doctor review."

        def _remedy_purpose(remedy_items):
            remedy_text = normalize_text_for_matching(" ".join(remedy_items))
            if any(term in remedy_text for term in ("water", "fluid", "hydrate", "ors")):
                return "Hydration and recovery support"
            if any(term in remedy_text for term in ("skin", "scratch", "soap", "moistur", "calamine")):
                return "Skin comfort and irritation control"
            if any(term in remedy_text for term in ("food", "meal", "diet", "spicy", "oily", "bland")):
                return "Diet comfort and digestion support"
            if any(term in remedy_text for term in ("warm", "steam", "compress")):
                return "Comfort care and symptom relief"
            if any(term in remedy_text for term in ("rest", "sleep")):
                return "Rest and symptom monitoring"
            return "Supportive symptom care"

        remedy_rows_html = []
        for _, row in matched.drop_duplicates(subset=["symptom"]).sort_values("symptom").iterrows():
            remedies = [item.strip() for item in str(row["home_remedies"]).split(";") if item.strip()]
            symptom_name = row["symptom"].replace("_", " ").title()
            remedies = remedies[:5] or ["General care guidance unavailable"]
            remedy_records.append((symptom_name, remedies, _remedy_purpose(remedies)))
            remedy_steps_html = "".join(
                (
                    "<li class='remedy-step'>"
                    f"<span class='remedy-step-index'>{idx}</span>"
                    f"<span>{html.escape(item)}</span>"
                    "</li>"
                )
                for idx, item in enumerate(remedies, 1)
            )
            remedy_rows_html.append(
                "<tr>"
                f"<td data-label='Symptom'><div class='remedy-symptom-cell'><span class='remedy-symptom-name'>{html.escape(symptom_name)}</span><span class='remedy-symptom-tag'>Selected symptom</span></div></td>"
                f"<td data-label='Home care steps'><ul class='remedy-step-list'>{remedy_steps_html}</ul></td>"
                f"<td data-label='Purpose'><div class='remedy-purpose'><strong>Why this helps</strong>{html.escape(_remedy_purpose(remedies))}</div></td>"
                f"<td data-label='Safety'><div class='remedy-safety'><strong>Safety note</strong>{html.escape(safety_note)}</div></td>"
                "</tr>"
            )
        total_steps = sum(len(record[1]) for record in remedy_records)
        summary_html = f"""
            <div class="remedy-summary-strip">
                <div class="remedy-summary-card">
                    <div class="remedy-summary-label">Matched Symptoms</div>
                    <div class="remedy-summary-value">{len(remedy_records)}</div>
                </div>
                <div class="remedy-summary-card">
                    <div class="remedy-summary-label">Care Steps</div>
                    <div class="remedy-summary-value">{total_steps}</div>
                </div>
                <div class="remedy-summary-card">
                    <div class="remedy-summary-label">Care Level</div>
                    <div class="remedy-summary-value">{html.escape(care_plan["severity"])}</div>
                </div>
            </div>
        """
        st.markdown(
            (
                summary_html
                + "<div class='remedy-table'>"
                + "<table class='remedy-care-table'>"
                + "<colgroup><col><col><col><col></colgroup>"
                + "<thead><tr><th>Symptom</th><th>Home Care Steps</th><th>Purpose</th><th>Safety Note</th></tr></thead>"
                + f"<tbody>{''.join(remedy_rows_html)}</tbody>"
                + "</table>"
                + "</div>"
                + "<div class='remedy-table-note'>Home remedies are supportive care only. Follow medical advice and use the doctor, hospital, or lab sections when symptoms are severe or persistent.</div>"
            ),
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Predicted Disease:** {predicted_disease}")
        st.write(f"**Doctor Consultation Required:** {care_plan['doctor_required']}")
        st.write(f"**Suggested Specialist:** {care_plan['specialist']}")
        st.write(f"**When to Consult:** {care_plan['doctor_urgency']}")
        st.write("**Home Remedies:**")
        for item in care_plan["home_remedies"]:
            st.write(f"- {item}")

    with c2:
        st.write("**Medicines:**")
        for item in care_plan["medicines"]:
            st.write(f"- {item}")
        st.write("**Consultation Note:**")
        st.info(care_plan["note"])


def render_dashboard(model_metadata, preprocessing_data, label_encoder, feature_names):
    render_section_intro(
        "Prediction Command Center",
        "Use one focused workspace for prediction, clinical guidance, diagnostics, and doctor consultation.",
    )

    render_prediction(model_metadata, preprocessing_data, label_encoder, feature_names)


def main():
    st.set_page_config(page_title="Disease Prediction System", layout="wide", initial_sidebar_state="auto")
    components.html("""
    <script>
    (() => {
        const parentDoc = window.parent.document;
        const root = parentDoc.documentElement;
        const sidebarSelector = 'section[data-testid="stSidebar"]';
        const sidebarActionSelector = [
            `${sidebarSelector} button`,
            `${sidebarSelector} a`,
            `${sidebarSelector} [role="button"]`
        ].join(',');
        const sidebarNavigationSelector = `${sidebarSelector} [role="radiogroup"]`;

        const sidebarControlSelector = [
            '[data-testid="stSidebarCollapsedControl"]',
            '[data-testid="stSidebarCollapseButton"]',
            '[data-testid="stExpandSidebarButton"]',
            'button[aria-label*="sidebar" i]',
            'button[title*="sidebar" i]'
        ].join(',');
        const autoCloseVersion = '2026-06-28-sidebar-autoclose-v6';
        const pendingCloseKey = 'dp-sidebar-pending-close-at';
        const sidebarAutoCloseDelay = 760;
        let suppressOpeningUntil = 0;

        const readButtonMeta = (button) => {
            const aria = (button?.getAttribute('aria-label') || '').toLowerCase();
            const title = (button?.getAttribute('title') || '').toLowerCase();
            const testid = (button?.getAttribute('data-testid') || '').toLowerCase();
            const kind = (button?.getAttribute('kind') || '').toLowerCase();
            return { aria, title, testid, kind };
        };

        const isSidebarOpenButton = (button) => {
            if (!button) return false;
            const { aria, title, testid } = readButtonMeta(button);
            return (
                aria.includes('open sidebar') ||
                aria.includes('expand sidebar') ||
                title.includes('open sidebar') ||
                title.includes('expand sidebar') ||
                testid.includes('expandsidebar') ||
                testid.includes('sidebarcollapsedcontrol')
            );
        };

        const isSidebarCloseButton = (button) => {
            if (!button) return false;
            const { aria, title, testid, kind } = readButtonMeta(button);
            return (
                aria.includes('close sidebar') ||
                aria.includes('collapse sidebar') ||
                title.includes('close sidebar') ||
                title.includes('collapse sidebar') ||
                testid.includes('sidebarcollapsebutton') ||
                (kind === 'header' && !aria.includes('expand') && !aria.includes('open'))
            );
        };

        const clearSidebarMotion = () => {
            root.classList.remove('dp-sidebar-opening', 'dp-sidebar-closing');
        };

        const markSidebarOpening = () => {
            root.classList.remove('dp-sidebar-closing');
            root.classList.add('dp-sidebar-opening');
            window.clearTimeout(window.__dpSidebarMotionTimer);
            window.__dpSidebarMotionTimer = window.setTimeout(clearSidebarMotion, 720);
        };

        const markSidebarClosing = () => {
            root.classList.remove('dp-sidebar-opening');
            root.classList.add('dp-sidebar-closing');
            suppressOpeningUntil = Date.now() + 1800;
            window.clearTimeout(window.__dpSidebarMotionTimer);
            window.__dpSidebarMotionTimer = window.setTimeout(clearSidebarMotion, 920);
        };

        const rememberSidebarCloseAfterRerun = () => {
            try {
                window.parent.localStorage.setItem(pendingCloseKey, String(Date.now()));
            } catch (error) {
                // Local storage can be unavailable in embedded contexts.
            }
        };

        const consumeSidebarCloseAfterRerun = () => {
            try {
                const savedAt = Number(window.parent.localStorage.getItem(pendingCloseKey) || 0);
                window.parent.localStorage.removeItem(pendingCloseKey);
                if (savedAt && Date.now() - savedAt < 6000) {
                    suppressOpeningUntil = Date.now() + 1800;
                    return true;
                }
            } catch (error) {
                return false;
            }
            return false;
        };

        const clickCloseSidebarButton = () => {
            const buttons = Array.from(parentDoc.querySelectorAll('button'));
            const closeButton = buttons.find(isSidebarCloseButton);
            if (!closeButton) return false;
            closeButton.click();
            return true;
        };

        const requestSidebarClose = (persistForRerun = true) => {
            if (persistForRerun) {
                rememberSidebarCloseAfterRerun();
            }
            markSidebarClosing();
            [120, 260, 460, 720].forEach((delay) => {
                window.setTimeout(clickCloseSidebarButton, delay);
            });
        };

        if (consumeSidebarCloseAfterRerun()) {
            window.setTimeout(() => requestSidebarClose(false), 260);
            window.setTimeout(() => requestSidebarClose(false), 620);
        }

        if (window.parent.__dpSidebarAutoCloseVersion !== autoCloseVersion) {
            window.parent.__dpSidebarAutoCloseVersion = autoCloseVersion;

            if (window.parent.__dpSidebarAutoCloseHandler) {
                parentDoc.removeEventListener('click', window.parent.__dpSidebarAutoCloseHandler, true);
                parentDoc.removeEventListener('click', window.parent.__dpSidebarAutoCloseHandler, false);
            }

            const handleSidebarAutoClose = (event) => {
                const directSidebarControl = event.target.closest(sidebarControlSelector);
                if (directSidebarControl && isSidebarOpenButton(directSidebarControl)) {
                    markSidebarOpening();
                    return;
                }
                if (directSidebarControl && isSidebarCloseButton(directSidebarControl)) {
                    markSidebarClosing();
                    return;
                }

                const action = event.target.closest(sidebarActionSelector);
                const sidebar = parentDoc.querySelector(sidebarSelector);
                if (!action || !sidebar || !sidebar.contains(action)) return;
                if (action.closest(sidebarControlSelector)) return;
                if (action.closest(sidebarNavigationSelector)) return;

                window.setTimeout(() => requestSidebarClose(true), sidebarAutoCloseDelay);
            };

            window.parent.__dpSidebarAutoCloseHandler = handleSidebarAutoClose;
            parentDoc.addEventListener('click', handleSidebarAutoClose, false);

            let sidebarWasPresent = Boolean(parentDoc.querySelector(sidebarSelector));
            const observer = new MutationObserver(() => {
                const sidebarIsPresent = Boolean(parentDoc.querySelector(sidebarSelector));
                const openingIsSuppressed = Date.now() < suppressOpeningUntil;
                if (sidebarIsPresent && !sidebarWasPresent && !openingIsSuppressed && !root.classList.contains('dp-sidebar-closing')) {
                    markSidebarOpening();
                }
                sidebarWasPresent = sidebarIsPresent;
            });
            observer.observe(parentDoc.body, { childList: true, subtree: true });
        }
    })();
    </script>
    """, height=0, width=0)
    init_session()
    inject_custom_styles()
    ensure_database_ready()
    model_metadata, preprocessing_data = load_model_and_data(get_model_artifact_signature())
    label_encoder = model_metadata["label_encoder"]
    feature_names = model_metadata["feature_names"]
    X_train = pd.DataFrame(preprocessing_data["X_train_split"], columns=feature_names)

    selected_page = render_navbar()
    if st.session_state.get("close_sidebar_after_nav"):
        close_sidebar_after_navigation()
        st.session_state.close_sidebar_after_nav = False

    if selected_page == "Home":
        render_home(model_metadata, label_encoder, feature_names, X_train)
    elif selected_page == "Dashboard":
        if not st.session_state.logged_in:
            st.warning("Please login first.")
        else:
            render_dashboard(model_metadata, preprocessing_data, label_encoder, feature_names)
    elif selected_page == "Appointments":
        if not st.session_state.logged_in:
            st.warning("Please login first.")
        else:
            render_appointments_page()
    elif selected_page == "Lab Tests":
        if not st.session_state.logged_in:
            st.warning("Please login first.")
        else:
            render_lab_tests_page()
    elif selected_page == "About":
        render_about()
    elif selected_page == "Register":
        render_register()
    elif selected_page == "Login":
        render_login()
    elif selected_page == "Contact Us":
        render_contact()


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            print("Run this app with Streamlit, not plain Python.")
            print("Use: streamlit run s06_dash.py")
            sys.exit(0)
    except Exception:
        print("Run this app with Streamlit.")
        print("Use: streamlit run s06_dash.py")
        sys.exit(0)

    main()
