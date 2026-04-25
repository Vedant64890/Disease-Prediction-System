import hashlib
import hmac
import json
import os
import pickle
from urllib.parse import quote_plus
import warnings
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

USERS_FILE = "users.json"


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


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


@st.cache_resource
def load_model_and_data():
    with open("best_model.pkl", "rb") as f:
        model_metadata = pickle.load(f)
    with open("preprocessing_data.pkl", "rb") as f:
        preprocessing_data = pickle.load(f)
    return model_metadata, preprocessing_data


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
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "page" not in st.session_state:
        st.session_state.page = "Home"


def render_navbar():
    left_col, right_col = st.columns([7, 5])
    with left_col:
        st.title("Disease Prediction System")

    nav_options = ["Home", "About", "Register", "Login", "Contact Us"]
    if st.session_state.logged_in:
        nav_options = ["Home", "Dashboard", "About", "Contact Us"]

    if st.session_state.page not in nav_options:
        st.session_state.page = nav_options[0]

    with right_col:
        page = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(st.session_state.page),
            horizontal=True,
            label_visibility="collapsed",
        )

    st.session_state.page = page

    if st.session_state.logged_in:
        name = st.session_state.current_user["first_name"] + " " + st.session_state.current_user["last_name"]
        st.caption(f"Signed in as {name}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.page = "Home"
            st.rerun()

    st.divider()
    return page


def render_home(model_metadata, label_encoder, feature_names, X_train):
    st.subheader("Home")
    st.write("Welcome to the Disease Prediction System.")
    st.write("It predicts likely diseases from symptoms and provides interactive analytics similar to a Power BI dashboard.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Diseases", len(label_encoder.classes_))
    with col2:
        st.metric("Total Symptoms", len(feature_names))
    with col3:
        st.metric("Model Accuracy", f"{model_metadata['accuracy']:.1%}")
    with col4:
        st.metric("Training Samples", X_train.shape[0])

    st.write("### System Description")
    st.write("- Uses machine learning trained on symptom data")
    st.write("- Provides simple bar and pie visualizations for prediction insights")
    st.write("- Shows ranked disease predictions with confidence")
    st.write("- Displays symptom cues and basic disease measures")

    st.divider()

    if st.session_state.logged_in:
        st.success("Login successful. Go to Dashboard for predictions and interactive analytics.")
    else:
        st.info("Please register/login to access the Dashboard page.")


def render_about():
    st.subheader("About Us")
    st.write("This application is designed for educational and screening support.")
    st.write("It helps users explore likely diseases from symptoms using a trained model.")
    st.write("Predictions are not a medical diagnosis. Always consult a qualified doctor.")


def render_register():
    st.subheader("Register")
    st.write("User Management - Create your account")

    with st.form("register_form", clear_on_submit=True):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account")

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
    st.subheader("Login")
    st.write("Sign in to perform operations")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")

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

    st.divider()
    with st.expander("Forgot password? Reset here"):
        with st.form("reset_password_form", clear_on_submit=True):
            username = st.text_input("Username", key="reset_username")
            first_name = st.text_input("First Name", key="reset_first_name")
            last_name = st.text_input("Last Name", key="reset_last_name")
            new_password = st.text_input("New Password", type="password", key="reset_new_password")
            confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
            submitted_reset = st.form_submit_button("Reset Password")

        if submitted_reset:
            if not all([username.strip(), first_name.strip(), last_name.strip(), new_password, confirm_password]):
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

            user = users[user_key]
            if user.get("first_name", "").strip().lower() != first_name.strip().lower() or user.get("last_name", "").strip().lower() != last_name.strip().lower():
                st.error("Verification failed. Name details do not match.")
                return

            if len(new_password) < 8:
                st.error("New password must be at least 8 characters.")
                return

            users[user_key]["password_hash"] = hash_password(new_password)
            save_users(users)
            st.success("Password reset successful. Please sign in with your new password.")


def render_contact():
    st.subheader("Contact Us")
    st.write("Please share your query here.")

    with st.form("contact_form", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message", height=120)
        submitted = st.form_submit_button("Send")

    if submitted:
        if not all([name.strip(), email.strip(), message.strip()]):
            st.error("Please complete all contact details.")
        else:
            st.success("Thanks for contacting us. We received your message.")


def render_prediction(model_metadata, preprocessing_data, label_encoder, feature_names):
    model = model_metadata["model"]
    symptom_remedies_df = load_symptom_remedies()

    def predict_with_probabilities(feature_frame):
        prediction_value = np.asarray(model.predict(feature_frame))[0]
        if np.issubdtype(np.asarray([prediction_value]).dtype, np.floating):
            prediction_value = int(np.rint(prediction_value))

        class_count = len(label_encoder.classes_)
        prediction_value = int(np.clip(prediction_value, 0, class_count - 1))

        if hasattr(model, "predict_proba"):
            probabilities_value = model.predict_proba(feature_frame)[0]
            class_labels_value = model.classes_
        else:
            probabilities_value = np.zeros(class_count, dtype=float)
            probabilities_value[prediction_value] = 1.0
            class_labels_value = np.arange(class_count)

        return prediction_value, probabilities_value, class_labels_value

    st.header("Make a Prediction")
    st.write("Select symptoms and click PREDICT")

    selected_symptoms = st.multiselect("Choose symptoms:", feature_names, max_selections=20)
    predict_button = st.button("PREDICT", type="primary")

    if not predict_button:
        return

    if len(selected_symptoms) == 0:
        st.error("Please select at least one symptom.")
        return

    symptom_vector = {s: 0 for s in feature_names}
    for symptom in selected_symptoms:
        symptom_vector[symptom] = 1

    feature_df = pd.DataFrame([symptom_vector.values()], columns=feature_names)
    prediction, probabilities, class_labels = predict_with_probabilities(feature_df)

    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    predicted_class_pos = np.where(class_labels == prediction)[0][0]
    confidence = probabilities[predicted_class_pos] * 100

    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Primary Diagnosis", predicted_disease)
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    with col3:
        st.metric("Symptoms Selected", len(selected_symptoms))

    st.write("### Find Hospital")
    google_query = quote_plus(f"{predicted_disease} specialist hospital near me")
    hospital_search_url = f"https://www.google.com/search?q={google_query}"
    st.link_button("Find Hospital", hospital_search_url, width="stretch")

    # Show one high-confidence symptom pattern from training data for predicted disease
    X_train = pd.DataFrame(preprocessing_data["X_train_split"], columns=feature_names)
    y_train = pd.Series(preprocessing_data["y_train_split"], name="encoded_disease")
    train_disease_names = label_encoder.inverse_transform(y_train.astype(int).values)
    predicted_mask = train_disease_names == predicted_disease

    if predicted_mask.any():
        disease_symptom_pct = X_train.loc[predicted_mask, feature_names].mean().sort_values(ascending=False) * 100
        over_90 = disease_symptom_pct[disease_symptom_pct >= 90]

        if not over_90.empty:
            top_symptom = over_90.index[0]
            top_value = over_90.iloc[0]
            st.success(f"Symptom above 90% for {predicted_disease}: {top_symptom} ({top_value:.1f}%)")
        else:
            top_symptom = disease_symptom_pct.index[0]
            top_value = disease_symptom_pct.iloc[0]
            st.info(
                f"No symptom crosses 90% for {predicted_disease}. Highest is {top_symptom} ({top_value:.1f}%)."
            )

    st.write("### Prediction Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            symptom_scores = []
            for symptom in selected_symptoms:
                symptom_idx = feature_names.index(symptom)
                symptom_scores.append(feature_importances[symptom_idx])

            symptoms_chart_df = pd.DataFrame(
                {
                    "Symptom": selected_symptoms,
                    "InfluenceScore": symptom_scores,
                }
            ).sort_values("InfluenceScore", ascending=True)

            fig_symptoms = px.bar(
                symptoms_chart_df,
                x="InfluenceScore",
                y="Symptom",
                orientation="h",
                title="Selected Symptoms Influence (Bar Plot)",
                color="InfluenceScore",
                color_continuous_scale="Blues",
            )
            fig_symptoms.update_layout(height=420, xaxis_title="Influence Score", yaxis_title="Symptom")
        else:
            symptoms_chart_df = pd.DataFrame(
                {
                    "Symptom": sorted(selected_symptoms),
                    "Selected": [1] * len(selected_symptoms),
                }
            )
            fig_symptoms = px.bar(
                symptoms_chart_df,
                x="Selected",
                y="Symptom",
                orientation="h",
                title="Selected Symptoms (Bar Plot)",
                color="Selected",
                color_continuous_scale="Blues",
            )
            fig_symptoms.update_layout(height=420, xaxis_title="Selected", yaxis_title="Symptom")
        st.plotly_chart(fig_symptoms)

    with chart_col2:
        top_idx_for_pie = np.argsort(probabilities)[-5:][::-1]
        top_class_values = class_labels[top_idx_for_pie]
        disease_labels = label_encoder.inverse_transform(top_class_values).tolist()
        disease_values = (probabilities[top_idx_for_pie] * 100).tolist()
        others_value = max(0.0, 100.0 - sum(disease_values))

        if others_value > 0:
            disease_labels.append("Others")
            disease_values.append(others_value)

        fig_disease = px.pie(
            names=disease_labels,
            values=disease_values,
            title="Predicted Disease Distribution (Full 100%)",
        )
        fig_disease.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value:.2f}%<extra></extra>")
        fig_disease.update_layout(height=420)
        st.plotly_chart(fig_disease)

    st.write("### Ranked Predictions")
    top_idx = np.argsort(probabilities)[-10:][::-1]
    rows = []
    for rank, idx in enumerate(top_idx, 1):
        class_value = class_labels[idx]
        rows.append(
            {
                "Rank": rank,
                "Disease": label_encoder.inverse_transform([class_value])[0],
                "Confidence %": f"{(probabilities[idx] * 100):.2f}%",
            }
        )
    st.table(pd.DataFrame(rows))

    st.write("### Symptom and Cure")
    symptom_cues = {
        "high_fever": "May indicate severe infection or inflammation.",
        "headache": "Can indicate infection, stress, or neurological strain.",
        "chills": "Common fever-associated response.",
        "joint_pain": "Often associated with viral or inflammatory conditions.",
        "muscle_pain": "Common in systemic viral infections.",
        "skin_rash": "Can indicate viral or allergic response.",
        "vomiting": "May indicate gastrointestinal or systemic involvement.",
        "cough": "Suggests respiratory tract involvement.",
        "shortness_of_breath": "Could indicate respiratory compromise.",
        "wheezing": "Often linked to airway narrowing.",
    }

    symptom_table = pd.DataFrame(
        {
            "Symptom": sorted(selected_symptoms),
            "Indication": [symptom_cues.get(s, "Clinical correlation required.") for s in sorted(selected_symptoms)],
        }
    )
    st.dataframe(symptom_table, width="stretch")

    st.write("### Symptom-wise Home Remedies")
    selected_symptoms_lower = [s.lower() for s in selected_symptoms]
    matched = symptom_remedies_df[symptom_remedies_df["symptom"].isin(selected_symptoms_lower)]
    if matched.empty:
        st.info("No symptom-wise remedy found in dataset for selected symptoms.")
    else:
        remedy_rows = []
        for _, row in matched.drop_duplicates(subset=["symptom"]).iterrows():
            remedies = [item.strip() for item in str(row["home_remedies"]).split(";") if item.strip()]
            remedy_rows.append(
                {
                    "Symptom": row["symptom"].replace("_", " ").title(),
                    "Home Remedies": " | ".join(remedies),
                }
            )

        remedies_table = pd.DataFrame(remedy_rows).sort_values("Symptom").reset_index(drop=True)
        st.dataframe(remedies_table, width="stretch")

    cure_db = {
        "Dengue": {
            "home_remedies": ["Drink ORS and water", "Take proper rest", "Use light diet"],
            "medicines": ["Paracetamol (avoid aspirin)", "Doctor-advised anti-nausea medicine"],
            "doctor_required": "Yes",
            "note": "Consult doctor if persistent fever, bleeding, or severe weakness.",
        },
        "Malaria": {
            "home_remedies": ["Hydration", "Rest", "Monitor fever pattern"],
            "medicines": ["Antimalarial medicines only by doctor prescription"],
            "doctor_required": "Yes",
            "note": "Doctor consultation is required for confirmation and treatment.",
        },
        "Pneumonia": {
            "home_remedies": ["Rest", "Warm fluids", "Steam inhalation if advised"],
            "medicines": ["Antibiotics if prescribed", "Fever medicine as advised"],
            "doctor_required": "Yes",
            "note": "Breathing issues need clinical evaluation.",
        },
        "Bronchial Asthma": {
            "home_remedies": ["Avoid triggers (dust/smoke)", "Use clean air", "Breathing exercises"],
            "medicines": ["Rescue inhaler", "Controller inhaler (as prescribed)"],
            "doctor_required": "Yes",
            "note": "Use prescribed inhalers and consult if symptoms worsen.",
        },
        "Common Cold": {
            "home_remedies": ["Rest", "Warm water", "Salt-water gargle", "Steam inhalation"],
            "medicines": ["Paracetamol", "Cough syrup/decongestant if needed"],
            "doctor_required": "No (usually)",
            "note": "Consult doctor if symptoms persist > 5 days or become severe.",
        },
        "Typhoid": {
            "home_remedies": ["Hydration", "Soft diet", "Complete rest"],
            "medicines": ["Antibiotics only by doctor prescription"],
            "doctor_required": "Yes",
            "note": "Laboratory confirmation and medical treatment required.",
        },
        "Heart attack": {
            "home_remedies": ["No home remedy delay"],
            "medicines": ["Emergency treatment only"],
            "doctor_required": "Emergency",
            "note": "Call emergency services immediately.",
        },
    }

    cure_info = cure_db.get(
        predicted_disease,
        {
            "home_remedies": ["Rest", "Hydration", "Balanced light diet"],
            "medicines": ["Use medicines only after medical advice"],
            "doctor_required": "Depends",
            "note": "Consult a doctor for confirmed diagnosis and personalized treatment.",
        },
    )

    if confidence < 50 and cure_info["doctor_required"] == "No (usually)":
        doctor_required = "Yes (low confidence)"
    else:
        doctor_required = cure_info["doctor_required"]

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Predicted Disease:** {predicted_disease}")
        st.write(f"**Doctor Consultation Required:** {doctor_required}")
        st.write("**Home Remedies:**")
        for item in cure_info["home_remedies"]:
            st.write(f"- {item}")

    with c2:
        st.write("**Medicines:**")
        for item in cure_info["medicines"]:
            st.write(f"- {item}")
        st.write("**Consultation Note:**")
        st.info(cure_info["note"])

def render_dashboard(model_metadata, preprocessing_data, label_encoder, feature_names):
    st.subheader("Dashboard")
    st.write("Simple analytics dashboard")

    st.write("### Prediction Section")
    render_prediction(model_metadata, preprocessing_data, label_encoder, feature_names)
    st.divider()

    model = model_metadata["model"]
    X_train = pd.DataFrame(preprocessing_data["X_train_split"], columns=feature_names)
    y_train = pd.Series(preprocessing_data["y_train_split"], name="encoded_disease")

    disease_names = label_encoder.inverse_transform(y_train.astype(int).values)
    analysis_df = X_train.copy()
    analysis_df["Disease"] = disease_names

    disease_counts = analysis_df["Disease"].value_counts().reset_index()
    disease_counts.columns = ["Disease", "Cases"]

    st.write("### Machine Learning Model")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model", model_metadata.get("model_name", "Classifier"))
    with c2:
        st.metric("Validation Accuracy", f"{model_metadata.get('accuracy', 0):.1%}")
    with c3:
        st.metric("F1 Score", f"{model_metadata.get('f1_score', 0):.3f}")
    with c4:
        st.metric("Features", len(feature_names))

def main():
    st.set_page_config(page_title="Disease Prediction System", layout="wide")
    init_session()
    model_metadata, preprocessing_data = load_model_and_data()
    label_encoder = model_metadata["label_encoder"]
    feature_names = model_metadata["feature_names"]
    X_train = pd.DataFrame(preprocessing_data["X_train_split"], columns=feature_names)

    selected_page = render_navbar()

    if selected_page == "Home":
        render_home(model_metadata, label_encoder, feature_names, X_train)
    elif selected_page == "Dashboard":
        if not st.session_state.logged_in:
            st.warning("Please login first.")
        else:
            render_dashboard(model_metadata, preprocessing_data, label_encoder, feature_names)
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
