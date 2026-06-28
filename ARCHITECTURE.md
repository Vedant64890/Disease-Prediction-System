# Project Architecture

This document explains the architecture of the Disease Prediction System so the project structure is clear when the repository is uploaded to GitHub.

## Current Production Entry Point

The main Streamlit dashboard is:

```bash
streamlit run s06_dash.py
```

`s06_dash.py` is the active user-facing application. It contains the dashboard UI, authentication flow, symptom chatbot, prediction workflow, doctor booking, diagnostic lab booking, appointment history, theme switching, and external healthcare search.

## High-Level Architecture

```text
User Browser
    |
    v
Streamlit UI - s06_dash.py
    |
    +-- Authentication and session state
    |       +-- users table / legacy users.json migration
    |
    +-- Symptom input
    |       +-- manual symptom picker
    |       +-- guided chatbot symptom capture
    |
    +-- Prediction engine
    |       +-- best_model.pkl
    |       +-- preprocessing_data.pkl
    |       +-- label encoder and feature names
    |
    +-- Care plan layer
    |       +-- disease severity
    |       +-- specialist recommendation
    |       +-- blood report and lab-test guidance
    |       +-- warning signs and home-care guidance
    |
    +-- Booking and provider search
    |       +-- doctor and hospital search
    |       +-- diagnostic lab search
    |       +-- OpenStreetMap / Photon public listing lookup
    |       +-- appointment and lab appointment storage
    |
    +-- Optional live AI support
            +-- Gemini API key from .streamlit/secrets.toml
```

## Data and Model Pipeline

```text
csv_files/Training.csv
csv_files/Testing.csv
        |
        v
s01_prep.py
        |
        +-- preprocessing_data.pkl
        |
        v
s03_train.py or s03_train_advanced.py
        |
        +-- best_model.pkl / advanced_model.pkl
        |
        v
s04_eval.py
        |
        +-- model quality report
        |
        v
s06_dash.py
        |
        +-- loads trained artifacts for prediction
```

Important: dataset changes do not automatically appear in the dashboard. After editing or expanding datasets, rerun preprocessing and training before launching `s06_dash.py`.

Recommended refresh flow:

```bash
python s01_prep.py
python s03_train.py
python s04_eval.py
streamlit run s06_dash.py
```

## Application Pages

### Home
Shows the patient dashboard landing view, model/dataset summary, patient profile area, calendar, and project overview.

### Dashboard
Handles symptom capture, chatbot-assisted screening, manual symptom selection, prediction result, care guide, doctor/hospital search, doctor appointment booking, prediction charts, and home-care suggestions.

### Lab Tests
Shows lab-test guidance from the latest prediction only. If tests are suggested, the page lists prescribed/suggested tests, allows the user to add tests to cart, search nearby diagnostic labs by city/area, select a diagnostic lab, choose cash or online payment reference, and book a lab test appointment.

### Appointments
Shows active and historical doctor appointments and lab-test appointments. Users can cancel active doctor or lab bookings.

### Login / Register
Handles local account creation and login with hashed passwords.

### About / Contact
Shows project and support information.

## Storage Layer

The app uses SQLite for current runtime storage and keeps compatibility with older JSON files.

Main storage responsibilities:

- `users`: registered users and password hashes.
- `appointments`: doctor appointment bookings.
- `lab_appointments`: diagnostic lab appointment bookings, selected lab tests, payment method, payment status, appointment date/time, and selected disease context.
- Legacy JSON migration: `users.json`, `appointments.json`, and `lab_appointments.json` can be migrated into SQLite.

## Diagnostic Lab Search Flow

```text
User enters city or area in Lab Tests page
        |
        v
Location suggestions and geocoding
        |
        v
Diagnostic lab lookup
        |
        +-- fast Photon / OpenStreetMap diagnostic listing search
        +-- Overpass healthcare lab fallback
        |
        v
Nearby Diagnostic Labs list
        |
        v
User selects a lab
        |
        v
Lab name is added to the lab appointment form
```

The diagnostic search uses public map listings, so availability depends on published map data for the selected city or area.

## Chatbot and AI Layer

The chatbot has two modes:

- Local app logic for symptom extraction, follow-up questions, guardrails, and prediction readiness.
- Optional Gemini live response support when `GEMINI_API_KEY` is configured in `.streamlit/secrets.toml`.

The chatbot is designed to ask follow-up questions before prediction when symptoms are broad or overlapping.

## Security Notes

- Passwords are hashed with PBKDF2-SHA256.
- API keys should not be committed to GitHub.
- Use `.streamlit/secrets.toml` locally or deployment secrets in production.
- `.streamlit/secrets.example.toml` should show variable names only, not real keys.

## Deployment Notes

Before deployment:

```bash
python -m py_compile s06_dash.py
python s01_prep.py
python s03_train.py
python s04_eval.py
streamlit run s06_dash.py
```

For Streamlit Cloud or similar hosting, set:

- Entry point: `s06_dash.py`
- Requirements file: `requirements.txt`
- Secrets: configure `GEMINI_API_KEY` in the platform secret manager if live AI replies are needed.

## Medical Safety Boundary

This project is an educational screening-support system. It does not replace a qualified doctor, lab professional, or emergency service. Prediction results, medicines, lab tests, and care guidance should be confirmed by a qualified healthcare professional.
