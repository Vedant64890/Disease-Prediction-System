# Disease Prediction System

Patient-facing disease screening dashboard with symptom prediction, chatbot-guided symptom capture, care guidance, doctor booking, diagnostic lab search, lab-test cart/payment reference, and appointment tracking.

## Repository Description

Disease Prediction System is a Streamlit-based healthcare screening project that helps patients describe symptoms, run a machine-learning disease prediction, review care guidance, search doctors/hospitals, find nearby diagnostic labs, book doctor or lab-test appointments, and track appointment history from one dashboard.

### Short GitHub Description

```text
AI-powered disease screening dashboard with symptom chatbot, ML prediction, doctor booking, diagnostic lab search, lab-test booking, and appointment tracking.
```

### Project Highlights

- Patient-friendly symptom chatbot and manual symptom picker.
- Machine-learning prediction using trained model artifacts.
- Care-plan guidance with severity, specialist, warning signs, home-care notes, and lab-test guidance.
- Doctor and hospital search with OpenStreetMap/Photon/Overpass public listings, plus optional HERE fallback.
- Separate Lab Tests section with diagnostic lab search, test cart, payment reference, and lab appointment booking.
- Appointment page for doctor and lab-test bookings with cancellation support.
- Light/dark theme UI and responsive Streamlit dashboard.
- GitHub-visible architecture documentation in [ARCHITECTURE.md](ARCHITECTURE.md).

## Main App

```bash
streamlit run s06_dash.py
```

## Project Architecture

The complete architecture is documented in a separate file:

**[ARCHITECTURE.md](ARCHITECTURE.md)**

It explains:

- Streamlit dashboard structure
- Data and model training pipeline
- Prediction and care-plan flow
- Chatbot flow
- Doctor, hospital, and diagnostic lab search
- Lab-test booking and payment-reference flow
- MySQL storage design
- Deployment notes

## Quick Start

```bash
cd "C:\Data science_Project"
pip install -r requirements.txt
python s01_prep.py
python s03_train.py
python s04_eval.py
streamlit run s06_dash.py
```

Then open:

```text
http://localhost:8501
```

## Current User Flow

```text
Register / Login
    |
    v
Dashboard
    |
    +-- guided chatbot symptom capture
    +-- manual symptom selection
    +-- disease screening prediction
    +-- care guidance
    +-- doctor and hospital search
    +-- doctor appointment booking
    |
    v
Lab Tests
    |
    +-- shows tests from the latest prediction
    +-- searches nearby diagnostic labs by city or area
    +-- lets the user select a lab
    +-- adds lab tests to cart
    +-- records cash or online payment reference
    +-- books lab-test appointment
    |
    v
Appointments
    |
    +-- doctor bookings
    +-- lab-test bookings
    +-- booking history
    +-- cancellation
```

## Data and Model Pipeline

```text
csv_files/Training.csv + csv_files/Testing.csv
        |
        v
s01_prep.py
        |
        v
preprocessing_data.pkl
        |
        v
s03_train.py
        |
        v
best_model.pkl
        |
        v
s04_eval.py
        |
        v
s06_dash.py
```

If dataset files are changed, rerun:

```bash
python s01_prep.py
python s03_train.py
python s04_eval.py
```

The dashboard loads trained pickle artifacts; it does not automatically retrain from CSV files while running.

## Important Files

| File | Purpose |
|---|---|
| `s06_dash.py` | Main Streamlit dashboard |
| `ARCHITECTURE.md` | Full project architecture |
| `s01_prep.py` | Dataset preprocessing |
| `s03_train.py` | Main model training |
| `s03_train_advanced.py` | Advanced/experimental training script |
| `s04_eval.py` | Model evaluation |
| `s05_predict.py` | Standalone prediction helper |
| `preprocessing_data.pkl` | Processed training artifacts |
| `best_model.pkl` | Trained model used by dashboard |
| `requirements.txt` | Python dependencies |
| `.streamlit/secrets.toml` | Local secrets file, not for GitHub |

## Features

- Secure local login/register flow with hashed passwords.
- MySQL storage for users, doctor appointments, and lab appointments.
- Light and dark theme support.
- Patient-facing symptom chatbot.
- Manual symptom picker.
- Disease prediction from trained ML artifacts.
- Care-plan guidance with severity, specialist, home-care notes, warning signs, and lab-test guidance.
- Doctor and hospital search using public map listings.
- Separate Lab Tests page with diagnostic lab search.
- Lab-test cart, estimated total, cash/online payment reference, and appointment booking.
- Doctor and lab appointment history with cancellation.
- Plotly-based prediction visualizations.

## Diagnostic Lab Search

The Lab Tests page can search diagnostic labs by city or area. It uses public Photon/OpenStreetMap listings first, then optional HERE fallback if `HERE_API_KEY` is configured.

The selected diagnostic lab is automatically filled into the lab booking form.

## Secrets and API Keys

Do not commit real API keys to GitHub.

For Gemini chatbot support and optional map fallbacks, put keys in:

```text
.streamlit/secrets.toml
```

Use deployment platform secrets for hosted deployment.

Required/optional secret names:

```text
GEMINI_API_KEY=replace-with-your-gemini-key
HERE_API_KEY=optional-here-api-key
MAPBOX_ACCESS_TOKEN=optional-mapbox-access-token
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=disease_prediction
MYSQL_USER=root
MYSQL_PASSWORD=replace-with-your-mysql-password
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=replace-with-your-email-app-password
SMTP_FROM_EMAIL=your-email@gmail.com
SMTP_FROM_NAME=Disease Prediction System
SMTP_USE_TLS=true
```

Map fallback behavior:

- No map key: uses OpenStreetMap Nominatim, Photon, and Overpass public listings.
- `HERE_API_KEY`: adds fallback doctor, hospital, and diagnostic lab place search.
- `MAPBOX_ACCESS_TOKEN`: adds fallback city/area geocoding when public geocoding fails.

Required MySQL app storage:

- Create a MySQL database, for example `disease_prediction`.
- Add `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, and `MYSQL_PASSWORD` to `.streamlit/secrets.toml` or deployment secrets.
- Register/login/password reset, account email addresses, doctor bookings, and lab bookings use MySQL only.
- The app will stop with a clear configuration error if MySQL credentials are missing or MySQL is unreachable.

Appointment email confirmations:

- New registrations require an email address.
- Doctor and lab appointments are saved first, then the app emails the booking details to the registered email address.
- Add the `SMTP_*` secrets above to enable delivery. Gmail requires an app password, not the normal account password.
- If SMTP is not configured or delivery fails, the appointment remains saved and the UI shows a warning.

## Deployment Checklist

Before uploading or deploying:

```bash
python -m py_compile s06_dash.py
python s01_prep.py
python s03_train.py
python s04_eval.py
streamlit run s06_dash.py
```

For Streamlit Cloud:

- Main file path: `s06_dash.py`
- Dependencies: `requirements.txt`
- Secrets: configure `GEMINI_API_KEY`; optionally add `HERE_API_KEY`, `MAPBOX_ACCESS_TOKEN`, and MySQL credentials

## Medical Safety Note

This project provides educational screening support only. It does not replace a doctor, emergency care, or certified lab guidance. Predictions, medicines, tests, and next steps should be confirmed by a qualified healthcare professional.
