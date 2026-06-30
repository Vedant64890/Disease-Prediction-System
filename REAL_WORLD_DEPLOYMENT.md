# Real-World Deployment Checklist

This app is now closer to a real application because users and appointments are stored in MySQL. Before using it with real patients, complete the items below.

## Current Foundation

- User accounts use password hashing.
- Users, doctor appointments, and lab appointments are persisted in MySQL.
- MySQL credentials are required through `MYSQL_HOST`, `MYSQL_DATABASE`, `MYSQL_USER`, and `MYSQL_PASSWORD`.
- Appointment slot checks and cancellations now update database rows directly.
- The app still presents predictions as screening support, not diagnosis.

## Before Real Patient Use

- Get medical review for disease guidance, medicines, warning signs, and lab-test advice.
- Add explicit patient consent before saving symptoms or appointment details.
- Publish a privacy policy and data deletion process.
- Use HTTPS in production.
- Store secrets only in environment variables or Streamlit secrets.
- Back up the database regularly.
- Add admin access for verifying doctors, hospitals, and appointment status.
- Add monitoring for crashes, failed API calls, and model errors.
- Back up and monitor the MySQL database before multiple staff members or many users rely on it.

## Deployment Path

1. Create the MySQL database and user.
2. Deploy the Streamlit app with MySQL and API secrets configured outside the codebase.
3. Use MySQL for users, doctor appointments, and lab appointments in production.
4. Add role-based admin views for doctors, hospitals, and booking operations.
5. Run clinical validation before marketing it as a healthcare product.

## Streamlit Cloud Deployment Package

Use `s06_dash.py` as the app entrypoint.

Keep these files in the repository for deployment:

- `s06_dash.py`
- `requirements.txt`
- `best_model.pkl`
- `preprocessing_data.pkl`
- `csv_files/Training.csv`
- `csv_files/Testing.csv`
- `csv_files/home_remedies_training.csv`
- `assets/chatbot_launcher.png`
- `.streamlit/config.toml`
- `.streamlit/secrets.example.toml`

Do not commit `.streamlit/secrets.toml`, `.env`, or real API keys. The app requires MySQL credentials at runtime.

## Before Every Dataset Deployment

After changing `csv_files/Training.csv` or `csv_files/Testing.csv`, regenerate and verify the artifacts before deploying:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python s01_prep.py
python s03_train.py
python s04_eval.py
python -m py_compile s06_dash.py
```

Then test locally:

```powershell
streamlit run s06_dash.py
```

## Streamlit Cloud Settings

In Streamlit Community Cloud:

1. Select the GitHub repository and branch.
2. Set the main file path to `s06_dash.py`.
3. In Advanced settings, choose a Python version compatible with the project, such as Python 3.11 or 3.12.
4. Paste the secret values from `.streamlit/secrets.example.toml`, replacing the placeholders with the real Gemini key and any optional map fallback keys.
5. Deploy and watch the build logs for dependency or missing-file errors.

## Environment Variables

Copy `.env.example` as a reference. For Streamlit, keep real secrets in `.streamlit/secrets.toml` or your hosting provider secret manager.

```text
GEMINI_API_KEY=replace-with-your-gemini-key
GEMINI_MODEL=gemini-2.5-flash
HERE_API_KEY=optional-here-api-key
MAPBOX_ACCESS_TOKEN=optional-mapbox-access-token
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=disease_prediction
MYSQL_USER=root
MYSQL_PASSWORD=replace-with-your-mysql-password
```
