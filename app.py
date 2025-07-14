import os
import sqlite3
import json
import requests
import pandas as pd
import joblib
from flask import Flask, render_template, session, redirect, url_for, request, flash
from authlib.integrations.flask_client import OAuth

# --- APP CONFIGURATION ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- GEMINI API CONFIGURATION ---
# It's recommended to use environment variables for API keys in production
GEMINI_API_KEY = "" #Add your Gemini API key here

# --- OAUTH (GOOGLE LOGIN) CONFIGURATION ---
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='', # Add your Google Client ID here
    client_secret='', # Add your Google Client Secret here
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- LOAD MODEL AND HELPER FILES ---
try:
    # Load the list of symptom columns for the prediction form
    symptom_columns = joblib.load('symptom_columns.joblib')
    # Clean up symptom names for better display
    symptom_list_display = [s.replace('_', ' ').title() for s in symptom_columns]
    # Create a mapping from display name back to original column name
    symptom_map_reverse = {display: original for display, original in zip(symptom_list_display, symptom_columns)}

except FileNotFoundError:
    symptom_list_display = []
    symptom_map_reverse = {}
    print("Warning: Model helper files not found. The 'Search Symptoms' feature will not work until train_data.py is run.")


# --- DATABASE (SQLITE) SETUP ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect('symptom_checker.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the history table if it doesn't exist."""
    with app.app_context():
        db = get_db_connection()
        # The schema is now defined directly here for simplicity
        schema = """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            smoke BOOLEAN,
            cholesterol BOOLEAN,
            pressure BOOLEAN,
            diabetes BOOLEAN,
            symptoms TEXT NOT NULL,
            report TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        db.executescript(schema)
        db.commit()
        db.close()

# Initialize DB on startup
init_db()


# --- MODEL PREDICTION FUNCTION ---
def predict_disease_from_model(symptoms):
    """
    Predicts disease using the pre-trained RandomForest model.
    """
    try:
        model = joblib.load('disease_prediction_model.joblib')
        symptom_cols = joblib.load('symptom_columns.joblib')
        severity_dict = joblib.load('symptom_severity_dict.joblib')
        description_df = pd.read_pickle('disease_description.pkl')
        precaution_df = pd.read_pickle('disease_precaution.pkl')
    except FileNotFoundError:
        return {"error": "Model files not found. Please run the training script."}

    input_vector = pd.DataFrame(0, index=[0], columns=symptom_cols)
    for symptom in symptoms:
        clean_symptom = symptom.strip()
        if clean_symptom in input_vector.columns:
            input_vector.loc[0, clean_symptom] = severity_dict.get(clean_symptom, 0)
            
    predicted_disease = model.predict(input_vector)[0]
    
    desc_series = description_df[description_df['Disease'] == predicted_disease]['Description']
    description = desc_series.values[0] if not desc_series.empty else "No description available."

    precautions_row = precaution_df[precaution_df['Disease'] == predicted_disease]
    precautions_list = []
    if not precautions_row.empty:
        precautions = [precautions_row[f'Precaution_{i}'].values[0] for i in range(1, 5)]
        precautions_list = [p for p in precautions if pd.notna(p)]
    else:
        precautions_list = ["No precautions found."]
        
    # Format the output to be compatible with the report.html template
    report = {
        "diagnoses": [{
            "condition": predicted_disease,
            "likelihood": "High", # The model gives its best prediction
            "suggestions": f"{description} Recommended Precautions: {', '.join(precautions_list)}"
        }]
    }
    return report


# --- APP ROUTES ---

@app.route('/')
def index():
    """Main page: shows health info form if logged in, otherwise login page."""
    if 'user' in session:
        return render_template('health_info.html')
    return redirect(url_for('login_page'))

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/guest-login')
def guest_login():
    session['user'] = {'name': 'Guest User', 'picture': 'https://placehold.co/40x40/64748B/FFFFFF?text=G', 'is_guest': True}
    return redirect(url_for('index'))

@app.route('/google-login')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/google/callback')
def google_callback():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    user_info['is_guest'] = False
    session['user'] = user_info
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))

@app.route('/symptoms', methods=['POST'])
def symptoms_page():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    session['health_info'] = {
        'age': request.form.get('age'), 'gender': request.form.get('gender'),
        'smoke': 'smoke' in request.form, 'cholesterol': 'cholesterol' in request.form,
        'pressure': 'pressure' in request.form, 'diabetes': 'diabetes' in request.form,
    }
    # Pass the list of symptoms to the template
    return render_template('symptoms.html', symptoms=symptom_list_display)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyzes symptoms from the TEXTBOX using the Gemini API."""
    if 'user' not in session or 'health_info' not in session:
        return redirect(url_for('login_page'))

    symptoms = request.form.get('symptoms')
    if not symptoms:
        flash('Please describe your symptoms.', 'error')
        return render_template('symptoms.html', symptoms=symptom_list_display)

    full_info = {**session['health_info'], 'symptoms': symptoms}

    # --- Call Gemini API ---
    prompt = f"""
        You are a helpful medical AI assistant. Based on the following user data, provide a list of possible medical conditions.
        User Data:
        - Age: {full_info['age']}
        - Gender: {full_info['gender']}
        - Smoker: {'Yes' if full_info['smoke'] else 'No'}
        - High Cholesterol: {'Yes' if full_info['cholesterol'] else 'No'}
        - High/Low Blood Pressure: {'Yes' if full_info['pressure'] else 'No'}
        - Diabetic: {'Yes' if full_info['diabetes'] else 'No'}
        - Reported Symptoms: "{full_info['symptoms']}"

        Please respond with only a JSON object. The object must have a single key "diagnoses".
        The value of "diagnoses" must be an array of objects.
        Each object in the array must have three string keys: "condition", "likelihood" (rated as "High", "Medium", or "Low"), and "suggestions".
        Order the array from the most likely condition to the least likely.
        Do not include any text or markdown formatting outside of the JSON object.
    """

    try:
        api_response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}',
            json={'contents': [{'parts': [{'text': prompt}]}]}
        )
        api_response.raise_for_status() # Raise an exception for bad status codes
        
        result_text = api_response.json()['candidates'][0]['content']['parts'][0]['text']
        cleaned_text = result_text.replace("```json", "").replace("```", "").strip()
        report_data = json.loads(cleaned_text)
        
        # --- Save to Database (ONLY if not a guest) ---
        if not session['user'].get('is_guest'):
            db = get_db_connection()
            db.execute(
                'INSERT INTO history (user_id, age, gender, smoke, cholesterol, pressure, diabetes, symptoms, report) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    session['user']['sub'], full_info['age'], full_info['gender'],
                    full_info['smoke'], full_info['cholesterol'], full_info['pressure'],
                    full_info['diabetes'], symptoms, json.dumps(report_data)
                )
            )
            db.commit()
            db.close()
        
        session.pop('health_info', None)
        return render_template('report.html', report=report_data)

    except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"An error occurred: {e}")
        flash('Sorry, the analysis could not be completed at this time. Please try again later.', 'error')
        return render_template('symptoms.html', symptoms=symptom_list_display)


@app.route('/analyze_model', methods=['POST'])
def analyze_model():
    """Analyzes symptoms from the CHECKBOXES using the trained model."""
    if 'user' not in session or 'health_info' not in session:
        return redirect(url_for('login_page'))

    # Get selected symptoms (which are the display names)
    selected_symptoms_display = request.form.getlist('symptoms')
    if not selected_symptoms_display:
        flash('Please select at least one symptom.', 'error')
        return render_template('symptoms.html', symptoms=symptom_list_display)

    # Convert display names back to original names for the model
    symptoms_for_model = [symptom_map_reverse[s] for s in selected_symptoms_display]

    # Get the prediction from our model
    report_data = predict_disease_from_model(symptoms_for_model)

    if "error" in report_data:
        flash(report_data["error"], 'error')
        return render_template('symptoms.html', symptoms=symptom_list_display)

    # Save to DB if not a guest
    if not session['user'].get('is_guest'):
        full_info = {**session['health_info']}
        db = get_db_connection()
        db.execute(
            'INSERT INTO history (user_id, age, gender, smoke, cholesterol, pressure, diabetes, symptoms, report) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                session['user']['sub'], full_info['age'], full_info['gender'],
                full_info['smoke'], full_info['cholesterol'], full_info['pressure'],
                full_info['diabetes'], ", ".join(selected_symptoms_display), json.dumps(report_data)
            )
        )
        db.commit()
        db.close()
    
    session.pop('health_info', None)
    return render_template('report.html', report=report_data)

@app.route('/history')
def history():
    """Displays the user's past checkup history."""
    if 'user' not in session or session['user'].get('is_guest'):
        flash('Please log in with Google to view your history.', 'info')
        return redirect(url_for('index'))

    db = get_db_connection()
    history_records = db.execute(
        'SELECT * FROM history WHERE user_id = ? ORDER BY timestamp DESC', (session['user']['sub'],)
    ).fetchall()
    db.close()
    
    processed_records = [dict(record, report=json.loads(record['report'])) for record in history_records]
    return render_template('history.html', history=processed_records)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
