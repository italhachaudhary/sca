# Symptom Checker Application

This project is a Symptom Checker web application that predicts diseases based on user-input symptoms and provides health information, disease descriptions, and precautionary advice.

## Features
- User authentication (login)
- Symptom input and disease prediction
- Health information and disease descriptions
- Precautionary advice for predicted diseases
- User report and history tracking

## Project Structure
```
app.py                        # Main Flask application
train_data.py                 # Model training and data processing script
dataset.csv                   # Dataset for training
symptom_Description.csv       # Symptom descriptions
symptom_precaution.csv        # Symptom precautions
Symptom-severity.csv          # Symptom severity data
disease_description.pkl       # Pickled disease descriptions
disease_precaution.pkl        # Pickled disease precautions
disease_prediction_model.joblib # Trained prediction model
symptom_columns.joblib        # Pickled symptom columns
symptom_severity_dict.joblib  # Pickled symptom severity dictionary
symptom_checker.db            # SQLite database for user data

/templates/
  layout.html                 # Base HTML layout
  login.html                  # Login page
  symptoms.html               # Symptom input page
  report.html                 # User report page
  history.html                # User history page
  health_info.html            # Health information page
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   - Python 3.x
   - Flask
   - scikit-learn
   - joblib
   - pandas
   - (Other dependencies as required)
   
   Install with pip:
   ```bash
   pip install flask scikit-learn joblib pandas
   ```
3. **Run the application**
   ```bash
   python app.py
   ```
4. **Access the app**
   - Open your browser and go to `http://127.0.0.1:5000/`

## Usage
- Register or log in to your account
- Enter your symptoms
- View predicted disease, description, and precautions
- Check your report and history

## License
This project is for educational purposes.
