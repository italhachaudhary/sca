import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    """
    This function handles the end-to-end process of training the disease
    prediction model by incorporating symptom severity, and saving the
    trained model for future use.
    """
    print("--- Starting Model Training ---")

    # 1. Load Datasets
    # ------------------
    try:
        training_df = pd.read_csv('dataset.csv')
        severity_df = pd.read_csv('Symptom-severity.csv')
        description_df = pd.read_csv('symptom_Description.csv')
        precaution_df = pd.read_csv('symptom_precaution.csv')
        print("Successfully loaded all CSV files.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all required CSV files are in the same directory.")
        return

    # 2. Data Preprocessing & Feature Engineering
    # -------------------------------------------
    print("Starting data preprocessing...")

    # Create a dictionary mapping symptoms to their severity weight
    severity_df['Symptom'] = severity_df['Symptom'].str.strip()
    severity_dict = severity_df.set_index('Symptom')['weight'].to_dict()

    # Clean column names in the training data
    training_df.columns = training_df.columns.str.strip()
    symptom_cols = [col for col in training_df.columns if col != 'Disease']

    # Get a list of all unique symptoms from the training data
    all_symptoms = sorted(list(set(symptom.strip() for col in symptom_cols for symptom in training_df[col].dropna().unique())))
    print(f"Identified {len(all_symptoms)} unique symptoms.")

    # Create a new DataFrame with symptoms as columns, initialized to zero
    encoded_df = pd.DataFrame(0, index=training_df.index, columns=all_symptoms)

    # Populate the DataFrame with severity scores
    for index, row in training_df.iterrows():
        for col in symptom_cols:
            symptom = row[col]
            if pd.notna(symptom):
                clean_symptom = symptom.strip()
                if clean_symptom in all_symptoms:
                    # Use severity score instead of just '1'
                    encoded_df.loc[index, clean_symptom] = severity_dict.get(clean_symptom, 0)
    
    encoded_df['Disease'] = training_df['Disease']
    
    # 3. Prepare Data for Modeling
    # ------------------------------
    X = encoded_df.drop('Disease', axis=1)
    y = encoded_df['Disease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # 4. Train the Model (Using a more robust RandomForestClassifier)
    # --------------------
    print("Training the Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 5. Evaluate the Model
    # -----------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- Model Evaluation ---")
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    print("------------------------")

    # 6. Save the Model and Helper Files
    # ----------------------------------
    print("Saving the trained model and helper files...")
    joblib.dump(model, 'disease_prediction_model.joblib')
    joblib.dump(X.columns.tolist(), 'symptom_columns.joblib')
    joblib.dump(severity_dict, 'symptom_severity_dict.joblib') # Save severity mapping
    
    description_df.to_pickle('disease_description.pkl')
    precaution_df.to_pickle('disease_precaution.pkl')
    
    print("--- Model and helper files saved successfully! ---")


def predict_disease(symptoms):
    """
    Predicts the disease based on a list of input symptoms using severity scores.
    """
    try:
        model = joblib.load('disease_prediction_model.joblib')
        symptom_columns = joblib.load('symptom_columns.joblib')
        severity_dict = joblib.load('symptom_severity_dict.joblib')
        description_df = pd.read_pickle('disease_description.pkl')
        precaution_df = pd.read_pickle('disease_precaution.pkl')
    except FileNotFoundError:
        return {"error": "Model files not found. Please run the training script first."}
        
    input_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)
    
    for symptom in symptoms:
        clean_symptom = symptom.strip()
        if clean_symptom in input_vector.columns:
            # Use severity score for the input vector
            input_vector.loc[0, clean_symptom] = severity_dict.get(clean_symptom, 0)
            
    predicted_disease = model.predict(input_vector)[0]
    
    description = description_df[description_df['Disease'] == predicted_disease]['Description'].values
    description_text = description[0] if len(description) > 0 else "No description available."

    precautions_row = precaution_df[precaution_df['Disease'] == predicted_disease]
    if not precautions_row.empty:
        precautions = [precautions_row[f'Precaution_{i}'].values[0] for i in range(1, 5)]
        precautions_list = [p for p in precautions if pd.notna(p)]
    else:
        precautions_list = ["No precautions available."]
        
    return {
        "predicted_disease": predicted_disease,
        "description": description_text,
        "precautions": precautions_list
    }

if __name__ == "__main__":
    # Step 1: Train the model
    train_and_save_model()
    
    # Step 2: Test the prediction function
    print("\n--- Testing the Prediction Function ---")
    example_symptoms = ['vomiting', 'loss_of_appetite', 'abdominal_pain']
    prediction_result = predict_disease(example_symptoms)
    
    if "error" in prediction_result:
        print(prediction_result["error"])
    else:
        print(f"Input Symptoms: {example_symptoms}")
        print(f"Predicted Disease: {prediction_result['predicted_disease']}")
        print(f"Description: {prediction_result['description']}")
        print(f"Precautions: {', '.join(prediction_result['precautions'])}")
