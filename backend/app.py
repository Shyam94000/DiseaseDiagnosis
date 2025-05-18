import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import json
from difflib import SequenceMatcher

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
# --- Global Variables & Configuration ---
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.joblib')
ALL_SYMPTOMS_PATH = os.path.join(MODEL_DIR, 'all_symptoms.json')
X_COLUMNS_PATH = os.path.join(MODEL_DIR, 'x_columns.json')
DATASET_PATH = 'dataset.csv' # Assuming dataset.csv is in the same directory as app.py

# Ensure MODEL_DIR exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load or Train Model ---
best_model = None
encoder = None
all_symptoms_list = []
X_columns = []
model_training_details = {} # To store accuracies

def create_demo_dataset():
    print("Demo mode: Creating sample data since actual dataset couldn't be loaded or is being initialized.")
    diseases = ['Common Cold', 'Flu', 'Malaria', 'Dengue', 'Typhoid']
    symptoms = ['headache', 'fever', 'cough', 'fatigue', 'joint_pain', 'vomiting',
               'chills', 'sweating', 'muscle_pain', 'dizziness', 'runny_nose',
               'sore_throat', 'body_ache', 'rash', 'nausea', 'diarrhea', 'abdominal_pain'] # Added more common symptoms

    data = []
    for _ in range(500):
        disease = np.random.choice(diseases)
        num_symptoms_to_pick = np.random.randint(2, 7) # Each record has 2 to 6 symptoms
        # Ensure we don't pick more symptoms than available
        sample_symptoms = np.random.choice(symptoms, size=min(num_symptoms_to_pick, len(symptoms)), replace=False)

        row = [disease] + list(sample_symptoms) + [''] * (17 - len(sample_symptoms)) # Assuming max 17 symptom columns
        data.append(row)

    columns = ['Disease'] + [f'Symptom_{i+1}' for i in range(17)]
    df = pd.DataFrame(data, columns=columns)
    # Ensure all symptom columns exist even if not filled by this random generation
    for i in range(1,18):
        col_name = f'Symptom_{i}'
        if col_name not in df.columns:
            df[col_name] = '' # or np.nan
    return df

def load_and_preprocess_data():
    global all_symptoms_list, encoder, X_columns
    try:
        df = pd.read_csv(DATASET_PATH)
        if df.empty or len(df.columns) <= 1: # Basic check for valid dataset
            print(f"Dataset at {DATASET_PATH} is empty or not formatted correctly. Using demo data.")
            df = create_demo_dataset()
    except FileNotFoundError:
        print(f"Dataset file not found at {DATASET_PATH}. Using demo data.")
        df = create_demo_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}. Using demo data.")
        df = create_demo_dataset()

    # Fill NaN values with empty strings to handle them consistently
    df = df.fillna('')

    # Data preprocessing
    current_all_symptoms = []
    for symptom_col in df.columns[1:]:
        unique_symptoms = df[symptom_col].dropna().unique()
        for symptom in unique_symptoms:
            clean_symptom = str(symptom).strip() # Ensure it's a string before stripping
            if clean_symptom and clean_symptom not in current_all_symptoms:
                current_all_symptoms.append(clean_symptom)
    
    all_symptoms_list = sorted(list(set(current_all_symptoms))) # Ensure unique and sorted

    symptoms_df = pd.DataFrame(0, index=range(len(df)), columns=all_symptoms_list)

    for i, row in df.iterrows():
        for symptom_col in df.columns[1:]:
            symptom_value = str(row[symptom_col]).strip() # Ensure it's a string
            if pd.notna(row[symptom_col]) and symptom_value and symptom_value in all_symptoms_list:
                symptoms_df.at[i, symptom_value] = 1
    
    X = symptoms_df
    X_columns = list(X.columns) # Store column order for consistent input creation later
    y_raw = df['Disease']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    return X, y, df # Return original df for potential further use

def train_and_save_models():
    global best_model, encoder, all_symptoms_list, X_columns, model_training_details
    
    X, y, _ = load_and_preprocess_data() # encoder, all_symptoms_list, X_columns are set globally here

    if X.empty or len(y) == 0:
        print("Cannot train models with empty data.")
        # Try to load existing models if any, or fail gracefully
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and \
           os.path.exists(ALL_SYMPTOMS_PATH) and os.path.exists(X_COLUMNS_PATH):
            load_saved_assets()
            print("Loaded previously saved model assets as training data was empty.")
        else:
            print("CRITICAL: No data to train and no saved models found.")
        return


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42), # Ensure probability=True for predict_proba
        'KNN': KNeighborsClassifier(n_neighbors=min(5, len(X_train) -1 if len(X_train) > 1 else 1)) # Ensure n_neighbors is valid
    }
    
    results = {}
    best_accuracy = 0.0

    for name, model_instance in models.items():
        if len(X_train) == 0: # Cannot train if X_train is empty
            print(f"Skipping training for {name} due to empty training data.")
            results[name] = {'accuracy': 0, 'cv_mean': 0, 'model': None}
            continue

        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        cv_scores = [0] # Default if CV cannot be run
        if len(X) >= 5 and len(np.unique(y)) > 1 : # CV needs enough samples and classes
             try:
                cv_scores = cross_val_score(model_instance, X, y, cv=min(5, len(np.unique(y))), scoring='accuracy')
             except ValueError as e:
                print(f"CV score error for {name}: {e}. Setting to 0.")


        results[name] = {
            'model': model_instance,
            'accuracy': accuracy,
            'cv_scores': cv_scores.tolist(), # Convert to list for JSON
            'cv_mean': cv_scores.mean()
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Cross-validation mean: {cv_scores.mean():.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_instance
    
    model_training_details = {name: {'accuracy': data['accuracy'], 'cv_mean': data['cv_mean']} for name, data in results.items()}

    if best_model:
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        with open(ALL_SYMPTOMS_PATH, 'w') as f:
            json.dump(all_symptoms_list, f)
        with open(X_COLUMNS_PATH, 'w') as f:
            json.dump(X_columns, f)
        print(f"Best model ({type(best_model).__name__}) saved with accuracy: {best_accuracy:.4f}")
    else:
        print("No best model was selected or trained. Models may not be saved.")


def load_saved_assets():
    global best_model, encoder, all_symptoms_list, X_columns
    try:
        best_model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        with open(ALL_SYMPTOMS_PATH, 'r') as f:
            all_symptoms_list = json.load(f)
        with open(X_COLUMNS_PATH, 'r') as f:
            X_columns = json.load(f)
        print("Model, encoder, symptoms list, and X_columns loaded successfully.")
        return True
    except FileNotFoundError:
        print("Saved model assets not found.")
        return False
    except Exception as e:
        print(f"Error loading saved assets: {e}")
        return False

# Initialize: Try to load, if not, train and save.
if not load_saved_assets():
    print("Training models for the first time or because assets were missing...")
    train_and_save_models()
else:
    # If loaded successfully, we might still want to have model_training_details
    # This part is tricky as details are from last training.
    # For simplicity, we can re-run a light version or save training details too.
    # Or, just acknowledge they are not available unless a retrain happens.
    print("Model assets loaded. Training details would require a re-train or saved metrics.")


# --- API Endpoints ---
@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    print("--- DEBUG: ENTERED /api/symptoms route ---", flush=True) # DEBUG
    global all_symptoms_list

    print(f"--- DEBUG: Initial all_symptoms_list is {'populated' if all_symptoms_list else 'None or Empty'}. Length: {len(all_symptoms_list) if all_symptoms_list else 'N/A'} ---", flush=True)

    if not all_symptoms_list:
        print("--- DEBUG: all_symptoms_list is empty or None, calling load_and_preprocess_data() ---", flush=True)
        try:
            # This function should modify the global all_symptoms_list
            load_and_preprocess_data() 
            print(f"--- DEBUG: After load_and_preprocess_data, all_symptoms_list is {'populated' if all_symptoms_list else 'None or Empty'}. Length: {len(all_symptoms_list) if all_symptoms_list else 'N/A'} ---", flush=True)
        except Exception as e:
            print(f"--- DEBUG: ERROR in load_and_preprocess_data: {e} ---", flush=True)
            return jsonify(error=f"Failed to load symptoms data: {str(e)}"), 500 # Should be 500

    if all_symptoms_list is None: # Explicit check
        print("--- DEBUG: CRITICAL - all_symptoms_list is None before jsonify. Returning 500. ---", flush=True)
        return jsonify(error="Symptom list is None, cannot process."), 500

    # Ensure it's a list before sorting
    symptoms_to_return = []
    if isinstance(all_symptoms_list, list):
        symptoms_to_return = sorted(all_symptoms_list)
    else:
        print(f"--- DEBUG: WARNING - all_symptoms_list is not a list ({type(all_symptoms_list)}), attempting conversion. ---", flush=True)
        try:
            symptoms_to_return = sorted(list(all_symptoms_list)) # Attempt to convert
        except TypeError as te:
            print(f"--- DEBUG: ERROR converting/sorting non-list all_symptoms_list: {te} ---", flush=True)
            return jsonify(error=f"Symptom data in wrong format: {str(te)}"), 500


    print(f"--- DEBUG: About to jsonify {len(symptoms_to_return)} symptoms. First 5: {symptoms_to_return[:5]} ---", flush=True)
    try:
        response = jsonify(symptoms_to_return)
        print("--- DEBUG: jsonify successful. Returning response. ---", flush=True)
        return response
    except Exception as e:
        print(f"--- DEBUG: ERROR during jsonify: {e} ---", flush=True)
        return jsonify(error=f"Error serializing symptom list: {str(e)}"), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    global best_model, encoder, X_columns # Ensure we're using the globally loaded/trained assets
    
    if best_model is None or encoder is None or not X_columns:
        return jsonify({'error': 'Model not loaded or trained yet. Please try again later or contact admin.'}), 500

    data = request.get_json()
    symptom_list = data.get('symptoms', [])

    if not symptom_list:
        return jsonify({'error': 'No symptoms provided'}), 400

    # Create input DataFrame with correct columns based on X_columns from training
    input_df = pd.DataFrame(0, index=[0], columns=X_columns)
    
    valid_symptoms_provided = False
    for symptom in symptom_list:
        if symptom in X_columns: # Check against trained feature names
            input_df[symptom] = 1
            valid_symptoms_provided = True
        else:
            print(f"Warning: Symptom '{symptom}' provided by user but not in known symptoms list during training.")
    
    if not valid_symptoms_provided:
        return jsonify({'error': 'None of the provided symptoms are recognized by the model.'}), 400

    try:
        prediction_encoded = best_model.predict(input_df)[0]
        disease = encoder.inverse_transform([prediction_encoded])[0]
        
        probabilities = []
        if hasattr(best_model, 'predict_proba'):
            probs = best_model.predict_proba(input_df)[0]
            # Get top 3 or fewer if less classes
            num_classes = len(encoder.classes_)
            top_n = min(3, num_classes)
            top_indices = np.argsort(probs)[::-1][:top_n]
            
            top_diseases = encoder.inverse_transform(top_indices)
            top_probs = probs[top_indices]
            
            probabilities = [{'disease': d, 'probability': float(p)} for d, p in zip(top_diseases, top_probs)]
        else: # For models like basic SVM without probability=True
            probabilities = [{'disease': disease, 'probability': 1.0}] # Placeholder

        return jsonify({
            'predicted_disease': disease,
            'probabilities': probabilities
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/api/find_similar_symptoms', methods=['POST'])
def find_similar():
    data = request.get_json()
    query = data.get('query', '').strip()
    threshold = data.get('threshold', 0.6) # Default threshold

    if not query:
        return jsonify([])

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    similar_symptoms_found = []
    for symptom in all_symptoms_list: # Use the global all_symptoms_list
        sim_score = similarity(query, symptom)
        if sim_score >= threshold:
            similar_symptoms_found.append({'symptom': symptom, 'score': round(sim_score, 2)})
    
    similar_symptoms_found.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(similar_symptoms_found)

@app.route('/api/retrain', methods=['POST'])
def force_retrain():
    print("Retraining models as per API request...")
    try:
        train_and_save_models()
        # After retraining, model_training_details global var should be updated.
        return jsonify({
            "message": "Models retrained successfully.",
            "training_summary": model_training_details
            }), 200
    except Exception as e:
        print(f"Error during retraining: {e}")
        return jsonify({"error": f"Failed to retrain models: {str(e)}"}), 500

@app.route('/api/model_performance', methods=['GET'])
def get_model_performance():
    # This endpoint will return the performance metrics stored during the last training.
    # If model_training_details is empty, it implies models were loaded without fresh training in this session.
    if not model_training_details:
        # You could attempt to load saved metrics if you decide to save them as a JSON file too.
        # For now, let's indicate they are available after training/retraining.
        return jsonify({
            "message": "Model performance details are available after training. Consider using the /api/retrain endpoint if needed.",
            "details": {}
        }), 200 # Or 404 if you prefer
    
    return jsonify(model_training_details)


@app.route('/api/feature_importance', methods=['GET'])
def get_feature_importance():
    global best_model, X_columns
    if best_model and hasattr(best_model, 'feature_importances_') and X_columns:
        if isinstance(best_model, RandomForestClassifier): # Check if it's RF or similar
            importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X_columns, # Use stored X_columns
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            top_n = min(20, len(feature_importance_df))
            return jsonify(feature_importance_df.head(top_n).to_dict(orient='records'))
        else:
            return jsonify({'message': f'Feature importance is not available for the current best model type ({type(best_model).__name__}). Only available for models like Random Forest.'}), 400
    return jsonify({'error': 'Model not loaded or does not support feature importance.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000) # Port 5000 for backend