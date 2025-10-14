import os
import joblib
from flask import Flask, request, render_template
import pandas as pd 

MODEL_FILENAME = 'gradient_boosting_model.joblib'
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

app = Flask(__name__, template_folder="templates")
model = None

LABELS = {
    "barely-true": "Barely True", 
    "false": "False",
    "half-true": "Half True",
    "mostly-true": "Mostly True",
    "pants-on-fire": "Pants on Fire",
    "true": "True"
}

def load_model():
    """Loads the saved scikit-learn pipeline using the robust absolute path."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}. Ensure the model is in the same directory as app.py.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        model = None
    return model

@app.route('/', methods=['GET', 'POST'])
def predict():
    global model
    
    if model is None:
        return render_template('error.html', title="Deployment Error", 
                               message=f"The prediction model ({MODEL_FILENAME}) could not be loaded. Please ensure it is saved correctly and is accessible."), 500

    predicted_label = None
    statement = ""

    if request.method == 'POST':
        statement = request.form.get('statement', '').strip()
        
        if statement:
            try:
                data = {
                    'Statement': [statement],
                    
                    'char_count': [len(statement)],
                    'word_count': [len(statement.split())],
                    'avg_word_length': [5.0],
                    'exclamation_count': [0],
                    'question_count': [0],
                    'true_ratio': [0.5],
                    'false_ratio': [0.5],
                    'uppercase_ratio': [0.05],
                    'complex_word_ratio': [0.1],
                    'sentence_count': [1],
                    'avg_sentence_length': [len(statement.split())],

                    'Speaker Job Title': ['unknown_speaker_job_title'],
                    'Party Affiliation': ['none'],
                    'State Info': ['unknown_state']
                }

                input_data = pd.DataFrame(data)
                
                prediction_index = model.predict(input_data)[0]
                predicted_label = LABELS.get(prediction_index, f"Unknown Index ({prediction_index})")
            except Exception as e:
                predicted_label = f"Prediction Error: {type(e).__name__}: {e}"
                print(f"Prediction attempt failed: {e}")
        else:
            predicted_label = "Please paste a statement to classify."

    return render_template('index.html', statement=statement, prediction=predicted_label)

@app.errorhandler(500)
def internal_error(error):
    print(error)
    return render_template('error.html', title="Server Error", message=error), 500

if __name__ == '__main__':
    load_model()
    if model:
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Application terminated due to model loading failure.")
