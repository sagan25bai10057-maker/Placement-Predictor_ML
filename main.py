import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
print("Welcome! \n")
print("Hello! I am SARTHAK JINDAL (25BCE10189) presenting you my ML project....\n")


DATA_FILE = 'placement_data.csv'
MODEL_FILE = 'placement_model.pkl'
FEATURE_NAMES = ['CGPA', 'Internships', 'Projects', 'Technical_Skills_Score', 'Soft_Skills_Score']

def train_model():
    print("\n--- Initializing Machine Learning Model ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please ensure your dataset is in the same folder.")
        return None

    
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURE_NAMES]
    y = df['Placed']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Status: Model trained successfully.")
    print(f"Model Accuracy: {acc*100:.2f}%")

    
    joblib.dump(model, MODEL_FILE)
    return model

def predict_placement(model):
    print("\n" + "="*40)
    print("      VIT PLACEMENT PREDICTOR 2026      ")
    print("="*40)
    
    try:
        
        cgpa = float(input("Enter Student CGPA (e.g., 8.5): "))
        interns = int(input("Number of Internships completed: "))
        projects = int(input("Number of Projects completed: "))
        tech_score = int(input("Technical Skill Score (1-100): "))
        soft_score = int(input("Soft Skill Score (1-100): "))

        
        user_input_df = pd.DataFrame([[cgpa, interns, projects, tech_score, soft_score]], 
                                     columns=FEATURE_NAMES)
        
       
        prediction = model.predict(user_input_df)
        probability = model.predict_proba(user_input_df)[0][1] 

        print("\n--- PREDICTION ANALYSIS ---")
        if prediction[0] == 1:
            print(f"STATUS: PLACED!")
            print(f"PROBABILITY: {probability*100:.1f}%")
            print("ADVICE: Excellent profile. You are eligible for Super Dream offers.")
        else:
            print(f"STATUS: NOT PLACED YET")
            print(f"PROBABILITY: {(1-probability)*100:.1f}% (Chance of rejection)")
            print("ADVICE: Consider improving CGPA or adding one more internship.")
            
    except ValueError:
        print("\n[!] ERROR: Please enter valid numeric values.")

if __name__ == "__main__":
    
    if os.path.exists(MODEL_FILE):
        placement_model = joblib.load(MODEL_FILE)
    else:
        
        placement_model = train_model()

    if placement_model:
        
        while True:
            predict_placement(placement_model)
            print("-" * 40)
            cont = input("Check another student? (y/n): ").lower()
            if cont != 'y':
                print("\nThank you for using the Predictor. Good luck !")
                break