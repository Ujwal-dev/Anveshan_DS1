from pycaret.classification import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pycaret.classification import load_model, predict_model


df = pd.read_csv('C:\\Users\\ujwal.k.dilip.bhoot\\Desktop\\Ujwal\\files\\NqndMEyZakuimmFI.xlsx')
embeddings_df = pd.read_csv('C:\\Users\\ujwal.k.dilip.bhoot\\Desktop\\Ujwal\\files\\embeddings.csv')
encodings_df = pd.read_csv('C:\\Users\\ujwal.k.dilip.bhoot\\Desktop\\Ujwal\\files\\encoded_features_latest.csv')

df['salary_range_missing'] = df['salary_range'].isnull().astype(int)

columns_to_drop = ['job_id', 'title', 'location', 'department', 'salary_range', 'company_profile',
       'description', 'requirements', 'benefits', 'telecommuting',
       'has_company_logo', 'has_questions', 'employment_type',
       'required_experience', 'required_education', 'industry', 'function',
       ]

df = df.drop(columns=columns_to_drop, errors='ignore') 

combined_df = pd.concat([df, embeddings_df, encoded_df], axis=1)

clf = setup(data=combined_df, target='fraudulent', normalize=True, remove_multicollinearity=True, polynomial_features=True, fix_imbalance=True)

# Step 2: Compare models using F1 score
best_model = compare_models(sort='F1')

# Step 3: Tune the best model for better performance
tuned_model = tune_model(best_model, optimize='F1')

# Step 4: Optionally blend and stack models
blended = blend_models(estimator_list=[best_model, tuned_model], optimize='F1')
stacked = stack_models(estimator_list=[best_model, tuned_model], meta_model=best_model)

# Step 5: Evaluate the tuned model
evaluate_model(tuned_model)

save_model(tuned_model, 'final_fraud_detection_model')

print("✅ Model training and evaluation complete. Final model saved as 'final_fraud_detection_model.pkl'.")

# Load the saved model
model = load_model('final_fraud_detection_model')

# Load your test data
test_df = pd.read_csv('C:\\Users\\ujwal.k.dilip.bhoot\\Desktop\\Ujwal\\files\\0tkf3jUGLYjCEJGz.csv')
