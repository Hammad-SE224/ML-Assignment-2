import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load_dataset(file_path='loan_data.csv'):
    """
    Ye function CSV file ko load karta hai aur features (X) 
    aur labels (y) mein split karke train/test sets return karta hai.
    """
    
    # Agar file nahi milti toh testing ke liye dummy file bana lo
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Creating a dummy dataset...")
        create_dummy_data(file_path)

    # Dataset read karna
    df = pd.read_csv(file_path)

    # Features: Income, CreditScore, DebtRatio, LoanAmount
    X = df[['Income', 'CreditScore', 'DebtRatio', 'LoanAmount']]
    
    # Target: Status (1 = Approved, 0 = Rejected)
    y = df['Status']

    # Data ko Train aur Test mein split karna (80% Training, 20% Testing)
    # Stratify use kiya hai taake 2:1 ka imbalance dono sets mein barabar rahe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def create_dummy_data(file_path):
    """
    Ye function ek sample loan_data.csv file create karega 
    jis mein 2:1 ka imbalance hoga (Approved:Rejected).
    """
    np.random.seed(42)
    n_samples = 600
    
    data = {
        'Income': np.random.randint(2000, 15000, n_samples),
        'CreditScore': np.random.randint(300, 850, n_samples),
        'DebtRatio': np.random.uniform(0.1, 0.7, n_samples),
        'LoanAmount': np.random.randint(5000, 60000, n_samples),
        # Status: 1 (Approved) aur 0 (Rejected) - 2:1 Ratio
        'Status': np.random.choice([0, 1], n_samples, p=[0.33, 0.67]) 
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Successfully created: {file_path}")