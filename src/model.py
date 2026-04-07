import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class MaintenanceModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, data: pd.DataFrame):
        X = data.drop('failure', axis=1)
        y = data['failure']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        print("--- Model Evaluation ---")
        print(classification_report(y_test, predictions))
        return confusion_matrix(y_test, predictions)

    def save_model(self, path: str):
        joblib.dump(self.model, path)
