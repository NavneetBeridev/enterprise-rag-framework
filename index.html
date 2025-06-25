"""
main - Data analysis and machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

class DataAnalyzer:
    """Perform data analysis and modeling."""

    def __init__(self, data_path: str = None):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()

        if data_path:
            self.load_data(data_path)

    def load_data(self, path: str) -> None:
        """Load dataset from file."""
        self.data = pd.read_csv(path)
        print(f"Loaded data with shape: {self.data.shape}")

    def explore_data(self) -> None:
        """Perform exploratory data analysis."""
        if self.data is None:
            raise ValueError("No data loaded")

        print("Dataset Info:")
        print(self.data.info())
        print("\nSummary Statistics:")
        print(self.data.describe())

    def train_model(self, target_column: str) -> None:
        """Train machine learning model."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.3f}")

def main():
    """Demo the data analyzer."""
    analyzer = DataAnalyzer()
    print("Data analyzer initialized.")

if __name__ == "__main__":
    main()
