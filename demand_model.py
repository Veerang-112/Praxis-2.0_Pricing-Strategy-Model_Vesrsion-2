import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

class DemandModel:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'sentiment_score', 'category_main']
        self.target = 'rating_count'

    def prepare_pipeline(self):
        """Creates the ML pipeline."""
        numeric_features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'sentiment_score']
        categorical_features = ['category_main']

        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    def train(self, df):
        """Trains the model."""
        print("Preparing data for training...")
        X = df[self.features]
        # Log transform target to handle skewness (demand is often log-normally distributed)
        y = np.log1p(df[self.target])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.pipeline is None:
            self.prepare_pipeline()

        print("Training Random Forest model...")
        self.pipeline.fit(X_train, y_train)
        
        score = self.pipeline.score(X_test, y_test)
        print(f"Model R^2 Score on Test Set: {score:.4f}")
        return score

    def predict(self, feature_df):
        """Predicts demand (converts back from log scale)."""
        if self.pipeline is None:
            raise Exception("Model not trained or loaded.")
        
        # Ensure input has correct columns
        for col in self.features:
            if col not in feature_df.columns:
                 # Add missing columns with default 0/empty if needed, or raise error.
                 # For simplicity, we assume input is well-formed or we fill na.
                 feature_df[col] = 0 if col != 'category_main' else 'Other'

        log_pred = self.pipeline.predict(feature_df[self.features])
        return np.expm1(log_pred) # Inverse of log1p

    def save(self, filepath):
        """Saves the trained pipeline."""
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Loads a trained pipeline."""
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
