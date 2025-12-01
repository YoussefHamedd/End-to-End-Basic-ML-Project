"""
Unified Prediction Pipeline for Both Traditional ML and Transformer Models

This pipeline can handle:
1. Traditional ML models (sklearn, xgboost, catboost)
2. Transformer models (RoBERTa, BERT, etc.)

The pipeline automatically detects model type and uses the appropriate
prediction method.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from src.exceptions import CustomException
from src.utils import load_object


class UnifiedPredictPipeline:
    """Unified pipeline that handles both ML and transformer models"""

    def __init__(self, model_type="auto"):
        """
        Initialize prediction pipeline

        Args:
            model_type: "ml", "transformer", or "auto" (auto-detect)
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.tokenizer = None
        self.metadata = None

    def _detect_model_type(self):
        """Auto-detect which type of model is available"""
        # Check for transformer model
        transformer_path = "artifacts/transformers"
        if os.path.exists(transformer_path):
            # Get the first model directory
            models = [
                d
                for d in os.listdir(transformer_path)
                if os.path.isdir(os.path.join(transformer_path, d))
            ]
            if models:
                return "transformer", os.path.join(transformer_path, models[0])

        # Check for traditional ML model
        ml_model_path = "artifacts/model.pkl"
        if os.path.exists(ml_model_path):
            return "ml", ml_model_path

        raise CustomException("No model found in artifacts", sys)

    def load_model(self):
        """Load the appropriate model based on type"""
        try:
            if self.model_type == "auto":
                detected_type, model_path = self._detect_model_type()
                self.model_type = detected_type
                print(f"🔍 Auto-detected model type: {self.model_type}")

            if self.model_type == "ml":
                self._load_ml_model()
            elif self.model_type == "transformer":
                self._load_transformer_model()
            else:
                raise CustomException(f"Unknown model type: {self.model_type}", sys)

            print("✓ Model loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def _load_ml_model(self):
        """Load traditional ML model"""
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")

        self.model = load_object(file_path=model_path)
        self.preprocessor = load_object(file_path=preprocessor_path)

    def _load_transformer_model(self):
        """Load transformer model (RoBERTa, BERT, etc.)"""
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
        except ImportError:
            raise CustomException(
                "transformers library not installed. Run: pip install transformers",
                sys,
            )

        # Find transformer model
        transformer_dir = "artifacts/transformers"
        models = [
            d
            for d in os.listdir(transformer_dir)
            if os.path.isdir(os.path.join(transformer_dir, d))
        ]

        if not models:
            raise CustomException("No transformer model found", sys)

        model_path = os.path.join(transformer_dir, models[0])

        # Load metadata
        metadata_path = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode

    def predict(self, features):
        """
        Make predictions using the loaded model

        Args:
            features: pandas DataFrame with input features

        Returns:
            numpy array of predictions
        """
        try:
            if self.model is None:
                self.load_model()

            if self.model_type == "ml":
                return self._predict_ml(features)
            elif self.model_type == "transformer":
                return self._predict_transformer(features)

        except Exception as e:
            raise CustomException(e, sys)

    def _predict_ml(self, features):
        """Predict using traditional ML model"""
        data_scaled = self.preprocessor.transform(features)
        predictions = self.model.predict(data_scaled)
        return predictions

    def _predict_transformer(self, features):
        """Predict using transformer model"""
        try:
            import torch
        except ImportError:
            raise CustomException(
                "torch not installed. Run: pip install torch", sys
            )

        # Convert features to text
        text_inputs = self._features_to_text(features)

        # Tokenize
        max_length = self.metadata.get("max_length", 128) if self.metadata else 128
        inputs = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.squeeze().numpy()

        # Handle single prediction case
        if predictions.ndim == 0:
            predictions = np.array([predictions.item()])

        return predictions

    def _features_to_text(self, features):
        """Convert DataFrame features to text for transformer input"""
        texts = []
        for _, row in features.iterrows():
            text = f"""Student profile:
Gender: {row.get('gender', 'N/A')},
Ethnicity: {row.get('race_ethnicity', 'N/A')},
Parent Education: {row.get('parental_level_of_education', 'N/A')},
Lunch: {row.get('lunch', 'N/A')},
Test Prep: {row.get('test_preparation_course', 'N/A')},
Reading Score: {row.get('reading_score', 'N/A')},
Writing Score: {row.get('writing_score', 'N/A')}"""
            texts.append(text)

        return texts


class CustomData:
    """Data class for single prediction input (same as before)"""

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Test the pipeline
    print("🧪 Testing Unified Prediction Pipeline\n")

    # Create test data
    test_data = CustomData(
        gender="male",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="completed",
        reading_score=80,
        writing_score=75,
    )

    df = test_data.get_data_as_data_frame()

    # Initialize pipeline (auto-detect model type)
    pipeline = UnifiedPredictPipeline(model_type="auto")

    # Make prediction
    prediction = pipeline.predict(df)

    print(f"\n📊 Prediction: {prediction[0]:.2f}")
    print(f"   Model Type: {pipeline.model_type}")
