"""
Fake News Detection Prediction Pipeline

This pipeline loads the RoBERTa model trained on Kaggle and makes predictions
on news articles to detect if they are real or fake.
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path

from src.exceptions import CustomException


class FakeNewsPredictionPipeline:
    """Pipeline for fake news detection using RoBERTa"""

    def __init__(self, model_name="roberta_fakenews_model"):
        """
        Initialize prediction pipeline

        Args:
            model_name: Name of the model directory in artifacts/transformers/
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.metadata = None
        self.max_length = 256

    def load_model(self):
        """Load the RoBERTa model and tokenizer"""
        try:
            from transformers import (
                RobertaTokenizer,
                RobertaForSequenceClassification,
            )
        except ImportError:
            raise CustomException(
                "transformers library not installed. Run: pip install transformers torch",
                sys,
            )

        # Find model path
        model_path = os.path.join("artifacts", "transformers", self.model_name)

        if not os.path.exists(model_path):
            raise CustomException(
                f"Model not found at: {model_path}\n"
                f"Please integrate your Kaggle model first.",
                sys,
            )

        print(f"📦 Loading model from: {model_path}")

        # Load metadata
        metadata_path = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.max_length = self.metadata.get("max_length", 256)

        # Load label mapping
        label_map_path = os.path.join(model_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                self.label_map = json.load(f)
        else:
            self.label_map = {0: "Real News", 1: "Fake News"}

        # Load model and tokenizer
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model.eval()

        print("✓ Model loaded successfully")

    def predict(self, texts):
        """
        Predict if news articles are real or fake

        Args:
            texts: List of text strings or single text string

        Returns:
            dict with predictions, probabilities, and labels
        """
        try:
            if self.model is None:
                self.load_model()

            # Handle single text
            if isinstance(texts, str):
                texts = [texts]

            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

            # Convert to readable format
            results = []
            for i, pred in enumerate(predictions):
                pred_label = int(pred.item())
                probs = probabilities[i].tolist()

                result = {
                    "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                    "prediction": self.label_map.get(str(pred_label), f"Label {pred_label}"),
                    "prediction_id": pred_label,
                    "confidence": max(probs),
                    "probabilities": {
                        "real": probs[0],
                        "fake": probs[1],
                    },
                }
                results.append(result)

            return results

        except Exception as e:
            raise CustomException(e, sys)

    def predict_single(self, title, text):
        """
        Predict for a single news article with title and body

        Args:
            title: News article title
            text: News article text/body

        Returns:
            dict with prediction result
        """
        # Combine title and text (same format as training)
        combined = f"{title.lower()} [SEP] {text.lower()}"
        result = self.predict([combined])[0]
        return result


# ==================== Example Usage ====================
if __name__ == "__main__":
    print("\n🧪 Testing Fake News Detection Pipeline\n")

    # Initialize pipeline
    pipeline = FakeNewsPredictionPipeline()

    # Example 1: Real news
    real_news = {
        "title": "Scientists discover new species in Amazon rainforest",
        "text": """Researchers from the University of São Paulo have discovered
        a new species of frog in the Amazon rainforest. The discovery was made
        during a six-month expedition studying biodiversity in the region.""",
    }

    # Example 2: Fake news (made up)
    fake_news = {
        "title": "Aliens land in New York City, demand pizza",
        "text": """Extraterrestrial beings landed in Times Square yesterday and
        reportedly demanded pepperoni pizza from local restaurants. The incident
        was witnessed by thousands of people.""",
    }

    print("Example 1: Real News Article")
    print("-" * 50)
    result1 = pipeline.predict_single(real_news["title"], real_news["text"])
    print(f"Title: {real_news['title']}")
    print(f"Prediction: {result1['prediction']}")
    print(f"Confidence: {result1['confidence']:.2%}")
    print(f"Probabilities: Real={result1['probabilities']['real']:.2%}, "
          f"Fake={result1['probabilities']['fake']:.2%}")

    print("\n" + "=" * 50 + "\n")

    print("Example 2: Fake News Article")
    print("-" * 50)
    result2 = pipeline.predict_single(fake_news["title"], fake_news["text"])
    print(f"Title: {fake_news['title']}")
    print(f"Prediction: {result2['prediction']}")
    print(f"Confidence: {result2['confidence']:.2%}")
    print(f"Probabilities: Real={result2['probabilities']['real']:.2%}, "
          f"Fake={result2['probabilities']['fake']:.2%}")

    print("\n✅ Pipeline test complete!\n")
