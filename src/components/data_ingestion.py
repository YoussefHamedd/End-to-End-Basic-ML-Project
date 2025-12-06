"""
Data Ingestion for Fake News Detection
Loads and splits the fake news dataset
"""
import os
import sys
import re
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestionFakeNews:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def clean_text(self, text):
        """Clean text data"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Remove special chars
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    def initiate_data_ingestion(self, data_path='data/fake_news.csv', sample_size=None):
        """
        Load fake news dataset and prepare for training

        Args:
            data_path: Path to CSV file with columns [title, text, label]
            sample_size: Optional - limit dataset size for quick testing
        """
        logging.info("Entered the fake news data ingestion method")

        try:
            # Load dataset
            logging.info(f"Reading dataset from {data_path}")
            df = pd.read_csv(data_path)

            # Verify required columns exist
            required_cols = ['text', 'label']
            if 'title' not in df.columns:
                df['title'] = ''  # Add empty title if not present

            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in dataset")

            logging.info(f'Dataset loaded: {df.shape}')
            logging.info(f'Label distribution:\n{df["label"].value_counts()}')

            # Sample if requested (for quick testing)
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                logging.info(f'Sampled {sample_size} records for testing')

            # Clean text
            logging.info("Cleaning text data...")
            df['text_clean'] = df['text'].apply(self.clean_text)
            df['title_clean'] = df['title'].apply(self.clean_text)

            # Combine title and text
            df['combined'] = df['title_clean'] + " [SEP] " + df['text_clean']

            # Filter out too short texts
            df = df[df['text_clean'].str.len() > 10]
            df = df.dropna(subset=['label'])

            logging.info(f'After cleaning: {len(df)} samples')

            # Keep only necessary columns
            df_final = df[['combined', 'label']].copy()
            df_final.columns = ['text', 'label']

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df_final.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-test split with stratification
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df_final,
                test_size=0.2,
                random_state=42,
                stratify=df_final['label']
            )

            # Save splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Data ingestion completed")
            logging.info(f"Train: {len(train_set)}, Test: {len(test_set)}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestionFakeNews()

    # Test with sample size for quick validation
    train_data, test_data = obj.initiate_data_ingestion(
        data_path='data/fake_news.csv',
        sample_size=1000  # Use None for full dataset
    )

    print(f"✓ Train data: {train_data}")
    print(f"✓ Test data: {test_data}")
