"""
Data Preparation Module
Handles text cleaning and preprocessing for fake news detection
"""
import os
import sys
import re
import pandas as pd
from dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging


@dataclass
class DataPreparationConfig:
    """Configuration for data preparation"""
    min_text_length: int = 10
    max_length: int = 512
    test_size: float = 0.2
    random_state: int = 42


class DataPreparation:
    """Clean and prepare text data for model training"""

    def __init__(self):
        self.config = DataPreparationConfig()
        logging.info("Data Preparation initialized")

    def clean_text(self, text):
        """
        Clean text data

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"www\S+", "", text)  # Remove www links
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Remove special chars
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        text = re.sub(r"\d+", "", text)  # Remove numbers (optional)

        return text

    def combine_title_text(self, title, text):
        """
        Combine title and text with separator

        Args:
            title: Article title
            text: Article text

        Returns:
            Combined string with [SEP] token
        """
        title_clean = self.clean_text(title) if title else ""
        text_clean = self.clean_text(text) if text else ""

        if title_clean and text_clean:
            return f"{title_clean} [SEP] {text_clean}"
        elif text_clean:
            return text_clean
        elif title_clean:
            return title_clean
        else:
            return ""

    def prepare_dataset(self, df):
        """
        Prepare dataset for training

        Args:
            df: DataFrame with columns [text, label, optional: title]

        Returns:
            Cleaned DataFrame ready for training
        """
        try:
            logging.info("Starting data preparation...")

            # Add title column if not present
            if 'title' not in df.columns:
                df['title'] = ''

            # Clean text
            logging.info("Cleaning text data...")
            df['text_clean'] = df['text'].apply(self.clean_text)
            df['title_clean'] = df['title'].apply(self.clean_text)

            # Combine title and text
            logging.info("Combining title and text...")
            df['combined'] = df.apply(
                lambda row: self.combine_title_text(row['title'], row['text']),
                axis=1
            )

            # Filter out short texts
            df = df[df['combined'].str.len() >= self.config.min_text_length]

            # Filter out very long texts
            df = df[df['combined'].str.len() <= self.config.max_length]

            # Drop NaN labels
            df = df.dropna(subset=['label'])

            # Keep only necessary columns
            df_final = df[['combined', 'label']].copy()
            df_final.columns = ['text', 'label']

            logging.info(f"Data preparation completed: {len(df_final)} samples")

            return df_final

        except Exception as e:
            raise CustomException(e, sys)

    def get_statistics(self, df):
        """
        Get dataset statistics

        Args:
            df: DataFrame

        Returns:
            Dict with statistics
        """
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'min_text_length': df['text'].str.len().min(),
            'max_text_length': df['text'].str.len().max(),
            'null_values': df.isnull().sum().to_dict()
        }

        logging.info(f"Dataset statistics: {stats}")
        return stats


if __name__ == "__main__":
    # Test data preparation
    import pandas as pd

    test_data = {
        'title': ['Breaking News', 'Scientists Discover'],
        'text': ['This is fake news!!!', 'Real scientific discovery'],
        'label': [1, 0]
    }

    df = pd.DataFrame(test_data)
    prep = DataPreparation()

    df_clean = prep.prepare_dataset(df)
    stats = prep.get_statistics(df_clean)

    print("✓ Data preparation test passed")
    print(f"Stats: {stats}")
