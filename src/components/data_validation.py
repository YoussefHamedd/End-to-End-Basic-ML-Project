"""
Data Validation Module
Validates data quality and schema for fake news detection
"""
import os
import sys
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
from src.exceptions import CustomException
from src.logger import logging


@dataclass
class DataValidationConfig:
    """Configuration for data validation"""
    required_columns: List[str] = None
    min_samples: int = 100
    max_missing_ratio: float = 0.1  # Max 10% missing values
    min_text_length: int = 10
    valid_labels: List[int] = None

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['text', 'label']
        if self.valid_labels is None:
            self.valid_labels = [0, 1]  # Binary classification


class DataValidation:
    """Validate data quality before training"""

    def __init__(self):
        self.config = DataValidationConfig()
        self.validation_errors = []
        logging.info("Data Validation initialized")

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate dataset schema

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            logging.info("Validating schema...")

            # Check required columns
            missing_cols = set(self.config.required_columns) - set(df.columns)
            if missing_cols:
                error = f"Missing required columns: {missing_cols}"
                self.validation_errors.append(error)
                logging.error(error)
                return False

            logging.info("✓ Schema validation passed")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            logging.info("Validating data quality...")
            is_valid = True

            # Check minimum samples
            if len(df) < self.config.min_samples:
                error = f"Insufficient samples: {len(df)} < {self.config.min_samples}"
                self.validation_errors.append(error)
                logging.error(error)
                is_valid = False

            # Check missing values
            for col in self.config.required_columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > self.config.max_missing_ratio:
                    error = f"Too many missing values in {col}: {missing_ratio:.2%}"
                    self.validation_errors.append(error)
                    logging.error(error)
                    is_valid = False

            # Check text length
            if 'text' in df.columns:
                df['text_length'] = df['text'].astype(str).str.len()
                short_texts = (df['text_length'] < self.config.min_text_length).sum()
                if short_texts > len(df) * 0.1:  # More than 10% short texts
                    warning = f"Many short texts detected: {short_texts} samples"
                    logging.warning(warning)

            # Check labels
            if 'label' in df.columns:
                invalid_labels = ~df['label'].isin(self.config.valid_labels)
                if invalid_labels.any():
                    error = f"Invalid labels found: {df[invalid_labels]['label'].unique()}"
                    self.validation_errors.append(error)
                    logging.error(error)
                    is_valid = False

                # Check label balance
                label_counts = df['label'].value_counts()
                min_label_ratio = label_counts.min() / len(df)
                if min_label_ratio < 0.1:  # Less than 10%
                    warning = f"Imbalanced dataset: {label_counts.to_dict()}"
                    logging.warning(warning)

            if is_valid:
                logging.info("✓ Data quality validation passed")
            else:
                logging.error("✗ Data quality validation failed")

            return is_valid

        except Exception as e:
            raise CustomException(e, sys)

    def validate_labels(self, df: pd.DataFrame) -> bool:
        """
        Validate labels specifically

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            logging.info("Validating labels...")

            if 'label' not in df.columns:
                error = "Label column not found"
                self.validation_errors.append(error)
                logging.error(error)
                return False

            # Check label types
            unique_labels = df['label'].unique()
            invalid = set(unique_labels) - set(self.config.valid_labels)
            if invalid:
                error = f"Invalid label values: {invalid}"
                self.validation_errors.append(error)
                logging.error(error)
                return False

            # Check label distribution
            label_dist = df['label'].value_counts()
            logging.info(f"Label distribution:\n{label_dist}")

            if len(label_dist) < 2:
                error = "Only one class present in labels"
                self.validation_errors.append(error)
                logging.error(error)
                return False

            logging.info("✓ Label validation passed")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_all(self, df: pd.DataFrame) -> Dict:
        """
        Run all validations

        Args:
            df: DataFrame to validate

        Returns:
            Dict with validation results
        """
        try:
            self.validation_errors = []  # Reset errors

            schema_valid = self.validate_schema(df)
            quality_valid = self.validate_data_quality(df)
            labels_valid = self.validate_labels(df)

            is_valid = schema_valid and quality_valid and labels_valid

            results = {
                'is_valid': is_valid,
                'schema_valid': schema_valid,
                'quality_valid': quality_valid,
                'labels_valid': labels_valid,
                'errors': self.validation_errors,
                'sample_count': len(df),
                'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
            }

            if is_valid:
                logging.info("✅ All validations passed")
            else:
                logging.error(f"❌ Validation failed. Errors: {self.validation_errors}")

            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data validation
    import pandas as pd

    # Valid dataset
    test_data = {
        'text': ['This is news ' * 10 for _ in range(200)],
        'label': [0] * 100 + [1] * 100
    }
    df = pd.DataFrame(test_data)

    validator = DataValidation()
    results = validator.validate_all(df)

    print("✓ Data validation test")
    print(f"Results: {results}")
