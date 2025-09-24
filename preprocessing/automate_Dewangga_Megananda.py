#!/usr/bin/env python3
"""
Automated Preprocessing Script for Heart Disease Classification
Author: Dewangga Megananda
Level: Skilled

This script automates the preprocessing pipeline:
1. Load raw data
2. Handle missing values
3. Encode categorical variables
4. Scale features
5. Create train/test split
6. Save processed data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeartDiseasePreprocessor:
    """
    Automated preprocessor for Heart Disease dataset
    """

    def __init__(self, input_path=None, output_dir='dataset_preprocessing'):
        """
        Initialize preprocessor

        Args:
            input_path (str): Path to input CSV file
            output_dir (str): Directory to save processed data
        """
        self.input_path = input_path or 'dataset_preprocessing/heart_disease_full.csv'
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("HeartDiseasePreprocessor initialized")
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """
        Load dataset from CSV file

        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Input file not found: {self.input_path}")

            df = pd.read_csv(self.input_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        logger.info("Handling missing values...")

        # Check for missing values
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before handling: {missing_before}")

        # Handle missing values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])

        # Handle missing values for categorical columns (if any)
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")

        return df

    def encode_categorical_variables(self, df):
        """
        Encode categorical variables using Label Encoding

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")

        # Define categorical columns (based on UCI Heart Disease dataset)
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

        # Apply label encoding
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                logger.info(f"Encoded column: {col}")

        return df

    def scale_features(self, df, target_col='target'):
        """
        Scale numerical features using StandardScaler

        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column to exclude from scaling

        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        logger.info("Scaling features...")

        # Features to scale (exclude target)
        features_to_scale = [col for col in df.columns if col != target_col]

        # Apply scaling
        df_scaled = df.copy()
        df_scaled[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])

        logger.info(f"Scaled {len(features_to_scale)} features")
        return df_scaled

    def create_train_test_split(self, df, test_size=0.2, random_state=42, target_col='target'):
        """
        Create train/test split with stratification

        Args:
            df (pd.DataFrame): Input dataframe
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            target_col (str): Name of target column

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Creating train/test split...")

        # Split features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Create stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """
        Save processed train and test data to CSV files

        Args:
            X_train, X_test, y_train, y_test: Split datasets
        """
        logger.info("Saving processed data...")

        # Combine features and target for saving
        train_df = X_train.copy()
        train_df['target'] = y_train

        test_df = X_test.copy()
        test_df['target'] = y_test

        # Save to CSV
        train_path = os.path.join(self.output_dir, 'train_processed.csv')
        test_path = os.path.join(self.output_dir, 'test_processed.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Train data saved to: {train_path}")
        logger.info(f"Test data saved to: {test_path}")

    def preprocess_pipeline(self):
        """
        Execute complete preprocessing pipeline

        Returns:
            tuple: (X_train, X_test, y_train, y_test) processed datasets
        """
        logger.info("=== STARTING PREPROCESSING PIPELINE ===")

        try:
            # Step 1: Load data
            df = self.load_data()

            # Step 2: Handle missing values
            df = self.handle_missing_values(df)

            # Step 3: Encode categorical variables
            df = self.encode_categorical_variables(df)

            # Step 4: Scale features
            df_scaled = self.scale_features(df)

            # Step 5: Create train/test split
            X_train, X_test, y_train, y_test = self.create_train_test_split(df_scaled)

            # Step 6: Save processed data
            self.save_processed_data(X_train, X_test, y_train, y_test)

            logger.info("=== PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY ===")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise

def main():
    """
    Main function to run preprocessing
    """
    parser = argparse.ArgumentParser(description='Automated Heart Disease Data Preprocessing')
    parser.add_argument('--input', '-i', type=str, help='Path to input CSV file')
    parser.add_argument('--output', '-o', type=str, default='dataset_preprocessing',
                       help='Output directory for processed data')

    args = parser.parse_args()

    print("=== Heart Disease Automated Preprocessing ===")
    print("Author: Dewangga Megananda")
    print("Level: Skilled")
    print()

    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor(
        input_path=args.input,
        output_dir=args.output
    )

    # Run preprocessing pipeline
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()

        print("\n✅ Preprocessing completed successfully!")
        print(f"📊 Train set shape: {X_train.shape}")
        print(f"📊 Test set shape: {X_test.shape}")
        print(f"📁 Output directory: {args.output}")

    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())