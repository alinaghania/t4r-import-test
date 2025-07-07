"""
Data Preprocessor for Transformers4Rec

This module handles data preparation and schema creation for training
sequential recommendation models with Transformers4Rec.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import tempfile
import os


class DataPreprocessor:
    """
    Prepares banking data for Transformers4Rec training.
    
    Handles data cleaning, type conversion, temporal splitting,
    and Merlin schema creation for sequential recommendation models.
    """
    
    def __init__(self, test_split_ratio: float = 0.2):
        """
        Initialize preprocessor.
        
        Args:
            test_split_ratio: Ratio for temporal train/test split
        """
        self.test_split_ratio = test_split_ratio
        self.schema_info = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for Transformers4Rec.
        
        Args:
            df: Raw banking interactions DataFrame
            
        Returns:
            Cleaned DataFrame with compatible data types
        """
        print("Cleaning data for Transformers4Rec compatibility...")
        
        df_clean = df.copy()
        
        # Convert timestamps to Unix timestamp (int64)
        # First ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean["timestamp"]):
            print("Converting timestamp strings to datetime...")
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
        
        # Convert to Unix timestamp
        df_clean["timestamp"] = df_clean["timestamp"].astype("int64") // 10**9
        
        # Convert categorical string columns to numeric codes
        categorical_string_cols = ["product_name", "product_category", "channel", "customer_segment"]
        for col in categorical_string_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.Categorical(df_clean[col]).codes.astype("int32")
        
        # Ensure proper data types for DLPack compatibility
        type_conversions = {
            "item_id": "int32",
            "customer_id": "int32", 
            "session_id": "int32",
            "amount": "float32",
            "customer_age": "int32",
            "customer_income": "int32",
            "position_in_sequence": "int32",
            "revenue_potential": "float32",
        }
        
        for col, target_type in type_conversions.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(target_type)
        
        print(f"Data cleaned: {len(df_clean)} rows with {len(df_clean.columns)} columns")
        return df_clean
    
    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally for training and testing.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print(f"Splitting data temporally ({1-self.test_split_ratio:.0%} train, {self.test_split_ratio:.0%} test)...")
        
        # Calculate split point based on timestamp quantile
        cutoff_timestamp = df["timestamp"].quantile(1 - self.test_split_ratio)
        
        train_df = df[df["timestamp"] <= cutoff_timestamp].copy()
        test_df = df[df["timestamp"] > cutoff_timestamp].copy()
        
        print(f"Train set: {len(train_df)} interactions")
        print(f"Test set: {len(test_df)} interactions")
        
        return train_df, test_df
    
    def create_merlin_schema(self, df: pd.DataFrame) -> Dict:
        """
        Create schema information for Transformers4Rec (format compatible avec model_trainer).
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Schema information dictionary compatible avec les deux formats
        """
        print("Creating schema info compatible with model_trainer...")
        
        # Format SIMPLE (actuel)
        categorical_cols = [
            "item_id",
            "customer_id", 
            "session_id",
        ]
        
        # Add encoded categorical features if they exist
        encoded_categoricals = ["product_category", "channel", "customer_segment"]
        for col in encoded_categoricals:
            if col in df.columns and df[col].dtype == "int32":
                categorical_cols.append(col)
        
        continuous_cols = ["amount", "timestamp"]
        
        # Add demographic features 
        demographic_features = ["customer_age", "customer_income"]
        for col in demographic_features:
            if col in df.columns:
                continuous_cols.append(col)
        
        # Format COMPATIBLE avec model_trainer (ajouter les deux formats)
        schema_info = {
            # Format simple (actuel)
            "target": "item_id",
            "user_id": "customer_id", 
            "session_id": "session_id",
            "categorical": categorical_cols,
            "continuous": continuous_cols,
            
            # Format complexe (attendu par model_trainer) - AJOUT
            "categorical_features": [
                ("item_id", ["CATEGORICAL", "ITEM_ID"]),
                ("customer_id", ["CATEGORICAL", "USER_ID"]),
                ("session_id", ["CATEGORICAL", "SESSION_ID"]),
            ],
            "continuous_features": continuous_cols
        }
        
        # Ajouter les features catégorielles encodées au format complexe
        for col in encoded_categoricals:
            if col in df.columns and df[col].dtype == "int32":
                schema_info["categorical_features"].append((col, ["CATEGORICAL"]))
        
        self.schema_info = schema_info
        print(f"Schema created with {len(categorical_cols)} categorical and {len(continuous_cols)} continuous features")
        print(f"✅ Compatible avec les deux formats (simple + complexe)")
        
        return schema_info
    
    def save_processed_data(self, 
                           train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           output_dir: str = "./processed_data") -> Dict[str, str]:
        """
        Save processed data to disk.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            output_dir: Output directory path
            
        Returns:
            Dictionary with file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save DataFrames
        train_path = output_path / "train.parquet"
        test_path = output_path / "test.parquet"
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        # Save schema info if available
        schema_path = None
        if self.schema_info:
            import json
            schema_path = output_path / "schema_info.json"
            with open(schema_path, "w") as f:
                json.dump(self.schema_info, f, indent=2)
        
        file_paths = {
            "train": str(train_path),
            "test": str(test_path),
            "schema": str(schema_path) if schema_path else None,
        }
        
        print(f"Data saved to {output_dir}")
        return file_paths
    
    def prepare_for_transformers4rec(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline for Transformers4Rec.
        
        Args:
            df: Raw banking interactions DataFrame
            
        Returns:
            Tuple of (train_df, test_df, schema_info)
        """
        print("Starting complete preprocessing pipeline...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Temporal split
        train_df, test_df = self.temporal_split(df_clean)
        
        # Create schema
        schema_info = self.create_merlin_schema(df_clean)
        
        print("Preprocessing pipeline completed successfully!")
        return train_df, test_df, schema_info
    
    def get_data_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for processed data.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_customers": train_df["customer_id"].nunique(),
            "test_customers": test_df["customer_id"].nunique(),
            "train_products": train_df["item_id"].nunique(),
            "test_products": test_df["item_id"].nunique(),
            "avg_sequence_length_train": train_df.groupby("customer_id").size().mean(),
            "avg_sequence_length_test": test_df.groupby("customer_id").size().mean(),
            "temporal_split_point": train_df["timestamp"].max(),
            "data_types": dict(train_df.dtypes),
        }
        
        return summary


def check_transformers4rec_availability() -> bool:
    """
    Check if Transformers4Rec is available in the environment.
    
    Returns:
        True if T4Rec is available
        
    Raises:
        RuntimeError: If Transformers4Rec is not available
    """
    try:
        import transformers4rec.torch as tr
        import merlin.schema as ms
        print("✅ Transformers4Rec is available")
        return True
    except ImportError as e:
        raise RuntimeError(
            f"❌ Transformers4Rec not available: {e}\n"
            "Install with: pip install 'transformers4rec[torch]>=23.02.00'"
        )


if __name__ == "__main__":
    # Example usage
    print("Testing Data Preprocessor...")
    
    # Check T4Rec availability
    t4rec_available = check_transformers4rec_availability()
    
    # Load sample data (assuming it exists)
    try:
        df = pd.read_csv("banking_interactions.csv")
        print(f"Loaded dataset with {len(df)} rows")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(test_split_ratio=0.2)
        
        # Run preprocessing pipeline
        train_df, test_df, schema_info = preprocessor.prepare_for_transformers4rec(df)
        
        # Get summary
        summary = preprocessor.get_data_summary(train_df, test_df)
        print("\nData Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Save processed data
        file_paths = preprocessor.save_processed_data(train_df, test_df)
        print(f"\nFiles saved: {file_paths}")
        
    except FileNotFoundError:
        print("banking_interactions.csv not found. Run data_generator.py first.")