"""
Dataset utilities for FeatureEnhancer
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Union
from sklearn.model_selection import train_test_split
from pathlib import Path


class DatasetLoader:
    """
    Utility class for loading and preprocessing datasets for FeatureEnhancer.
    """

    @staticmethod
    def load_csv(
        filepath: str,
        target_column: Union[str, int] = -1,
        feature_columns: Optional[List[Union[str, int]]] = None,
        drop_columns: Optional[List[Union[str, int]]] = None,
        sep: str = ",",
        header: Union[int, str] = 0,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a CSV dataset and separate features from target.

        Args:
            filepath: Path to the CSV file
            target_column: Column to use as target (name or index). Default is last column.
            feature_columns: Specific columns to use as features. If None, uses all except target.
            drop_columns: Columns to drop before processing
            sep: Column separator
            header: Row to use as column names
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            X: Feature DataFrame
            y: Target Series
        """
        # Load the dataset
        df = pd.read_csv(filepath, sep=sep, header=header, **kwargs)

        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")

        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=drop_columns)
            print(f"Dropped columns: {drop_columns}")

        # Handle target column
        if isinstance(target_column, str):
            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found in dataset"
                )
            y = df[target_column]
            target_col_name = target_column
        elif isinstance(target_column, int):
            if target_column >= len(df.columns) or target_column < -len(df.columns):
                raise ValueError(f"Target column index {target_column} out of range")
            target_col_name = df.columns[target_column]
            y = df.iloc[:, target_column]
        else:
            raise ValueError(
                "target_column must be string (column name) or int (column index)"
            )

        # Handle feature columns
        if feature_columns is None:
            # Use all columns except target
            X = df.drop(columns=[target_col_name])
        else:
            # Use specified feature columns
            if isinstance(feature_columns[0], str):
                X = df[feature_columns]
            else:
                X = df.iloc[:, feature_columns]

        print(f"Features: {X.shape[1]} columns")
        print(f"Target: '{target_col_name}' (type: {y.dtype})")

        # Check for missing values
        if X.isnull().sum().sum() > 0:
            print(f"Warning: {X.isnull().sum().sum()} missing values found in features")

        if y.isnull().sum() > 0:
            print(f"Warning: {y.isnull().sum()} missing values found in target")

        return X, y

    @staticmethod
    def preprocess_dataset(
        X: pd.DataFrame,
        y: pd.Series,
        handle_missing: str = "drop",
        encode_categorical: bool = True,
        target_type: str = "auto",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the dataset for FeatureEnhancer.

        Args:
            X: Feature DataFrame
            y: Target Series
            handle_missing: How to handle missing values ('drop', 'mean', 'median', 'mode')
            encode_categorical: Whether to encode categorical features
            target_type: Target type ('auto', 'classification', 'regression')

        Returns:
            X_processed: Processed features
            y_processed: Processed target
        """
        X_processed = X.copy()
        y_processed = y.copy()

        print(f"Preprocessing dataset...")

        # Handle missing values
        if handle_missing == "drop":
            # Drop rows with any missing values
            mask = ~(X_processed.isnull().any(axis=1) | y_processed.isnull())
            X_processed = X_processed[mask]
            y_processed = y_processed[mask]
            print(f"Dropped rows with missing values. New shape: {X_processed.shape}")

        elif handle_missing in ["mean", "median"]:
            # Fill numerical columns with mean/median
            numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
            if handle_missing == "mean":
                X_processed[numerical_cols] = X_processed[numerical_cols].fillna(
                    X_processed[numerical_cols].mean()
                )
            else:
                X_processed[numerical_cols] = X_processed[numerical_cols].fillna(
                    X_processed[numerical_cols].median()
                )

            # Fill categorical columns with mode
            categorical_cols = X_processed.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])

            print(f"Filled missing values with {handle_missing}/mode")

        # Encode categorical features
        if encode_categorical:
            categorical_cols = X_processed.select_dtypes(exclude=[np.number]).columns
            if len(categorical_cols) > 0:
                print(f"Encoding categorical columns: {list(categorical_cols)}")
                X_processed = pd.get_dummies(
                    X_processed, columns=categorical_cols, drop_first=True
                )
                print(f"After encoding: {X_processed.shape[1]} features")

        # Determine target type
        if target_type == "auto":
            if y_processed.dtype == "object" or y_processed.nunique() < 5:
                target_type = "classification"
            else:
                target_type = "regression"

        print(f"Target type detected: {target_type}")

        # Encode target if classification
        if target_type == "classification" and y_processed.dtype == "object":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_processed = pd.Series(
                le.fit_transform(y_processed), index=y_processed.index
            )
            print(f"Encoded target labels: {le.classes_}")

        return X_processed, y_processed

    @staticmethod
    def get_dataset_info(X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Get comprehensive information about the dataset.

        Args:
            X: Features
            y: Target

        Returns:
            Dictionary with dataset information
        """
        info = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
            "feature_types": X.dtypes.to_dict(),
            "target_name": y.name or "target",
            "target_type": y.dtype,
            "target_unique_values": y.nunique(),
            "missing_values_features": X.isnull().sum().to_dict(),
            "missing_values_target": y.isnull().sum(),
            "numerical_features": list(X.select_dtypes(include=[np.number]).columns),
            "categorical_features": list(X.select_dtypes(exclude=[np.number]).columns),
        }

        # Statistical summary for numerical features
        if info["numerical_features"]:
            info["numerical_stats"] = X[info["numerical_features"]].describe().to_dict()

        # Value counts for categorical features (top 5)
        if info["categorical_features"]:
            info["categorical_stats"] = {}
            for col in info["categorical_features"]:
                info["categorical_stats"][col] = X[col].value_counts().head().to_dict()

        # Target statistics
        if y.dtype in [np.number, "int64", "float64"]:
            info["target_stats"] = {
                "mean": y.mean(),
                "std": y.std(),
                "min": y.min(),
                "max": y.max(),
                "median": y.median(),
            }
        else:
            info["target_stats"] = y.value_counts().to_dict()

        return info


def print_dataset_summary(info: dict):
    """Print a nice summary of dataset information."""
    print(f"\n=== Dataset Summary ===")
    print(f"Samples: {info['n_samples']:,}")
    print(f"Features: {info['n_features']}")
    print(f"Target: {info['target_name']} ({info['target_type']})")
    print(f"Unique target values: {info['target_unique_values']}")

    if info["numerical_features"]:
        print(
            f"\nNumerical features ({len(info['numerical_features'])}): {info['numerical_features'][:5]}{'...' if len(info['numerical_features']) > 5 else ''}"
        )

    if info["categorical_features"]:
        print(
            f"Categorical features ({len(info['categorical_features'])}): {info['categorical_features'][:5]}{'...' if len(info['categorical_features']) > 5 else ''}"
        )

    # Missing values
    total_missing_features = sum(
        v for v in info["missing_values_features"].values() if v > 0
    )
    if total_missing_features > 0 or info["missing_values_target"] > 0:
        print(f"\nMissing values:")
        print(f"  Features: {total_missing_features}")
        print(f"  Target: {info['missing_values_target']}")
    else:
        print(f"\nNo missing values found ✓")

    print(f"=" * 25)
