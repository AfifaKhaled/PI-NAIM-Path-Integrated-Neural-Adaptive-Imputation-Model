# pi_naim/utils/mimic_data.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from typing import Dict, List, Tuple, Optional


class MIMICDataset(Dataset):
    def __init__(self, features, labels, mask=None, subject_ids=None, hadm_ids=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32) if mask is not None else torch.ones_like(self.features)
        self.subject_ids = subject_ids
        self.hadm_ids = hadm_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            "X_true": self.features[idx],
            "y": self.labels[idx],
            "mask": self.mask[idx]
        }
        if self.subject_ids is not None:
            item["subject_id"] = self.subject_ids[idx]
        if self.hadm_ids is not None:
            item["hadm_id"] = self.hadm_ids[idx]
        return item


class MIMICDataLoader:
    def __init__(self, data_dir: str, mimic_version: str = "iii"):
        self.data_dir = data_dir
        self.mimic_version = mimic_version
        self.tables_loaded = False
        self.patients = None
        self.admissions = None
        self.icustays = None
        self.diagnoses_icd = None

    def load_tables(self):
        """Load MIMIC tables from CSV files"""
        try:
            print("Loading MIMIC tables...")
            self.patients = pd.read_csv(os.path.join(self.data_dir, 'PATIENTS.csv'))
            self.admissions = pd.read_csv(os.path.join(self.data_dir, 'ADMISSIONS.csv'))
            self.icustays = pd.read_csv(os.path.join(self.data_dir, 'ICUSTAYS.csv'))
            self.diagnoses_icd = pd.read_csv(os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv'))
            self.tables_loaded = True
            print("MIMIC tables loaded successfully!")
        except Exception as e:
            print(f"Error loading MIMIC tables: {e}")
            raise

    # pi_naim/utils/mimic_data.py

    # Add this function to handle column name variations
    # pi_naim/utils/mimic_data.py

    # Change this method definition:
    def _standardize_column_names(self, df):
        """Standardize column names across different MIMIC versions"""
        column_mapping = {
            # Common variations for PATIENTS table
            'subject_id': ['subject_id', 'SUBJECT_ID', 'Subject_ID'],
            'gender': ['gender', 'GENDER', 'Gender'],
            'dob': ['dob', 'DOB', 'date_of_birth', 'Date_of_Birth'],

            # Common variations for ADMISSIONS table
            'hadm_id': ['hadm_id', 'HADM_ID', 'Hadm_ID'],
            'admittime': ['admittime', 'ADMITTIME', 'Admit_Time'],
            'dischtime': ['dischtime', 'DISCHTIME', 'Discharge_Time'],
            'admission_type': ['admission_type', 'ADMISSION_TYPE', 'Admission_Type'],
            'insurance': ['insurance', 'INSURANCE', 'Insurance'],
            'ethnicity': ['ethnicity', 'ETHNICITY', 'Ethnicity'],
            'hospital_expire_flag': ['hospital_expire_flag', 'HOSPITAL_EXPIRE_FLAG', 'Hospital_Expire_Flag'],

            # Common variations for ICUSTAYS table
            'intime': ['intime', 'INTIME', 'In_Time'],
            'outtime': ['outtime', 'OUTTIME', 'Out_Time'],
            'los': ['los', 'LOS', 'Length_of_Stay'],

            # Common variations for DIAGNOSES_ICD table
            'icd9_code': ['icd9_code', 'ICD9_CODE', 'ICD9_Code']
        }

        # Create a reverse mapping from possible column names to standard names
        reverse_mapping = {}
        for standard_name, variations in column_mapping.items():
            for variation in variations:
                reverse_mapping[variation.lower()] = standard_name

        # Rename columns
        new_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in reverse_mapping:
                new_columns.append(reverse_mapping[col_lower])
            else:
                new_columns.append(col)

        df.columns = new_columns
        return df

    # Then update your load_tables method to call it correctly:
    def load_tables(self):
        """Load MIMIC tables from CSV files with column name standardization"""
        try:
            print("Loading MIMIC tables...")
            self.patients = pd.read_csv(os.path.join(self.data_dir, 'PATIENTS.csv'))
            self.patients = self._standardize_column_names(self.patients)

            self.admissions = pd.read_csv(os.path.join(self.data_dir, 'ADMISSIONS.csv'))
            self.admissions = self._standardize_column_names(self.admissions)

            self.icustays = pd.read_csv(os.path.join(self.data_dir, 'ICUSTAYS.csv'))
            self.icustays = self._standardize_column_names(self.icustays)

            self.diagnoses_icd = pd.read_csv(os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv'))
            self.diagnoses_icd = self._standardize_column_names(self.diagnoses_icd)

            self.tables_loaded = True
            print("MIMIC tables loaded successfully!")

        except Exception as e:
            print(f"Error loading MIMIC tables: {e}")
            raise
    # Update the load_tables method
    def load_tables(self):
        """Load MIMIC tables from CSV files with column name standardization"""
        try:
            print("Loading MIMIC tables...")
            self.patients = pd.read_csv(os.path.join(self.data_dir, 'PATIENTS.csv'))
            self.patients = self._standardize_column_names(self.patients)

            self.admissions = pd.read_csv(os.path.join(self.data_dir, 'ADMISSIONS.csv'))
            self.admissions = self._standardize_column_names(self.admissions)

            self.icustays = pd.read_csv(os.path.join(self.data_dir, 'ICUSTAYS.csv'))
            self.icustays = self._standardize_column_names(self.icustays)

            self.diagnoses_icd = pd.read_csv(os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv'))
            self.diagnoses_icd = self._standardize_column_names(self.diagnoses_icd)

            self.tables_loaded = True
            print("MIMIC tables loaded successfully!")

            # Print column names for debugging
            print("Patients columns:", list(self.patients.columns))
            print("Admissions columns:", list(self.admissions.columns))
            print("Icustays columns:", list(self.icustays.columns))
            print("Diagnoses_icd columns:", list(self.diagnoses_icd.columns))

        except Exception as e:
            print(f"Error loading MIMIC tables: {e}")
            raise
    def preprocess_data(self, target_condition: str = "sepsis",
                        min_age: int = 18, max_age: int = 100,
                        feature_columns: Optional[List[str]] = None):
        """
        Preprocess MIMIC data for mortality prediction
        target_condition: "mortality", "sepsis", or specific ICD code
        """
        if not self.tables_loaded:
            self.load_tables()

        print("Preprocessing MIMIC data...")

        # Merge tables
        merged_data = self._merge_tables()

        # Filter by age
        merged_data = merged_data[
            (merged_data['age'] >= min_age) &
            (merged_data['age'] <= max_age)
            ]

        # Create target variable
        if target_condition == "mortality":
            merged_data['target'] = (merged_data['hospital_expire_flag'] == 1).astype(int)
        elif target_condition == "sepsis":
            # Sepsis ICD codes (simplified)
            sepsis_codes = ['99591', '99592', '78552']  # Add more specific codes
            merged_data['target'] = merged_data['icd9_code'].isin(sepsis_codes).astype(int)
        else:
            # Specific ICD code
            merged_data['target'] = (merged_data['icd9_code'] == target_condition).astype(int)

        # Extract features
        if feature_columns is None:
            feature_columns = [
                'age', 'gender_num', 'admission_type', 'insurance',
                'ethnicity_num', 'los_hospital', 'los_icu'
            ]

        features = merged_data[feature_columns].values
        labels = merged_data['target'].values

        # Store IDs for tracking
        subject_ids = merged_data['subject_id'].values
        hadm_ids = merged_data['hadm_id'].values

        print(f"Preprocessed data shape: {features.shape}")
        print(f"Class distribution: {np.bincount(labels)}")

        return features, labels, subject_ids, hadm_ids, feature_columns

    def _merge_tables(self):
        """Merge MIMIC tables into a single dataframe with proper date handling"""
        # Basic patient info
        patients_clean = self.patients[['subject_id', 'gender', 'dob']].copy()
        patients_clean['gender_num'] = patients_clean['gender'].map({'M': 0, 'F': 1})

        # Admissions info
        admissions_clean = self.admissions[[
            'subject_id', 'hadm_id', 'admittime', 'dischtime',
            'admission_type', 'insurance', 'ethnicity', 'hospital_expire_flag'
        ]].copy()

        # ICU stays
        icustays_clean = self.icustays[[
            'subject_id', 'hadm_id', 'intime', 'outtime', 'los'
        ]].copy()
        icustays_clean.rename(columns={'los': 'los_icu'}, inplace=True)

        # Diagnoses
        diagnoses_clean = self.diagnoses_icd[['subject_id', 'hadm_id', 'icd9_code']].copy()

        # Merge all tables
        merged = admissions_clean.merge(patients_clean, on='subject_id', how='inner')
        merged = merged.merge(icustays_clean, on=['subject_id', 'hadm_id'], how='inner')
        merged = merged.merge(diagnoses_clean, on=['subject_id', 'hadm_id'], how='left')

        # Convert date columns to datetime with error handling
        print("Converting date columns...")

        # Convert with error handling and filtering
        for date_col in ['admittime', 'dischtime', 'intime', 'outtime', 'dob']:
            if date_col in merged.columns:
                # Convert to datetime, coerce errors to NaT
                merged[date_col] = pd.to_datetime(merged[date_col], errors='coerce')

        # Filter out rows with invalid dates
        initial_count = len(merged)
        merged = merged.dropna(subset=['admittime', 'dob'])
        filtered_count = len(merged)
        print(f"Filtered {initial_count - filtered_count} rows with invalid dates")

        # Calculate age with overflow protection
        print("Calculating age...")

        # Method 1: Safe age calculation using year extraction
        merged['admit_year'] = merged['admittime'].dt.year
        merged['birth_year'] = merged['dob'].dt.year

        # Calculate age (more stable than date difference)
        merged['age'] = merged['admit_year'] - merged['birth_year']

        # Filter out unreasonable ages
        merged = merged[(merged['age'] >= 0) & (merged['age'] <= 120)]

        # Hospital length of stay (safe calculation)
        merged['los_hospital'] = (merged['dischtime'] - merged['admittime']).dt.days

        # Filter out negative or unreasonable LOS
        merged = merged[(merged['los_hospital'] >= 0) & (merged['los_hospital'] <= 365)]

        # ICU length of stay (already calculated in icustays table)
        # Just ensure it's reasonable
        merged = merged[(merged['los_icu'] >= 0) & (merged['los_icu'] <= 90)]

        # Encode categorical variables
        print("Encoding categorical variables...")

        # Handle missing values in categorical columns first
        for cat_col in ['ethnicity', 'admission_type', 'insurance']:
            if cat_col in merged.columns:
                # Fill NaN with 'Unknown'
                merged[cat_col] = merged[cat_col].fillna('Unknown')

        # Encode with error handling
        try:
            merged['ethnicity_num'] = LabelEncoder().fit_transform(merged['ethnicity'].astype(str))
            merged['admission_type_num'] = LabelEncoder().fit_transform(merged['admission_type'].astype(str))
            merged['insurance_num'] = LabelEncoder().fit_transform(merged['insurance'].astype(str))
        except Exception as e:
            print(f"Warning: Error in encoding categorical variables: {e}")
            # Use simple numeric encoding as fallback
            for i, col in enumerate(['ethnicity', 'admission_type', 'insurance']):
                if col in merged.columns:
                    merged[f'{col}_num'] = pd.factorize(merged[col])[0]

        print(f"Final merged dataset shape: {merged.shape}")

        return merged

    def create_mimic_dataloaders(self, batch_size=32, missing_rate=0.3,
                                 target_condition="mortality", seed=42,
                                 test_size=0.2, val_size=0.15):
        """
        Create MIMIC-III dataloaders with missing data simulation
        """
        # Preprocess data
        features, labels, subject_ids, hadm_ids, feature_columns = self.preprocess_data(
            target_condition=target_condition
        )

        print(f"Features shape before encoding: {features.shape}")
        print(f"Feature columns: {feature_columns}")

        # Ensure all features are numeric before scaling
        features_numeric = self._ensure_numeric_features(features, feature_columns)

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_numeric)

        # Create missing mask with realistic patterns
        mask = self._create_realistic_missing_mask(features_normalized, labels, missing_rate, seed)

        # Create dataset
        dataset = MIMICDataset(features_normalized, labels, mask, subject_ids, hadm_ids)

        # Split dataset - FIXED: Extract labels for stratification
        labels_array = np.array([item["y"].item() for item in dataset])

        # Convert to indices for splitting
        indices = list(range(len(dataset)))

        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices, test_size=test_size + val_size, random_state=seed, stratify=labels_array
        )

        # Second split: val vs test
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_size / (test_size + val_size),
            random_state=seed,
            stratify=labels_array[temp_indices] if len(temp_indices) > 0 else None
        )

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        return train_loader, val_loader, test_loader, features_numeric.shape[1], len(np.unique(labels))
    def _ensure_numeric_features(self, features, feature_columns):
        """Convert categorical features to numeric representation"""
        features_df = pd.DataFrame(features, columns=feature_columns)

        # Identify categorical columns (non-numeric)
        categorical_cols = []
        numeric_cols = []

        for col in feature_columns:
            # Check if column contains non-numeric data
            if features_df[col].dtype == 'object' or any(isinstance(x, str) for x in features_df[col].head(10)):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

        print(f"Categorical columns to encode: {categorical_cols}")
        print(f"Numeric columns: {numeric_cols}")

        # Encode categorical columns
        if categorical_cols:
            from sklearn.preprocessing import LabelEncoder

            for col in categorical_cols:
                le = LabelEncoder()
                # Handle NaN values
                features_df[col] = features_df[col].fillna('Unknown')
                features_df[col] = le.fit_transform(features_df[col].astype(str))
                print(f"Encoded {col} with {len(le.classes_)} classes")

        # Convert back to numpy array
        features_numeric = features_df.values.astype(np.float32)

        return features_numeric
    def _create_realistic_missing_mask(self, features, labels, missing_rate, seed):
        """
        Create realistic missing data patterns for clinical data
        """
        np.random.seed(seed)
        n_samples, n_features = features.shape
        mask = np.ones((n_samples, n_features))

        # Different missingness patterns for different feature types
        for i in range(n_samples):
            missing_probs = np.full(n_features, missing_rate)

            # Increase missingness for extreme values (MNAR)
            for j in range(n_features):
                if abs(features[i, j]) > 2.0:
                    missing_probs[j] = min(missing_rate + 0.4, 0.9)
                elif abs(features[i, j]) > 1.5:
                    missing_probs[j] = min(missing_rate + 0.2, 0.7)

            # Increase missingness based on outcome (MAR)
            if labels[i] == 1:  # Positive cases (e.g., mortality)
                # Lab results and vital signs more likely to be missing for sick patients
                if n_features > 2:  # Assuming first few features are demographics
                    missing_probs[2:] = min(missing_rate + 0.3, 0.8)

            # Add random variation
            missing_probs = missing_probs * np.random.uniform(0.9, 1.1, n_features)
            missing_probs = np.clip(missing_probs, 0.1, 0.9)

            mask[i] = (np.random.rand(n_features) > missing_probs).astype(float)

        return mask


# Utility function for backward compatibility
def get_mimic_dataloaders(data_dir, batch_size=32, missing_rate=0.3,
                          target_condition="mortality", seed=42):
    """
    Convenience function to get MIMIC dataloaders
    """
    loader = MIMICDataLoader(data_dir)
    return loader.create_mimic_dataloaders(
        batch_size=batch_size,
        missing_rate=missing_rate,
        target_condition=target_condition,
        seed=seed
    )