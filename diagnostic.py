"""
COVID-19 Model Diagnostic Tool
Identifies data and model training issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

def diagnose_data():
    """Run comprehensive data diagnostics"""
    print("=" * 80)
    print("COVID-19 MODEL DIAGNOSTIC REPORT")
    print("=" * 80)
    
    try:
        # Load dataset
        df = pd.read_csv('Cleaned-Data.csv')
        print("\n✓ Dataset loaded successfully")
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nColumn Names:\n{df.columns.tolist()}")
        
        # Check for required columns
        feature_columns = [
            'Fever',
            'Tiredness',
            'Dry-Cough',
            'Difficulty-in-Breathing',
            'Sore-Throat'
        ]
        
        print("\n" + "=" * 80)
        print("1. FEATURE COLUMNS CHECK")
        print("=" * 80)
        
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ MISSING COLUMNS: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        print("✓ All required feature columns present")
        
        # Check target variable
        print("\n" + "=" * 80)
        print("2. TARGET VARIABLE CHECK")
        print("=" * 80)
        
        if 'Severity' in df.columns:
            target_col = 'Severity'
            print(f"✓ Target column found: '{target_col}'")
        else:
            target_col = df.columns[-1]
            print(f"⚠ 'Severity' not found, using last column: '{target_col}'")
        
        y = df[target_col]
        
        print(f"\nTarget Variable Distribution:")
        print(y.value_counts().sort_index())
        print(f"\nClass Balance:")
        for val, count in y.value_counts().sort_index().items():
            pct = (count / len(y)) * 100
            print(f"  Class {val}: {count} samples ({pct:.1f}%)")
        
        # Check for data quality issues
        print("\n" + "=" * 80)
        print("3. DATA QUALITY CHECK")
        print("=" * 80)
        
        print(f"\nMissing Values:")
        missing = df[feature_columns + [target_col]].isnull().sum()
        print(missing)
        
        if missing.sum() > 0:
            print("⚠ WARNING: Missing values detected!")
        else:
            print("✓ No missing values")
        
        print(f"\nData Types:")
        print(df[feature_columns + [target_col]].dtypes)
        
        print(f"\nFeature Statistics:")
        print(df[feature_columns].describe())
        
        # Check feature values
        print("\n" + "=" * 80)
        print("4. FEATURE VALUE RANGE CHECK")
        print("=" * 80)
        
        for col in feature_columns:
            unique_vals = df[col].unique()
            print(f"\n{col}:")
            print(f"  Unique values: {sorted(unique_vals)}")
            print(f"  Min: {df[col].min()}, Max: {df[col].max()}")
        
        # Prepare data
        X = df[feature_columns]
        
        print("\n" + "=" * 80)
        print("5. DATA PREPARATION")
        print("=" * 80)
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Train set size: {X_train.shape[0]}")
        print(f"✓ Test set size: {X_test.shape[0]}")
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n" + "=" * 80)
        print("6. MODEL TRAINING")
        print("=" * 80)
        
        # Train model
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        
        model.fit(X_train_scaled, y_train)
        print("✓ Model training completed")
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Get probabilities for ROC-AUC
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        print("\n" + "=" * 80)
        print("7. MODEL PERFORMANCE")
        print("=" * 80)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
        print(f"Testing Accuracy: {test_acc*100:.2f}%")
        
        # Check for overfitting/underfitting
        if train_acc < 0.60 and test_acc < 0.60:
            print("⚠ UNDERFITTING: Model performs poorly on both train and test sets")
        elif abs(train_acc - test_acc) > 0.15:
            print("⚠ OVERFITTING: Large gap between train and test accuracy")
        else:
            print("✓ Model shows balanced performance")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        
        # ROC-AUC (if binary classification)
        if len(np.unique(y)) == 2:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                print(f"\nROC-AUC Score: {roc_auc:.4f}")
            except:
                pass
        
        # Model coefficients
        print(f"\n" + "=" * 80)
        print("8. MODEL COEFFICIENTS (Feature Importance)")
        print("=" * 80)
        
        for feature, coef in zip(feature_columns, model.coef_[0]):
            print(f"{feature}: {coef:.4f}")
        
        print(f"\nIntercept: {model.intercept_[0]:.4f}")
        
        # Recommendations
        print(f"\n" + "=" * 80)
        print("9. RECOMMENDATIONS")
        print("=" * 80)
        
        issues = []
        
        if test_acc < 0.60:
            issues.append("❌ Very low test accuracy - Model needs improvement")
            issues.append("   → Check if features are properly correlated with target")
            issues.append("   → Verify data quality and labels")
            issues.append("   → Consider feature engineering or different algorithms")
        
        if len(np.unique(y)) < 2:
            issues.append("❌ Target variable has only one class - Check data!")
        
        class_counts = y.value_counts()
        if class_counts.min() < 10:
            issues.append("⚠ Imbalanced classes - Consider oversampling minority class")
        
        if missing.sum() > 0:
            issues.append("⚠ Missing values detected - May impact model")
        
        if len(issues) == 0:
            print("✓ No critical issues detected")
            print("✓ Model appears to be functioning normally")
        else:
            for issue in issues:
                print(issue)
        
        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease check:")
        print("1. Is 'Cleaned-Data.csv' in the current directory?")
        print("2. Are column names exactly as specified?")
        print("3. Is the data in the correct format?")

if __name__ == "__main__":
    diagnose_data()
