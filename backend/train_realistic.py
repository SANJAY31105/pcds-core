"""
Train PCDS ML Models on Realistic Data
Uses the generated 1M+ realistic network events
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import time
import os

def load_and_prepare_data():
    print("📂 Loading realistic training data...")
    df = pd.read_csv("data/pcds_realistic_training.csv")
    print(f"   Loaded {len(df):,} records")
    
    # Feature engineering
    print("🔧 Engineering features...")
    
    # Encode categorical features
    le_protocol = LabelEncoder()
    le_process = LabelEncoder()
    le_status = LabelEncoder()
    
    df['protocol_enc'] = le_protocol.fit_transform(df['protocol'])
    df['process_enc'] = le_process.fit_transform(df['process_name'])
    df['status_enc'] = le_status.fit_transform(df['status'])
    
    # Port-based features
    high_risk_ports = {22, 23, 3389, 445, 135, 139, 4444, 1337}
    lateral_ports = {22, 135, 139, 445, 3389, 5985, 5986}
    df['is_high_risk_port'] = df['dst_port'].isin(high_risk_ports).astype(int)
    df['is_lateral_port'] = df['dst_port'].isin(lateral_ports).astype(int)
    
    # Byte ratio
    df['byte_ratio'] = df['bytes_sent'] / (df['bytes_recv'] + 1)
    
    # Internal vs external
    df['is_internal_dst'] = df['dst_ip'].str.startswith(('192.168.', '10.0.', '127.')).astype(int)
    
    # Suspicious process
    suspicious_procs = ['powershell.exe', 'cmd.exe', 'wmic.exe', 'psexec.exe', 
                        'mimikatz.exe', 'nc.exe', 'certutil.exe', 'bitsadmin.exe',
                        'rundll32.exe', 'regsvr32.exe']
    df['is_suspicious_proc'] = df['process_name'].isin(suspicious_procs).astype(int)
    
    # Large transfer
    df['is_large_transfer'] = (df['bytes_sent'] > 100000).astype(int)
    
    # Features for training
    feature_cols = [
        'dst_port', 'bytes_sent', 'bytes_recv', 'packets', 'duration_ms',
        'protocol_enc', 'process_enc', 'status_enc',
        'is_high_risk_port', 'is_lateral_port', 'byte_ratio',
        'is_internal_dst', 'is_suspicious_proc', 'is_large_transfer'
    ]
    
    X = df[feature_cols].values
    
    # Binary classification: benign vs attack
    y_binary = (df['label'] != 'benign').astype(int)
    
    # Multi-class classification
    le_label = LabelEncoder()
    y_multi = le_label.fit_transform(df['label'])
    
    return X, y_binary, y_multi, le_label, feature_cols

def train_models(X, y_binary, y_multi, le_label, feature_cols):
    print("\n🧠 Training ML Models...")
    
    # Split data
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    _, _, y_train_multi, y_test_multi = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Isolation Forest (Anomaly Detection)
    print("\n1️⃣ Training Isolation Forest (Anomaly Detection)...")
    start = time.time()
    iso_forest = IsolationForest(n_estimators=100, contamination=0.15, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_scaled)
    iso_pred = (iso_forest.predict(X_test_scaled) == -1).astype(int)
    iso_acc = accuracy_score(y_test_bin, iso_pred)
    print(f"   Accuracy: {iso_acc*100:.2f}% | Time: {time.time()-start:.1f}s")
    results['isolation_forest'] = iso_acc
    
    # 2. Random Forest (Multi-class)
    print("\n2️⃣ Training Random Forest (Multi-class Classification)...")
    start = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train_multi)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test_multi, rf_pred)
    print(f"   Accuracy: {rf_acc*100:.2f}% | Time: {time.time()-start:.1f}s")
    print("\n   Classification Report:")
    print(classification_report(y_test_multi, rf_pred, target_names=le_label.classes_))
    results['random_forest'] = rf_acc
    
    # 3. Gradient Boosting (Binary)
    print("\n3️⃣ Training Gradient Boosting (Binary Detection)...")
    start = time.time()
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    gb_model.fit(X_train_scaled, y_train_bin)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test_bin, gb_pred)
    print(f"   Accuracy: {gb_acc*100:.2f}% | Time: {time.time()-start:.1f}s")
    results['gradient_boosting'] = gb_acc
    
    # Save models
    print("\n💾 Saving models...")
    os.makedirs("ml/models/trained", exist_ok=True)
    
    joblib.dump(iso_forest, "ml/models/trained/isolation_forest_realistic.joblib")
    joblib.dump(rf_model, "ml/models/trained/random_forest_multiclass.joblib")
    joblib.dump(gb_model, "ml/models/trained/gradient_boosting_binary.joblib")
    joblib.dump(scaler, "ml/models/trained/feature_scaler.joblib")
    joblib.dump(le_label, "ml/models/trained/label_encoder.joblib")
    
    # Save feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top Feature Importance:")
    print(importance.head(10).to_string(index=False))
    
    return results

def main():
    print("=" * 60)
    print("🚀 PCDS ML TRAINING ON REALISTIC DATA")
    print("=" * 60)
    
    start_total = time.time()
    
    X, y_binary, y_multi, le_label, feature_cols = load_and_prepare_data()
    results = train_models(X, y_binary, y_multi, le_label, feature_cols)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📈 Final Results:")
    for model, acc in results.items():
        print(f"   {model}: {acc*100:.2f}%")
    print(f"\n⏱️ Total time: {time.time()-start_total:.1f} seconds")
    print("\n💾 Models saved to: ml/models/trained/")

if __name__ == "__main__":
    main()
