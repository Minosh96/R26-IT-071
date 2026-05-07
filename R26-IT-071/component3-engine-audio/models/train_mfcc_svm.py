import os
import time
import json
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR = "data/raw"
TEST_DIR = "data/test"
MODEL_SAVE_PATH = "models/saved/mfcc_svm_model.joblib"
SCALER_SAVE_PATH = "models/saved/mfcc_scaler.joblib"
SAMPLE_RATE = 22050
N_MFCC = 40
CLASSES = ["healthy", "knocking", "misfiring",
           "tappet", "rotational_imbalance", "battery_fault"]
RANDOM_SEED = 42

def extract_mfcc_features(file_path_or_audio, sr=SAMPLE_RATE):
    try:
        if isinstance(file_path_or_audio, str):
            audio, _ = librosa.load(file_path_or_audio, sr=sr)
        else:
            audio = file_path_or_audio
            
        if len(audio) == 0:
            return None
            
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        feature_vector = np.concatenate((mfcc_mean, mfcc_std))
        return feature_vector
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def augment_audio(audio, sr):
    aug_type = np.random.choice(['time_stretch', 'pitch_shift', 'noise', 'time_shift'])
    
    if aug_type == 'time_stretch':
        rate = np.random.choice([0.9, 1.1])
        return librosa.effects.time_stretch(y=audio, rate=rate)
    elif aug_type == 'pitch_shift':
        steps = np.random.choice([-2, 2])
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)
    elif aug_type == 'noise':
        # SNR 20dB
        signal_power = np.mean(audio**2)
        if signal_power == 0:
            return audio
        noise_power = signal_power / (10 ** (20 / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    elif aug_type == 'time_shift':
        shift_max = int(len(audio) * 0.1)
        if shift_max == 0:
            return audio
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(audio, shift)
    
    return audio

def load_dataset(augment=True):
    X = []
    y = []
    label_map = {cls: i for i, cls in enumerate(CLASSES)}
    
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
            
        print(f"Loading {cls}...")
        files = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
        for f in files:
            file_path = os.path.join(cls_dir, f)
            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Original
                feat = extract_mfcc_features(audio, sr)
                if feat is not None:
                    X.append(feat)
                    y.append(label_map[cls])
                
                # Augmented
                if augment:
                    for _ in range(4):
                        aug_audio = augment_audio(audio, sr)
                        aug_feat = extract_mfcc_features(aug_audio, sr)
                        if aug_feat is not None:
                            X.append(aug_feat)
                            y.append(label_map[cls])
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                
    return np.array(X), np.array(y), label_map

def load_test_set(label_map):
    X_test = []
    y_test = []
    
    for cls in CLASSES:
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
            
        files = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
        for f in files:
            file_path = os.path.join(cls_dir, f)
            feat = extract_mfcc_features(file_path, SAMPLE_RATE)
            if feat is not None:
                X_test.append(feat)
                y_test.append(label_map[cls])
                
    return np.array(X_test), np.array(y_test)

if __name__ == "__main__":
    # Step 1: Load dataset with augmentation
    print("Step 1: Loading dataset with augmentation...")
    X, y, label_map = load_dataset(augment=True)
    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")
    
    if len(X) == 0:
        print("Error: No data loaded. Check if data directories exist and have .wav files.")
        exit(1)
        
    # Step 2: Split 70/15/15 train/val/test with stratify=y
    print("\nStep 2: Splitting dataset into 70/15/15 train/val/test...")
    # First split off 30% for val/test combined
    X_temp, X_test_split, y_temp, y_test_split = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    # Then split the 30% into two equal 15% halves for val and test
    X_val, X_test_split, y_val, y_test_split = train_test_split(
        X_test_split, y_test_split, test_size=0.5, stratify=y_test_split, random_state=RANDOM_SEED
    )
    X_train, y_train = X_temp, y_temp
    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Val shapes: {X_val.shape}, {y_val.shape}")
    print(f"Test split shapes: {X_test_split.shape}, {y_test_split.shape}")
    
    # Step 3: Scale with StandardScaler fit on train only. Save scaler.
    print("\nStep 3: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_split_scaled = scaler.transform(X_test_split)
    
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    # Step 4: Train these three classifiers and compare all on validation set
    print("\nStep 4: Training and comparing classifiers...")
    models = {
        "SVM (RBF)": SVC(C=10, gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    }
    
    results = []
    best_f1 = 0
    best_acc = 0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_val_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average='macro')
        
        results.append({
            "name": name,
            "acc": acc,
            "f1": f1,
            "time": train_time
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_model_name = name
            best_model = model
            
    # Step 5: Print comparison table
    print("\nStep 5: Model Comparison Table:")
    print("─────────────────────────────────────────────────────")
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<10} {'Time(s)'}")
    print("─────────────────────────────────────────────────────")
    for r in results:
        print(f"{r['name']:<20} {r['acc']:.2f}         {r['f1']:.2f}       {r['time']:.1f}")
    print("─────────────────────────────────────────────────────")
    
    # Step 6: Take best MFCC model. Print full classification report per class.
    print(f"\nStep 6: Best MFCC Model is {best_model_name}. Full Classification Report:")
    y_val_pred_best = best_model.predict(X_val_scaled)
    inv_label_map = {v: k for k, v in label_map.items()}
    # Sort target names by label index 0 to N-1
    target_names = [inv_label_map[i] for i in range(len(CLASSES))]
    print(classification_report(y_val, y_val_pred_best, target_names=target_names))
    
    # Step 7: Evaluate on real test set. Print test accuracy and F1.
    print("\nStep 7: Evaluating on real test set...")
    X_real_test, y_real_test = load_test_set(label_map)
    if len(X_real_test) > 0:
        X_real_test_scaled = scaler.transform(X_real_test)
        y_real_test_pred = best_model.predict(X_real_test_scaled)
        real_test_acc = accuracy_score(y_real_test, y_real_test_pred)
        real_test_f1 = f1_score(y_real_test, y_real_test_pred, average='macro')
        print(f"Real Test Accuracy: {real_test_acc:.2f}")
        print(f"Real Test Macro F1: {real_test_f1:.2f}")
    else:
        print("No real test data found in TEST_DIR. Evaluating on the earlier test split...")
        y_test_split_pred = best_model.predict(X_test_split_scaled)
        split_test_acc = accuracy_score(y_test_split, y_test_split_pred)
        split_test_f1 = f1_score(y_test_split, y_test_split_pred, average='macro')
        print(f"Split Test Accuracy: {split_test_acc:.2f}")
        print(f"Split Test Macro F1: {split_test_f1:.2f}")
        
    # Step 8: Save best MFCC model.
    print("\nStep 8: Saving best MFCC model...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Step 9: Print final comparison against YAMNet+SVM
    print("\nStep 9: Final Comparison")
    
    # YAMNet + SVM results from train_svm.py
    yamnet_val_accuracy = 0.9006
    yamnet_val_f1 = 0.8735
    
    best_mfcc_accuracy = best_acc
    best_mfcc_f1 = best_f1
    
    accuracy_diff = abs(yamnet_val_accuracy - best_mfcc_accuracy) * 100
    
    if yamnet_val_f1 >= best_mfcc_f1:
        winner = "YAMNet + SVM"
        loser = "MFCC + SVM"
        reason = f"YAMNet pre-trained audio embeddings provide richer feature representation than hand-crafted MFCC features, resulting in {accuracy_diff:.1f}% higher accuracy."
    else:
        winner = "MFCC + SVM"
        loser = "YAMNet + SVM"
        reason = f"Hand-crafted MFCC features outperformed YAMNet embeddings for this dataset, achieving {accuracy_diff:.1f}% higher accuracy."

    final_output = f"""════════════════════════════════════════════════════
EXPERIMENT COMPARISON RESULTS
════════════════════════════════════════════════════
Method                    Val Accuracy    Val F1
────────────────────────────────────────────────
Method 1: YAMNet + SVM        {yamnet_val_accuracy * 100:.0f}%          {yamnet_val_f1:.2f}   ← current
Method 2: MFCC + SVM          {best_mfcc_accuracy * 100:.0f}%          {best_mfcc_f1:.2f}   ← experiment
────────────────────────────────────────────────
Winner: {winner}
Reason: {reason}
════════════════════════════════════════════════════"""
    print(final_output)
