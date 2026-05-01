# Required: pip install tensorflow tensorflow-hub librosa soundfile numpy scikit-learn

import os
import json
import random
import pathlib

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import sklearn.preprocessing

DATA_DIR = "data/raw"
TEST_DIR = "data/test"
SAMPLE_RATE = 16000  # YAMNet requires 16kHz not 22050
CLASSES = ["healthy", "knocking", "misfiring", 
           "tappet", "rotational_imbalance", "battery_fault"]
AUGMENTATION_FACTOR = 4
EMBEDDINGS_SAVE_PATH = "data/processed/embeddings.npz"


def load_yamnet_model():
    """
    Load YAMNet from TensorFlow Hub using the URL https://tfhub.dev/google/yamnet/1.
    Returns the loaded model.
    """
    try:
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet loaded successfully")
        return model
    except Exception as e:
        print(f"Failed to load YAMNet model. Error: {e}")
        return None


def resample_audio(audio, original_sr, target_sr=16000):
    """
    Resample audio array from original_sr to target_sr using librosa.resample.
    Return resampled audio as float32 numpy array. 
    Ensure values are clipped between -1 and 1.
    """
    resampled = librosa.resample(y=audio, orig_sr=original_sr, target_sr=target_sr)
    resampled = np.clip(resampled, -1.0, 1.0)
    return resampled.astype(np.float32)


def extract_yamnet_embedding(yamnet_model, audio, sr):
    """
    Extract features from YAMNet.
    Resample audio to 16000Hz if not already. Convert to float32 and ensure mono.
    Pass through yamnet_model to get scores, embeddings, spectrogram.
    YAMNet returns embeddings of shape (num_frames, 1024). Take the mean across 
    frames to get a single (1024,) vector. Return the 1024-dimensional embedding.
    """
    try:
        if sr != 16000:
            audio = resample_audio(audio, sr, target_sr=16000)
            
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        # Convert to float32 just in case
        audio = audio.astype(np.float32)
        
        # Passing mono audio array to YAMNet model
        scores, embeddings, spectrogram = yamnet_model(audio)
        
        # Take the mean across frames 
        embedding_mean = np.mean(embeddings.numpy(), axis=0)
        return embedding_mean
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def augment_audio(audio, sr):
    """
    Apply one random augmentation from the list and return the augmented audio:
    - Time stretch: random rate between 0.85 and 1.15
    - Pitch shift: random semitones between -2 and +2
    - Add gaussian noise: random SNR between 15 and 25 dB
    - Time shift: shift audio left or right by up to 10% of length
    """
    aug_choice = random.choice(['time_stretch', 'pitch_shift', 'noise', 'time_shift'])
    
    if aug_choice == 'time_stretch':
        rate = random.uniform(0.85, 1.15)
        return librosa.effects.time_stretch(y=audio, rate=rate)
        
    elif aug_choice == 'pitch_shift':
        steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)
        
    elif aug_choice == 'noise':
        snr_db = random.uniform(15, 25)
        # Calculate signal and target noise power
        sig_power = np.mean(audio**2) + 1e-10
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
        
    elif aug_choice == 'time_shift':
        shift_pct = random.uniform(-0.1, 0.1)
        shift_amount = int(len(audio) * shift_pct)
        return np.roll(audio, shift_amount)


def load_dataset(yamnet_model, augment=True, data_dir=DATA_DIR):
    """
    Loop through each class folder in data_dir. For each .wav file, 
    extract YAMNet embedding. If augment=True, generate augmented versions.
    Collect into X and y arrays and print progress.
    """
    label_map = {class_name: idx for idx, class_name in enumerate(CLASSES)}
    X = []
    y = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
            
        wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        count_for_class = 0
        
        for i, wav_file in enumerate(wav_files):
            file_path = os.path.join(class_dir, wav_file)
            
            # Show progress
            if i % 10 == 0 or i == len(wav_files) - 1:
                print(f"Processing {class_name}: {i+1}/{len(wav_files)} files")
                
            try:
                audio, sr = librosa.load(file_path, sr=None)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
            # Original file embedding
            embedding = extract_yamnet_embedding(yamnet_model, audio, sr)
            if embedding is not None:
                X.append(embedding)
                y.append(label_map[class_name])
                count_for_class += 1
                
                # Augmentation
                if augment:
                    for _ in range(AUGMENTATION_FACTOR):
                        aug_audio = augment_audio(audio, sr)
                        aug_embedding = extract_yamnet_embedding(yamnet_model, aug_audio, sr)
                        if aug_embedding is not None:
                            X.append(aug_embedding)
                            y.append(label_map[class_name])
                            count_for_class += 1
                            
        print(f"Finished class '{class_name}': generated {count_for_class} total samples.")
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map


def load_test_set(yamnet_model):
    """
    Load testing dataset. Reads from TEST_DIR, no augmentation applied.
    """
    return load_dataset(yamnet_model, augment=False, data_dir=TEST_DIR)


def save_embeddings(X, y, X_test, y_test, label_map, path):
    """
    Save embeddings to an npz file and the label map to a json file.
    Prints confirmation with shapes.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    np.savez(path, X=X, y=y, X_test=X_test, y_test=y_test)
    
    json_path = str(pathlib.Path(path).with_suffix('.json'))
    with open(json_path, 'w') as f:
        json.dump(label_map, f, indent=4)
        
    print(f"Arrays and label map saved to {path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


if __name__ == "__main__":
    print("Initializing YAMNet model...")
    model = load_yamnet_model()
    
    if model is not None:
        print("\nLoading and processing training dataset...")
        X, y, label_map = load_dataset(model, augment=True)
        
        print("\nLoading and processing test dataset...")
        X_test, y_test, label_map_test = load_test_set(model)
        
        print("\n--- Array Shapes ---")
        print(f"X: {X.shape}")
        print(f"y: {y.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")
        
        print("\n--- Class Distribution (Training Set) ---")
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            inv_map = {v: k for k, v in label_map.items()}
            for u, c in zip(unique, counts):
                print(f"{inv_map[u]}: {c} samples")
        else:
            print("Training set is empty.")
            
        print("\nSaving dataset...")
        save_embeddings(X, y, X_test, y_test, label_map, EMBEDDINGS_SAVE_PATH)
        print("Dataset pipeline complete")
    else:
        print("Pipeline aborted due to YAMNet loading failure.")
