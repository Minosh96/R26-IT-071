import os
import random
import numpy as np
import librosa
import soundfile as sf

INPUT_FOLDER = "data/raw/battery_fault/"
TARGET_COUNT = 25
SAMPLE_RATE = 22050

def augment_audio(audio, sr, method_idx):
    if method_idx == 0:
        # time stretch 0.9x
        return librosa.effects.time_stretch(y=audio, rate=0.9)
    elif method_idx == 1:
        # time stretch 1.1x
        return librosa.effects.time_stretch(y=audio, rate=1.1)
    elif method_idx == 2:
        # pitch shift +1 semitone
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=1)
    elif method_idx == 3:
        # pitch shift -1 semitone
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=-1)
    elif method_idx == 4:
        # add gaussian noise at SNR 20dB
        signal_power = np.mean(audio**2)
        if signal_power > 0:
            noise_power = signal_power / (10 ** (20 / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
            return audio + noise
        return audio
    elif method_idx == 5:
        # time stretch 0.95x combined with pitch shift +0.5 semitone
        a1 = librosa.effects.time_stretch(y=audio, rate=0.95)
        a2 = librosa.effects.pitch_shift(y=a1, sr=sr, n_steps=0.5)
        return a2
    return audio

def generate_augmented_data():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Directory {INPUT_FOLDER} does not exist.")
        return

    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.wav')]
    original_files = [f for f in all_files if not f.startswith("battery_aug_")]
    
    if len(original_files) == 0:
        print("No original battery fault files found to augment.")
        return
        
    current_count = len(all_files)
    needed = max(0, TARGET_COUNT - current_count)
    if needed == 0:
        print(f"Folder already has {current_count} files, no generation needed.")
        return
        
    # figure out start index for naming incrementally
    aug_files = [f for f in all_files if f.startswith("battery_aug_")]
    start_idx = len(aug_files) + 1
    
    aug_count = 0
    method_idx = 0
    
    while aug_count < needed:
        src_filename = random.choice(original_files)
        src_path = os.path.join(INPUT_FOLDER, src_filename)
        
        try:
            audio, sr = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
            aug_audio = augment_audio(audio, sr, method_idx)
            
            method_idx = (method_idx + 1) % 6
            
            out_name = f"battery_aug_{start_idx:03d}.wav"
            out_path = os.path.join(INPUT_FOLDER, out_name)
            sf.write(out_path, aug_audio, sr)
            
            start_idx += 1
            aug_count += 1
        except Exception as e:
            # Silently skip on error
            continue
            
    final_count = len([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.wav')])
    print(f"Total .wav files in {INPUT_FOLDER}: {final_count}")

if __name__ == '__main__':
    generate_augmented_data()
