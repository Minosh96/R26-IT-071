import os
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

HEALTHY_FOLDER = "data/raw/healthy"
OUTPUT_BASE = "data/raw"
SAMPLES_PER_CLASS = 25
SAMPLE_RATE = 22050

def apply_random_augmentation(audio, sr):
    """
    Applies one random augmentation: time stretch, pitch shift, or add gaussian noise.
    """
    choice = random.choice(['time', 'pitch', 'noise'])
    if choice == 'time':
        rate = random.choice([0.9, 1.1])
        # Time stretching
        audio = librosa.effects.time_stretch(y=audio, rate=rate)
    elif choice == 'pitch':
        steps = random.choice([-1, 1])
        # Pitch shifting (±1 semitone)
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)
    elif choice == 'noise':
        # Add gaussian noise at approx SNR 25dB
        signal_power = np.mean(audio**2)
        if signal_power > 0:
            noise_power = signal_power / (10 ** (25 / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
            audio = audio + noise
        
    return audio

def simulate_knocking(audio, sr, real_knocking_files=None):
    """
    Simulates knocking by either mathematically injecting bursts or extracting
    real percussive knocking transients from existing real samples to overlay.
    """
    if real_knocking_files:
        try:
            real_file = random.choice(real_knocking_files)
            real_audio, _ = librosa.load(real_file, sr=sr, mono=True)
            
            if len(real_audio) > len(audio):
                start = random.randint(0, len(real_audio) - len(audio))
                real_audio = real_audio[start:start+len(audio)]
            elif len(real_audio) < len(audio):
                real_audio = np.pad(real_audio, (0, len(audio) - len(real_audio)), mode='wrap')
                
            _, percussive = librosa.effects.hpss(real_audio, margin=1.2)
            
            if np.max(np.abs(percussive)) > 0:
                percussive = percussive / np.max(np.abs(percussive)) * np.max(np.abs(audio))
                
            return audio + 0.5 * percussive
        except Exception:
            pass

    rate = random.uniform(6, 10)
    interval = 1.0 / rate
    
    knocking = np.zeros_like(audio)
    t = 0.0
    while t < len(audio) / sr:
        jitter = interval * random.uniform(-0.2, 0.2)
        impulse_time = t + jitter
        
        if 0 < impulse_time < len(audio) / sr:
            impulse_duration = random.uniform(0.005, 0.010)
            impulse_length = int(impulse_duration * sr)
            
            noise = np.random.randn(impulse_length)
            decay = np.exp(-np.linspace(0, 5, impulse_length))
            impulse = noise * decay
            
            start_idx = int(impulse_time * sr)
            end_idx = min(start_idx + impulse_length, len(knocking))
            
            knocking[start_idx:end_idx] += impulse[:end_idx-start_idx]
            
        t += interval
        
    if np.max(np.abs(knocking)) > 0:
        knocking = knocking / np.max(np.abs(knocking)) * np.max(np.abs(audio))
        
    mixed = audio + 0.3 * knocking
    return mixed

def simulate_misfiring(audio, sr):
    """
    Randomly silence or heavily attenuate short segments (30-80ms) of the audio at random intervals,
    with 15% probability per 100ms window.
    Adds slight irregular amplitude variation across the whole recording.
    """
    misfiring = audio.copy()
    window_len = int(0.100 * sr)
    
    for start_idx in range(0, len(audio), window_len):
        if random.random() < 0.15:
            seg_len = int(random.uniform(0.030, 0.080) * sr)
            end_idx = min(start_idx + seg_len, len(audio))
            misfiring[start_idx:end_idx] *= random.uniform(0.0, 0.2)
            
    num_points = max(2, int(len(audio) / sr * 5))
    mod_points = np.random.uniform(0.8, 1.2, num_points)
    mod_curve = np.interp(np.linspace(0, 1, len(audio)), np.linspace(0, 1, num_points), mod_points)
    
    misfiring *= mod_curve
    return misfiring

def simulate_tappet(audio, sr):
    """
    Inject rapid light high-frequency ticking at 20-30 ticks per second.
    Each tick is 2-5ms long. Mixed additively on top of original audio at 0.15 intensity.
    """
    rate = random.uniform(20, 30)
    interval = 1.0 / rate
    
    tappet = np.zeros_like(audio)
    t = 0.0
    while t < len(audio) / sr:
        impulse_duration = random.uniform(0.002, 0.005)
        impulse_length = int(impulse_duration * sr)
        
        tick = np.random.randn(impulse_length)
        
        start_idx = int(t * sr)
        if start_idx < len(tappet):
            end_idx = min(start_idx + impulse_length, len(tappet))
            tappet[start_idx:end_idx] += tick[:end_idx-start_idx]
        
        t += interval

    if np.max(np.abs(tappet)) > 0:
        tappet = tappet / np.max(np.abs(tappet)) * np.max(np.abs(audio))
        
    return audio + 0.15 * tappet

def simulate_rotational_imbalance(audio, sr):
    """
    Apply amplitude modulation using a sine wave at 3-6 Hz with modulation depth 0.35.
    Creates rhythmic wobbling volume variation.
    """
    freq = random.uniform(3, 6)
    t = np.arange(len(audio)) / sr
    modulation = 1.0 - (0.35 * 0.5 * (1 - np.sin(2 * np.pi * freq * t)))
    return audio * modulation

def generate_fault_dataset():
    """
    Main function to load healthy files, generate fault versions, mix augmentations, and save.
    """
    healthy_files = list(Path(HEALTHY_FOLDER).glob("*.wav"))
    if not healthy_files:
        print(f"No healthy files found in {HEALTHY_FOLDER}.")
        return

    knocking_folder = Path(OUTPUT_BASE) / "knocking"
    real_knocking_count = 7
    existing_real = []
    if knocking_folder.exists():
        existing_real = [f for f in knocking_folder.glob("*.wav") if "synthetic" not in f.name]
        if len(existing_real) > 0:
            real_knocking_count = len(existing_real)
            
    knocking_needed = max(0, SAMPLES_PER_CLASS - real_knocking_count)

    faults = ["misfiring", "tappet", "rotational_imbalance", "knocking"]
    for fault in faults:
        (Path(OUTPUT_BASE) / fault).mkdir(parents=True, exist_ok=True)
        
    counts = {fault: 0 for fault in faults}
    
    attempts = 0
    max_attempts = SAMPLES_PER_CLASS * 10
    
    while (counts["misfiring"] < SAMPLES_PER_CLASS or 
           counts["tappet"] < SAMPLES_PER_CLASS or 
           counts["rotational_imbalance"] < SAMPLES_PER_CLASS or 
           counts["knocking"] < knocking_needed):

        attempts += 1
        if attempts > max_attempts:
            print("Max attempts reached. Halting dataset generation to prevent infinite loop.")
            break

        healthy_file = random.choice(healthy_files)
        
        try:
            audio, sr = librosa.load(healthy_file, sr=SAMPLE_RATE, mono=True)
            
            if counts["misfiring"] < SAMPLES_PER_CLASS:
                fault_audio = simulate_misfiring(audio, sr)
                final_audio = apply_random_augmentation(fault_audio, sr)
                out_name = f"misfiring_synthetic_{counts['misfiring']+1:03d}.wav"
                sf.write(Path(OUTPUT_BASE) / "misfiring" / out_name, final_audio, sr)
                counts["misfiring"] += 1
                
            if counts["tappet"] < SAMPLES_PER_CLASS:
                fault_audio = simulate_tappet(audio, sr)
                final_audio = apply_random_augmentation(fault_audio, sr)
                out_name = f"tappet_synthetic_{counts['tappet']+1:03d}.wav"
                sf.write(Path(OUTPUT_BASE) / "tappet" / out_name, final_audio, sr)
                counts["tappet"] += 1

            if counts["rotational_imbalance"] < SAMPLES_PER_CLASS:
                fault_audio = simulate_rotational_imbalance(audio, sr)
                final_audio = apply_random_augmentation(fault_audio, sr)
                out_name = f"rotational_imbalance_synthetic_{counts['rotational_imbalance']+1:03d}.wav"
                sf.write(Path(OUTPUT_BASE) / "rotational_imbalance" / out_name, final_audio, sr)
                counts["rotational_imbalance"] += 1

            if counts["knocking"] < knocking_needed:
                fault_audio = simulate_knocking(audio, sr, real_knocking_files=existing_real)
                final_audio = apply_random_augmentation(fault_audio, sr)
                out_name = f"knocking_synthetic_{counts['knocking']+1:03d}.wav"
                sf.write(Path(OUTPUT_BASE) / "knocking" / out_name, final_audio, sr)
                counts["knocking"] += 1
                
        except Exception as e:
            # Silently skip file processing errors (as requested)
            continue

    print("\nDataset Generation Summary:")
    print("-" * 35)
    print(f"{'Class':<22} | {'Total Files'}")
    print("-" * 35)
    for fault in faults:
        total = counts[fault]
        if fault == "knocking":
            total += real_knocking_count
        print(f"{fault:<22} | {total}")
    print("-" * 35)

if __name__ == "__main__":
    generate_fault_dataset()
