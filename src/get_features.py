import librosa
import numpy as np

# Load the MP3 file
file_path = "Darude.mp3"  
y, sr = librosa.load(file_path)

# Calculate loudness using rms
loudness = librosa.feature.rms(y=y)

# Calculate liveness
liveness = librosa.feature.spectral_flatness(y=y)

# Calculate energy using rms (same as loudness)
energy = librosa.feature.rms(y=y)

# Calculate tempo using librosa's beat tracking
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
danceability = np.mean(onset_env) * (tempo / 100) 
harmonic, percussive = librosa.effects.hpss(y)
acousticness = np.sum(np.abs(harmonic)) / (np.sum(np.abs(harmonic)) + np.sum(np.abs(percussive)))
instrumentalness = np.sum(np.abs(harmonic)) / (np.sum(np.abs(harmonic)) + np.sum(np.abs(percussive)))

# Print the calculated features
print(f"Loudness: {loudness.mean()}")
print(f"Liveness: {liveness.mean()}")
print(f"Energy: {energy.mean()}")
print(f"Tempo: {tempo}")
print(f"Danceability (approximation): {danceability}")
print(f"Acousticness (approximation): {acousticness}")
print(f"Instrumentalness (approximation): {instrumentalness}")
