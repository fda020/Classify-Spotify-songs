import librosa

# Load the MP3 file
file_path = "march_to_the_sea.mp3"  
y, sr = librosa.load(file_path)

# Calculate loudness using rms
loudness = librosa.feature.rms(y=y)

# Calculate liveness
liveness = librosa.feature.spectral_flatness(y=y)

# Calculate energy using rms (same as loudness)
energy = librosa.feature.rms(y=y)

# Instrumentalness (using tempo as a rough proxy)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

print(f"Loudness: {loudness.mean()}")
print(f"Liveness: {liveness.mean()}")
print(f"Energy: {energy.mean()}")
print(f"Tempo (as a proxy for instrumentalness): {tempo}")


