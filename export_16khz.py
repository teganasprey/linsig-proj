import wave
import numpy as np

# Load the original WAV file
input_filename = "./252_quote.wav"
output_filename = "output_16khz.wav"
with wave.open(input_filename, 'rb') as wf:
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    original_sample_rate = wf.getframerate()
    n_frames = wf.getnframes()
    audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)

# Downsample by taking every 3rd frame
downsampled_audio_data = audio_data[::3]

# Update the sample rate
original_sample_rate = original_sample_rate // 3

# Save the downsampled audio to a new WAV file
with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(n_channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(original_sample_rate)
    wf.writeframes(downsampled_audio_data.tobytes())


