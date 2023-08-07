import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
import scipy.signal as signal
import sounddevice as sd
import wave

# Load the .wav file
sampling_rate, audio_data = wavfile.read('/Users/teganasprey/Desktop/252_quote.wav')

# Plot the time stream
time = [i / sampling_rate for i in range(len(audio_data))]
plt.plot(time, audio_data)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Voice Recording')
plt.show()

duration = len(audio_data) / sampling_rate
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {duration} seconds")

target_sampling_rate = 16000
if sampling_rate > target_sampling_rate:
    from scipy.signal import resample

    audio_data = resample(audio_data, int(len(audio_data) * target_sampling_rate / sampling_rate))
    sampling_rate = target_sampling_rate

    # Update the duration after down-sampling
    duration = len(audio_data) / sampling_rate
    print("Audio down-sampled to 16 kHz.")

if len(audio_data.shape) == 2:
    audio_data = audio_data[:, 0]  # Keep only the first channel (left channel)
    print("Audio is stereo. Keeping only the left channel.")

time_preprocessed = [i / sampling_rate for i in range(len(audio_data))]
plt.figure()
plt.plot(time_preprocessed, audio_data, label='Preprocessed Audio', color='orange')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Preprocessed Audio Stream')
plt.legend()

plt.show()


# Function to apply band-pass filter
def apply_bandpass_filter(audio_data, sampling_rate, center_frequency, bandwidth):
    nyquist = 0.5 * sampling_rate
    low = center_frequency - 0.5 * bandwidth
    high = center_frequency + 0.5 * bandwidth
    low = low / nyquist
    high = high / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    filtered_chunk = signal.lfilter(b, a, audio_data)
    return filtered_chunk


# Function for time segmentation
def time_segmentation(audio_data, sampling_rate, chunk_size_ms):
    chunk_size_samples = int(sampling_rate * chunk_size_ms / 1000)
    num_chunks = len(audio_data) // chunk_size_samples
    chunks = [audio_data[i*chunk_size_samples:(i+1)*chunk_size_samples] for i in range(num_chunks)]
    return chunks


# Load the audio file (replace 'input.wav' with your input file)
input_filename = '/Users/teganasprey/Desktop/252_quote.wav'
with wave.open(input_filename, 'rb') as wf:
    num_frames = wf.getnframes()
    audio_data = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)
    sampling_rate = wf.getframerate()

# Ensure mono audio
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

chunk_size_ms = 30

# Divide the audio data into chunks
chunks = time_segmentation(audio_data, sampling_rate, chunk_size_ms)

# Define the center frequencies and bandwidths for the band-pass filters
center_frequencies = [500, 1000, 2000, 4000, 8000]
bandwidth = 100

# Apply band-pass filter to each chunk and store the filtered chunks
filtered_chunks = [apply_bandpass_filter(chunk, sampling_rate, center_freq, bandwidth) for center_freq in center_frequencies for chunk in chunks]

# Calculate the RMS value for each band
rms_values = [np.sqrt(np.mean(chunk**2)) for chunk in filtered_chunks]


# Function to synthesize sine-waves for each band and superimpose them
def synthesize_bands(filtered_chunks, rms_values, sampling_rate):
    duration_per_chunk = len(filtered_chunks[0]) / sampling_rate
    synthesized_chunks = []

    for chunk_idx in range(len(filtered_chunks)//len(center_frequencies)):
        t_chunk = np.linspace(0, duration_per_chunk, len(filtered_chunks[0]))
        synthesized_chunk = np.zeros_like(filtered_chunks[0])

        for i in range(len(center_frequencies)):
            center_frequency = center_frequencies[i]
            rms_value = rms_values[i]
            amplitude = rms_value
            t = t_chunk
            band_wave = amplitude * np.sin(2 * np.pi * center_frequency * t)
            synthesized_chunk += band_wave

        synthesized_chunks.append(synthesized_chunk)

    return synthesized_chunks


# Synthesize the bands and superimpose them for each chunk
synthesized_chunks = synthesize_bands(filtered_chunks, rms_values, sampling_rate)

# Concatenate the synthesized chunks to obtain the final output stream
output_stream = np.concatenate(synthesized_chunks)

# Save the output stream as a .wav file
output_filename = "output.wav"
with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(1)  # Mono audio
    wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
    wf.setframerate(sampling_rate)
    wf.writeframes(output_stream.tobytes())




