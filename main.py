import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import wave
from scipy.signal import spectrogram


# Function to synthesize sine-waves for each band and superimpose them
def synthesize_chunk(chunk, rms, center_freq, sampling_rate, time):
    times = np.arange(time, time + len(chunk)) / sampling_rate
    band_wave = np.sin(2 * np.pi * center_freq * times)
    amplitude = rms * band_wave
    return amplitude


# Function to apply band-pass filter
def apply_bandpass_filter(chunk_data, sampling_rate, center_frequency, bandwidth):
    low = center_frequency - 0.5 * bandwidth
    high = center_frequency + 0.5 * bandwidth
    f = signal.butter(3,
                 [low, high],
                      btype='bandpass',
                      output='sos',
                      fs=sampling_rate)
    filtered_chunk = signal.sosfilt(f, chunk_data)
    return filtered_chunk


# Function for time segmentation
def time_segmentation(audio_data, sampling_rate, chunk_size_ms):
    chunk_size_samples = int(sampling_rate * chunk_size_ms / 1000)
    num_chunks = len(audio_data) // chunk_size_samples
    chunks = [audio_data[i*chunk_size_samples:(i+1)*chunk_size_samples] for i in range(num_chunks)]
    if (num_chunks * chunk_size_samples) < len(audio_data):
        chunks.append(audio_data[chunk_size_samples * num_chunks:-1])
    return chunks


# file path
PATH = './'

# whether to do plots
DO_PLOTS = False

# Load the .wav file
sampling_rate, audio_data, original_data = wavfile.read(PATH + 'output_16khz.wav')

if len(audio_data.shape) == 2:
    audio_data = audio_data[:, 0]  # Keep only the first channel (left channel)
    print("Audio is stereo. Keeping only the left channel.")

time_preprocessed = [i / sampling_rate for i in range(len(audio_data))]

if DO_PLOTS:
    plt.figure()
    plt.plot(time_preprocessed, audio_data, label='Preprocessed Audio', color='orange')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Preprocessed Audio Stream')
    plt.legend()
    plt.show()

# Ensure mono audio
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

chunk_size_ms = 40

# Divide the audio data into chunks
chunks = time_segmentation(audio_data, sampling_rate, chunk_size_ms)

# Define the center frequencies and bandwidths for the band-pass filters
center_frequencies = [100+50*i for i in range(150)]
bandwidth = 100

synthesized_chunks = []
rms_values = []
for chunk in chunks:
    filter_chunks = []
    for i, center_freq in enumerate(center_frequencies):
        time = len(chunks[0]) * i
        # Apply the band pass filter to the chunk
        filtered_chunk = apply_bandpass_filter(chunk, sampling_rate, center_freq, bandwidth)
        # Calculate the RMS of the filtered chunk
        rms = np.sqrt(np.mean(filtered_chunk ** 2))
        # For recording the RMS values for analysis
        rms_values.append(rms)
        # Synthesize the bands and superimpose them for each chunk
        synthesized_chunk = synthesize_chunk(chunk, rms, center_freq, sampling_rate, time)
        filter_chunks.append(synthesized_chunk)
    synthesized_chunks.append(np.sum(filter_chunks, axis=0))

# Concatenate the synthesized chunks to obtain the final output stream
output_stream = np.concatenate(synthesized_chunks)

# Save the output stream as a .wav file
output_filename = "output.wav"
with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(1)  # Mono audio
    wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
    wf.setframerate(sampling_rate)
    wf.setnframes(len(output_stream))
    wf.writeframes(output_stream.astype(np.int16).tobytes())





