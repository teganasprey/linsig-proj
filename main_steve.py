import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import wave


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
sampling_rate, audio_data = wavfile.read(PATH + 'output_16khz.wav')

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

################################################
# break this up and loop over chunks

# Apply band-pass filter to each chunk and store the filtered chunks
filtered_chunks = [apply_bandpass_filter(chunk, sampling_rate, center_freq, bandwidth) for center_freq in center_frequencies for chunk in chunks]

# Calculate the RMS value for each band
rms_values = [np.sqrt(np.mean(chunk**2)) for chunk in filtered_chunks]

# Synthesize the bands and superimpose them for each chunk
synthesized_chunks = synthesize_bands(filtered_chunks, rms_values, sampling_rate)

###############################################


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





