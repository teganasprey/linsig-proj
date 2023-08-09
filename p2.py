# Import necessary libraries
import scipy.io.wavfile as wavfile  # For reading WAV files
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # Numerical operations
import scipy.signal as signal  # Signal processing functions
import wave  # For working with audio files


# Function to synthesize sine-waves for each band and superimpose them
def synthesize_chunk(chunk, rms, center_freq, sampling_rate, time):
    # Create time vector for the chunk
    times = np.arange(time, time + len(chunk)) / sampling_rate
    # Generate a sine wave at the specified center frequency
    band_wave = np.sin(2 * np.pi * center_freq * times)
    # Modulate the sine wave amplitude based on the RMS value
    amplitude = rms * band_wave
    return amplitude


# Function to apply band-pass filter
def apply_bandpass_filter(chunk_data, sampling_rate, center_frequency, bandwidth):
    # Calculate the low and high frequencies for the band-pass filter
    low = center_frequency - 0.5 * bandwidth
    high = center_frequency + 0.5 * bandwidth
    # Design a band-pass filter using the Butterworth method
    f = signal.butter(3,
                      [low, high],
                      btype='bandpass',
                      output='sos',
                      fs=sampling_rate)
    # Apply the filter to the chunk data
    filtered_chunk = signal.sosfilt(f, chunk_data)
    return filtered_chunk


# Function for time segmentation
def time_segmentation(audio_data, sampling_rate, chunk_size_ms):
    # Calculate the number of samples per chunk
    chunk_size_samples = int(sampling_rate * chunk_size_ms / 1000)
    # Determine the number of chunks based on the data length
    num_chunks = len(audio_data) // chunk_size_samples
    # Divide the audio data into chunks
    chunks = [audio_data[i*chunk_size_samples:(i+1)*chunk_size_samples] for i in range(num_chunks)]
    # Handle the case when the last chunk is smaller
    if (num_chunks * chunk_size_samples) < len(audio_data):
        chunks.append(audio_data[chunk_size_samples * num_chunks:-1])
    return chunks


# Define file path
PATH = './'

# Define whether to generate plots
DO_PLOTS = False

# Load the .wav file
sampling_rate, audio_data = wavfile.read(PATH + 'output_16khz.wav')

# Handle stereo audio by keeping only the first channel
if len(audio_data.shape) == 2:
    audio_data = audio_data[:, 0]
    print("Audio is stereo. Keeping only the left channel.")

# Create a time vector for preprocessed data
time_preprocessed = [i / sampling_rate for i in range(len(audio_data))]

# Generate plots if requested
if DO_PLOTS:
    plt.figure()
    plt.plot(time_preprocessed, audio_data, label='Preprocessed Audio', color='orange')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Preprocessed Audio Stream')
    plt.legend()
    plt.show()

# Ensure mono audio by taking the mean if stereo
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Define the chunk size in milliseconds
chunk_size_ms = 40

# Divide the audio data into chunks using time_segmentation
chunks = time_segmentation(audio_data, sampling_rate, chunk_size_ms)

# Define the center frequencies and bandwidth for filters
center_frequencies = [100+50*i for i in range(150)]
bandwidth = 100

# Initialize lists for synthesized chunks and RMS values
synthesized_chunks = []
rms_values = []

# Loop through chunks and apply processing
for chunk in chunks:
    filter_chunks = []
    for i, center_freq in enumerate(center_frequencies):
        # Calculate time for superimposing synthesized chunks
        time = len(chunks[0]) * i
        # Apply band-pass filter to the chunk
        filtered_chunk = apply_bandpass_filter(chunk, sampling_rate, center_freq, bandwidth)
        # Calculate RMS of the filtered chunk
        rms = np.sqrt(np.mean(filtered_chunk ** 2))
        rms_values.append(rms)  # Record RMS values for analysis
        # Synthesize the bands and superimpose them
        synthesized_chunk = synthesize_chunk(chunk, rms, center_freq, sampling_rate, time)
        filter_chunks.append(synthesized_chunk)
    # Sum filter chunks to create synthesized chunk for the current time segment
    synthesized_chunks.append(np.sum(filter_chunks, axis=0))

# Concatenate synthesized chunks for the final output stream
output_stream = np.concatenate(synthesized_chunks)

# Save the output stream as a .wav file
output_filename = "output.wav"
with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(1)  # Mono audio
    wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
    wf.setframerate(sampling_rate)
    wf.setnframes(len(output_stream))
    wf.writeframes(output_stream.astype(np.int16).tobytes())
