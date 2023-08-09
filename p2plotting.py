import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np

original_file = './output_16khz.wav'
processed_file = './output.wav'

# Load the WAV files
original_sample_rate, original_audio = wavfile.read(original_file)
processed_sample_rate, processed_audio = wavfile.read(processed_file)

# Create time vectors
original_time = np.arange(0, len(original_audio)) / original_sample_rate
processed_time = np.arange(0, len(processed_audio)) / processed_sample_rate

# Plot the original audio in time domain
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(original_time, original_audio)
plt.title('Original Audio in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the processed audio in time domain
plt.subplot(2, 1, 2)
plt.plot(processed_time, processed_audio)
plt.title('Processed Audio in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Calculate FFT and frequencies for the original audio
original_fft = np.fft.fft(original_audio)
original_freqs = np.fft.fftfreq(len(original_audio), d=1/original_sample_rate)
original_pos_freqs = original_freqs[1:len(original_freqs)//2]
original_pos_fft = original_fft[1:len(original_fft)//2]

# Calculate FFT and frequencies for the processed audio
processed_fft = np.fft.fft(processed_audio)
processed_freqs = np.fft.fftfreq(len(processed_audio), d=1/processed_sample_rate)
processed_pos_freqs = processed_freqs[1:len(processed_freqs)//2]
processed_pos_fft = processed_fft[1:len(processed_fft)//2]

# Plot FFT for original audio
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(original_pos_freqs, np.abs(original_pos_fft))
plt.title('Original Audio FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Plot FFT for processed audio
plt.subplot(2, 1, 2)
plt.plot(processed_pos_freqs, np.abs(processed_pos_fft))
plt.title('Processed Audio FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
