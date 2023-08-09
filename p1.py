import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.io import wavfile
import pandas as pd


raw_data = pd.read_csv('EMG_Datasets.csv') #Taking EMG Data From CSV File
raw_data

data=np.array(raw_data)

time=data[:,0]
voltage_relaxed=data[:,1]
voltage_contracted=data[:,2]

#Contracted Signal vs. Time
plt.figure(1) #Put in Figure 1
plt.title("Contracted") #title
plt.xlabel("Time (s)") #x-axis title
plt.ylabel("Voltage Amplitude (V)") #y-axis title
plt.plot(time,voltage_contracted,label='Contracted Signal') #Make plot with a label for this data\

#Relaxed Signal vs. Time
plt.figure(1)
plt.plot(time,voltage_relaxed)
plt.title("Before Filtering vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(time,voltage_relaxed,label='Relaxed Signal')

#Contracted Signal vs. Frequency Before Filtering
N=len(voltage_contracted) #Choose data, in this case voltage contracted, for FFT conversion
yf_r=(2/N)*np.abs(fft(voltage_contracted)) #FFT of data
xf=fftfreq(N,1/2050) #Choose FFT frequency
yf_r=yf_r[0:N//2] #Establish y-axis data
xf=xf[0:N//2] #Establish x-axis data

plt.figure(3) #Put this plot of Figure 3
plt.title("Contracted FTT Before Filtering") #title
plt.xlabel("Frequency (Hz)") #x-axis title
plt.ylabel("Voltage Amplitude (V)") #y-axis title
plt.plot(xf,yf_r,label='Contracted Signal') #Make a plot with a label for this data
plt.xlim([0,1000]) #Size of plot to zoom in on important section in x-axis
plt.ylim([0,0.05]) #Size of plot to zoom in on important section in y-axis
plt.legend() #Make legened on plot

#Relaxed Signal vs. Frequency Before Filtering
N=len(voltage_relaxed)
yf_r=(2/N)*np.abs(fft(voltage_relaxed))
xf=fftfreq(N,1/2050)
yf_r=yf_r[0:N//2]
xf=xf[0:N//2]

plt.figure(3)
plt.title("Relaxed FTT Before Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(xf,yf_r, '--',label='Relaxed Signal')
plt.xlim([0,1000])
plt.ylim([0,0.05])
plt.grid()
plt.legend()

N=20500
x=np.linspace(0,10,N)

y_r = voltage_relaxed #New variable so we can write less while coding
y_c = voltage_contracted #New variable so we can write less while coding

order=2 #Order of bandstop filter
f1_l=55 #Lower cutoff frequency of bandstop filter
f2_h=65 #Higher cutoff frequency of bandstop filter

#Bandstop Filter For Relaxed Signal
sos=signal.butter(order, [f1_l,f2_h], btype = 'bandstop', fs=2050, output='sos') #Make bandstop filter
bsf_r=signal.sosfilt(sos,y_r)
N=len(y_r)
yf_r=(2/N)*np.abs(fft(y_r))
bsf_f_r=(2/N)*np.abs(fft(bsf_r))
xf=fftfreq(N,1/2050)
yf_r=yf_r[0:N//2]
bsf_f_r=bsf_f_r[0:N//2]
xf=xf[0:N//2]

#Bandstop Filter For Contracted Signal
sos=signal.butter(order, [f1_l,f2_h], btype = 'bandstop', fs=2050, output='sos')
bsf_c=signal.sosfilt(sos,y_c)
N=len(y_c)
yf_c=(2/N)*np.abs(fft(y_c))
bsf_f_c=(2/N)*np.abs(fft(bsf_c))
xf=fftfreq(N,1/2050)
yf_c=yf_c[0:N//2]
bsf_f_c=bsf_f_c[0:N//2]
xf=xf[0:N//2]

print("Transfer Function of Bandstop Filter:") #Make the transfer function data look nice

numerator, denominator = signal.sos2tf(sos) #Finding numerator and denominator coefficients of the bandstop filter

num = np.array(numerator) #Make array of coefficients found for numerator
den = np.array(denominator) #Make array of coefficients found for denominator

print('')

H = signal.TransferFunction(num, den) #Make transfer function
print("H(s) = ", H) #Print the transfer function made

print('')

#Relaxed Signal vs. Frequency After Bandstop
plt.title("Relaxed FTT After Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(xf,yf_r, '--',label='signal')
plt.plot(xf,bsf_f_r,'k',label='filtered signal')
plt.xlim([0,1000])
plt.ylim([0,0.05])
plt.grid()
plt.legend()

#Contracted Signal vs. Frequency After Bandstop
plt.title("Contracted FTT After Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(xf,yf_c, '--',label='signal')
plt.plot(xf,bsf_f_c,'k',label='filtered signal')
plt.xlim([0,1000])
plt.ylim([0,0.05])
plt.grid()
plt.legend()

order=2 #Order of bandpass filter
f1=0.1 #Lower cutoff frequency of bandpass filter
f2=450 #Higher cutoff frequency of bandpass filter

#Bandpass Filter For Relaxed Signal
sos=signal.butter(order,[f1, f2], btype='bandpass', fs=2050, output='sos') #Make bandpass filter
bpf_r=signal.sosfilt(sos,bsf_r)
N=len(bsf_r)
y_f_r=(2/N)*np.abs(fft(bsf_r))
bpf_f_r=(2/N)*np.abs(fft(bpf_r))
x_f=fftfreq(N,1/2050)
y_f_r=y_f_r[0:N//2]
bpf_f_r=bpf_f_r[0:N//2]
x_f=x_f[0:N//2]

#Bandpass Filter For Contracted Signal
sos=signal.butter(order,[f1, f2], btype='bandpass', fs=2050, output='sos')
bpf_c=signal.sosfilt(sos,bsf_c)
N=len(bsf_c)
y_f_c=(2/N)*np.abs(fft(bsf_c))
bpf_f_c=(2/N)*np.abs(fft(bpf_c))
x_f=fftfreq(N,1/2050)
y_f_c=y_f_c[0:N//2]
bpf_f_c=bpf_f_c[0:N//2]
x_f=x_f[0:N//2]

#Contracted Signal vs. Time After Both Filters
plt.figure(2)
plt.plot(time,bpf_c)
plt.title("Contracted")
plt.xlabel("Time (s)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(time,bpf_c, label='Contracted Signal')
plt.grid()
plt.legend()

#Relaxed Signal vs. Time After Both Filters
plt.figure(2)
plt.plot(time,bpf_r)
plt.title("After Filtering vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(time,bpf_r, label='Relaxed Signal')
plt.grid()
plt.legend()

#Contracted Signal vs. Frequency After Both Filters
plt.figure(4)
plt.title("Contracted FTT After Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(x_f,bpf_f_c,label='Filtered, Contracted Signal')
plt.xlim([0,1000])
plt.ylim([0,0.05])
plt.grid()
plt.legend()

#Relaxed Signal vs. Frequency After Both Filters
plt.figure(4)
plt.title("Relaxed FTT After Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Voltage Amplitude (V)")
plt.plot(x_f,bpf_f_r,label='Filtered, Relaxed Signal')
plt.xlim([0,1000])
plt.ylim([0,0.05])
plt.grid()
plt.legend()

plt.plot() #Plot data
plt.show() #Show the plots

rms_beforefilters_relaxed = np.sqrt(np.mean(voltage_relaxed**2)) #Calculating RMS value with RMS equation
rms_beforefilters_contracted = np.sqrt(np.mean(voltage_contracted**2))
rms_afterfilters_relaxed = np.sqrt(np.mean(bpf_r**2))
rms_afterfilters_contracted = np.sqrt(np.mean(bpf_c**2))

print('The RMS value for the relaxed signal before filtering is:') #Making the data look nice
print(rms_beforefilters_relaxed)
print('')
print('The RMS value for the contracted signal before filtering is:')
print(rms_beforefilters_contracted)
print('')
print('The RMS value for the relaxed signal after filtering is:')
print(rms_afterfilters_relaxed)
print('')
print('The RMS value for the contracted signal after filtering is:')
print(rms_afterfilters_contracted)

z,p,k = signal.butter(order, [f1_l,f2_h], btype='bandstop', fs = 2050, output='zpk') #Find components of the TF for further insight
print("Bandstop Transfer Function Components:")
print("Zeros:")
print(z) #Print the zeros of the TF
print("Poles:")
print(p) #Print the poles of the TF
print("Gain:")
print(k) #Print the gain/floating point of the TF

print('')

print("Transfer Function of Bandpass Filter:")

numerator, denominator = signal.sos2tf(sos)

num = np.array(numerator)
den = np.array(denominator)

print('')

H = signal.TransferFunction(num, den)
print("H(s) = ", H)

print('')

z,p,k = signal.butter(order, [f1, f2], btype='bandpass', fs=2050, output='zpk')
print("Bandpass Transfer Function Components:")
print("Zeros:")
print(z)
print("Poles:")
print(p)
print("Gain:")
print(k)

df = pd.DataFrame({'Time (s)': time, 'Filtered EMG Signal of Relaxed Signal (mV)': bpf_r, 'Filtered EMG Signal of Contracted Signal (mV)': bpf_c}) #Make data frame with data
filename = 'OutputEMGSignal.csv' #Make csv file name
df.to_csv(filename, index=False) #Make csv file