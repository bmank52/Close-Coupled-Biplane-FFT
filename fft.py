# Fast Fourier Transform of data in a CSV Format for cleaning resaerch data noise 
# Bryce Mankovsky, 8/24/25

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.fft import fft, fftfreq, ifft

rho = 1.225
V = 8.4
S = 0.018

time, aoa, X, Y = [], [], [], []

with open("C:/Users/bmank/Desktop/research/Code/FFT/real_time_aoa_45.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    
    for row in reader:
        time.append(float(row[0]))
        aoa.append(float(row[1]))
        X.append(float(row[2]))
        Y.append(-1 * float(row[3]))

N = len(time)
frq = 5000 #samples per second, ask TA
dt = 1/frq
time = np.arange(N) * dt
lowPass = 10 #Keep everything below this Hz


xFFT = fft(X)
xFFTfreq = fftfreq(N, dt)
maskX = np.abs(xFFTfreq) > lowPass    # <-- correct mask
xFFT[maskX] = 0
xFiltered = np.real(ifft(xFFT))

yFFT = fft(Y)
yFFTfreq = fftfreq(N, dt)
maskY = np.abs(yFFTfreq) > lowPass
yFFT[maskY] = 0
yFiltered = np.real(ifft(yFFT))

avg_drag = np.mean(xFiltered)
avg_lift = np.mean(yFiltered)
print(f"Average Drag (N): {avg_drag:.6f}")
print(f"Average Lift (N): {avg_lift:.6f}")

plt.plot(time, xFiltered, label="Drag")
plt.plot(time, yFiltered, label="Lift")
plt.title("Low Pass Filter 10 Hz 4.5 AoA")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()
plt.show()


'''
yFFTdb = 20 * np.log10(np.abs(yFFT))

pos_mask = yFFTfreq >= 0
yFFTfreq_pos = yFFTfreq[pos_mask]
yFFTdb_pos = yFFTdb[pos_mask]

plt.plot(yFFTfreq_pos, yFFTdb_pos)
plt.title("FFT of Y data (Lift)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.show()


xFFTdb = 20 * np.log10(np.abs(xFFT))

pos_mask = xFFTfreq >= 0
xFFTfreq = xFFTfreq[pos_mask]
xFFTdb = xFFTdb[pos_mask]



plt.plot(xFFTfreq, xFFTdb)
plt.title("FFT of data")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.show()
'''
