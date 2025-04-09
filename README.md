# EXP.NO.7-Simulation-of-Digital-Modulation-Techniques
7. Simulation of Digital Modulation Techniques Such as
   i) Amplitude Shift Keying (ASK)
   ii) Frequency Shift Keying (FSK)
   iii) Phase Shift Keying (PSK)

# AIM
```
To study and verify the simulation of Digital Modulation Techniques such as i)Amplitude Shift Keying(ASK) ii)Frequency Shift Keying(FSK) iii)Phase Shift Keying(PSK).
```

# SOFTWARE REQUIRED
```
Colab software.
```

# ALGORITHMS
```
Initialize Parameters:

Set bit rate (bits per second).

Set carrier frequency (Hz).

Define amplitude for bit 1 and bit 0.

Choose number of bits to transmit.

Define how many samples will represent each bit (resolution).

Generate Binary Data:

Create a random binary sequence of given length (e.g., 8 bits).

Create Time Vector:

Calculate bit duration (T = 1 / bit_rate).

Create a time vector for the full duration of the data sequence using samples_per_bit resolution.

Modulate Each Bit (ASK):

For each bit in the data: a. Create a time vector for one bit duration. b. Set the amplitude: use A1 if the bit is 1, otherwise A0. c. Generate a cosine carrier wave at the carrier frequency with the selected amplitude. d. Append this wave to the ASK signal.

Plot the Results:

Plot the binary data as a step/square waveform.

Plot the generated ASK signal against the time vector.

Label the axes and show both plots together.

Display the Plots:

Use matplotlib to visualize the data and ASK waveform.
```

# PROGRAM
```
#ASK Modulation
import numpy as np
import matplotlib.pyplot as plt

# Parameters
bit_rate = 1  # bits per second
f_c = 10      # carrier frequency in Hz
A1 = 1        # Amplitude for bit 1
A0 = 0        # Amplitude for bit 0
bit_count = 8  # Number of bits
samples_per_bit = 100  # Resolution

# Generate random binary data
data = np.random.randint(0, 2, bit_count)
print("Input data:", data)

# Time vector for one bit
T = 1 / bit_rate
t = np.linspace(0, bit_count * T, bit_count * samples_per_bit, endpoint=False)

# ASK signal generation
ask_signal = np.array([])
carrier = np.array([])
for bit in data:
    amplitude = A1 if bit == 1 else A0
    t_bit = np.linspace(0, T, samples_per_bit, endpoint=False)
    wave = amplitude * np.cos(2 * np.pi * f_c * t_bit)
    ask_signal = np.concatenate((ask_signal, wave))

# Plotting
plt.figure(figsize=(10, 6))

# Plot binary data
plt.subplot(2, 1, 1)
plt.title("Binary Data and ASK Modulated Signal")
plt.plot(np.repeat(data, samples_per_bit), label='Binary Data')
plt.ylim(-0.5, 1.5)
plt.ylabel("Bit Value")
plt.grid(True)

# Plot ASK modulated signal
plt.subplot(2, 1, 2)
plt.plot(t, ask_signal, label='ASK Signal', color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
```
```
#FSK Modulation
import numpy as np
import matplotlib.pyplot as plt

# Parameters
bit_rate = 1  # bits per second
f1 = 5        # Frequency for bit 0
f2 = 10       # Frequency for bit 1
sampling_rate = 1000  # Hz
bit_duration = 1 / bit_rate
t = np.arange(0, bit_duration, 1 / sampling_rate)

# Input binary data
data = [1, 0, 1, 1, 0]

# Create the FSK modulated signal
fsk_signal = np.array([])

for bit in data:
    freq = f2 if bit == 1 else f1
    waveform = np.sin(2 * np.pi * freq * t)
    fsk_signal = np.concatenate((fsk_signal, waveform))

# Time vector for the full signal
total_time = np.arange(0, bit_duration * len(data), 1 / sampling_rate)

# Plot the FSK signal
plt.figure(figsize=(10, 4))
plt.plot(total_time, fsk_signal, label='FSK Signal')
plt.title('Frequency Shift Keying (FSK) Modulation')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
```
```
#PSK Modulation
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000                  # Number of bits
Eb_N0_dB = 10             # SNR in dB
Fs = 100                  # Samples per bit
bit_rate = 1              # Bit rate

# Time vector
T = 1 / bit_rate
t = np.linspace(0, N*T, N*Fs)

# Generate random bits
bits = np.random.randint(0, 2, N)

# BPSK Mapping: 0 -> -1, 1 -> +1
bpsk_symbols = 2*bits - 1

# Repeat each symbol to match sample rate
modulated_signal = np.repeat(bpsk_symbols, Fs)

# Add AWGN noise
Eb_N0 = 10**(Eb_N0_dB / 10)
noise_power = 1 / (2 * Eb_N0)
noise = np.sqrt(noise_power) * np.random.randn(N*Fs)
received_signal = modulated_signal + noise

# Demodulation (sampling and decision)
sampled_signal = received_signal[Fs//2::Fs]
received_bits = (sampled_signal > 0).astype(int)

# Calculate BER
bit_errors = np.sum(bits != received_bits)
ber = bit_errors / N
print(f"Bit Error Rate (BER): {ber:.5f}")

# Plot signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.title("BPSK Modulated Signal (First 10 bits)")
plt.plot(t[:10*Fs], modulated_signal[:10*Fs])
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.title("Received Signal with Noise (First 10 bits)")
plt.plot(t[:10*Fs], received_signal[:10*Fs])
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Constellation diagram
plt.figure()
plt.title("BPSK Constellation Diagram")
plt.plot(sampled_signal[:100], np.zeros(100), 'bo')
plt.grid(True)
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.show()
```

# OUTPUT
ASK Modulation
![Screenshot 2025-04-09 193941](https://github.com/user-attachments/assets/2dd0f041-1922-4435-9971-ae005a4c67ab)

FSK Modulation
![Screenshot 2025-04-09 195100](https://github.com/user-attachments/assets/892e3f7b-f056-49bc-b1bf-8d0cfcffa7ba)

PSK Modulation
![Screenshot 2025-04-09 195413](https://github.com/user-attachments/assets/81026b85-5d18-40c2-8310-1e2d07aadaa7)
![Screenshot 2025-04-09 195430](https://github.com/user-attachments/assets/7dc4baab-957a-44cc-9c61-c42d4d3d840e)
 
# RESULT / CONCLUSIONS
```
Thus the simulation of digital modulation techniques of i)Amplitude Shift Keying(ASK) ii)Frequency Shift Keying(FSK) iii) Phase Shift Keying(PSK) are verified.
```
