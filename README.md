
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
Fs = 1000     # Sampling frequency (samples per second)
Fc = 10       # Carrier frequency in Hz
Tb = 1 / bit_rate  # Bit duration
A1 = 1        # Amplitude for bit 1
A0 = 0        # Amplitude for bit 0
num_bits = 10  # Number of bits to transmit

# Generate random bitstream
bits = np.random.randint(0, 2, num_bits)
print("Transmitted bits:", bits)

# Time vector for the entire signal
t = np.arange(0, num_bits * Tb, 1 / Fs)

# ASK modulation
modulated_signal = np.zeros_like(t)
carrier = np.cos(2 * np.pi * Fc * t)

for i, bit in enumerate(bits):
    amplitude = A1 if bit == 1 else A0
    modulated_signal[i*Fs*int(Tb):(i+1)*Fs*int(Tb)] = amplitude * carrier[i*Fs*int(Tb):(i+1)*Fs*int(Tb)]

# Add noise (optional)
# noise = np.random.normal(0, 0.2, size=t.shape)
# received_signal = modulated_signal + noise
received_signal = modulated_signal  # no noise

# Demodulation
demod_bits = []
for i in range(num_bits):
    segment = received_signal[i*Fs*int(Tb):(i+1)*Fs*int(Tb)]
    corr = np.sum(segment * carrier[i*Fs*int(Tb):(i+1)*Fs*int(Tb)])
    demod_bits.append(1 if corr > 0.5 else 0)

print("Demodulated bits: ", demod_bits)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Bitstream")
plt.step(np.arange(num_bits), bits, where='post')
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("ASK Modulated Signal")
plt.plot(t, modulated_signal)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Demodulated Bits")
plt.step(np.arange(num_bits), demod_bits, where='post', color='r')
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.tight_layout()
plt.show()
```
```
#FSK Modulation
import numpy as np
import matplotlib.pyplot as plt

# Parameters
bit_rate = 1          # bits per second
Fs = 1000             # Sampling frequency (samples per second)
Tb = 1 / bit_rate     # Bit duration
Fc0 = 5               # Frequency for bit 0
Fc1 = 15              # Frequency for bit 1
num_bits = 10         # Number of bits

# Generate random bitstream
bits = np.random.randint(0, 2, num_bits)
print("Transmitted bits:", bits)

# Time vector
t = np.arange(0, num_bits * Tb, 1 / Fs)
modulated_signal = np.zeros_like(t)

# Modulation
for i, bit in enumerate(bits):
    f = Fc1 if bit == 1 else Fc0
    t_bit = np.arange(i * Tb, (i + 1) * Tb, 1 / Fs)
    modulated_signal[i*int(Fs*Tb):(i+1)*int(Fs*Tb)] = np.cos(2 * np.pi * f * t_bit)

# Add noise (optional)
# noise = np.random.normal(0, 0.2, size=t.shape)
# received_signal = modulated_signal + noise
received_signal = modulated_signal

# Demodulation (simple energy detection)
demod_bits = []
for i in range(num_bits):
    segment = received_signal[i*int(Fs*Tb):(i+1)*int(Fs*Tb)]
    t_bit = np.arange(0, Tb, 1 / Fs)
    corr0 = np.sum(segment * np.cos(2 * np.pi * Fc0 * t_bit))
    corr1 = np.sum(segment * np.cos(2 * np.pi * Fc1 * t_bit))
    demod_bits.append(1 if corr1 > corr0 else 0)

print("Demodulated bits:", demod_bits)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Bitstream")
plt.step(np.arange(num_bits), bits, where='post')
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("FSK Modulated Signal")
plt.plot(t, modulated_signal)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Demodulated Bitstream")
plt.step(np.arange(num_bits), demod_bits, where='post', color='r')
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.tight_layout()
plt.show()
```
```
#PSK Modulation
import numpy as np
import matplotlib.pyplot as plt

# Parameters
bit_rate = 1          # bits per second
Fs = 1000             # Sampling frequency (samples per second)
Fc = 10               # Carrier frequency in Hz
Tb = 1 / bit_rate     # Bit duration
num_bits = 10         # Number of bits

# Generate random bitstream
bits = np.random.randint(0, 2, num_bits)
print("Transmitted bits:", bits)

# Time vector
t = np.arange(0, num_bits * Tb, 1 / Fs)
carrier = np.cos(2 * np.pi * Fc * t)
modulated_signal = np.zeros_like(t)

# BPSK Modulation
for i, bit in enumerate(bits):
    phase = 0 if bit == 1 else np.pi
    t_bit = np.arange(i * Tb, (i + 1) * Tb, 1 / Fs)
    modulated_signal[i*int(Fs*Tb):(i+1)*int(Fs*Tb)] = np.cos(2 * np.pi * Fc * t_bit + phase)

# Add noise (optional)
# noise = np.random.normal(0, 0.2, size=t.shape)
# received_signal = modulated_signal + noise
received_signal = modulated_signal

# BPSK Demodulation (coherent detection)
demod_bits = []
for i in range(num_bits):
    segment = received_signal[i*int(Fs*Tb):(i+1)*int(Fs*Tb)]
    t_bit = np.arange(0, Tb, 1 / Fs)
    reference = np.cos(2 * np.pi * Fc * t_bit)
    corr = np.sum(segment * reference)
    demod_bits.append(1 if corr > 0 else 0)

print("Demodulated bits:", demod_bits)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Bitstream")
plt.step(np.arange(num_bits), bits, where='post')
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("BPSK Modulated Signal")
plt.plot(t, modulated_signal)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Demodulated Bitstream")
plt.step(np.arange(num_bits), demod_bits, where='post', color='r')
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.tight_layout()
plt.show()
```

# OUTPUT
ASK Modulation
![Screenshot 2025-04-11 132940](https://github.com/user-attachments/assets/e16c27c5-86cc-433f-99b9-42c9b731a6bf)

FSK Modulation
![Screenshot 2025-04-11 133308](https://github.com/user-attachments/assets/6cf573ca-be8a-45d1-ab01-b91189481dfa)

PSK Modulation
![Screenshot 2025-04-11 133603](https://github.com/user-attachments/assets/8b205681-1e11-4957-9081-d734ad7eec8d)

# RESULT / CONCLUSIONS
```
Thus the simulation of digital modulation techniques of i)Amplitude Shift Keying(ASK) ii)Frequency Shift Keying(FSK) iii) Phase Shift Keying(PSK) are verified.
```
