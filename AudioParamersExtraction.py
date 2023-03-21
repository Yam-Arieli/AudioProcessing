import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft, fftfreq
from scipy.io import wavfile
from functools import reduce
import librosa
from IPython.display import Audio, display
import sounddevice as sd
from matplotlib import pyplot as plt

def get_piano_notes():
    octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    
    NOTES_FREQS = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    NOTES_FREQS[''] = 0.0 # stop
    return NOTES_FREQS

NOTES_FREQS = get_piano_notes()

def normalize_array_min_max(ar: np.array) -> np.array:
    mn, mx = ar.min(), ar.max()
    return (ar-mn) / mx

def step_note(base: float, n_halfs=0):
    return base * (2**(n_halfs/12))

def get_sine_wave(frequency, duration, sample_rate, amplitude=1):
    t = np.linspace(0, duration, int(sample_rate*duration)) # Time axis
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave

def clean_freqs(freqs, magnitudes, max_d=10):
    magnitudes = normalize_array_min_max(magnitudes)
    t = 0
    result = []
    cont_collector = []
    
    result_mag = []
    max_mag = -np.inf

    for f, m in zip(freqs, magnitudes):
        if m < 0.1:
            continue

        if f-t > max_d:
            if cont_collector:
                result.append(np.mean(cont_collector))
                result_mag.append(max_mag)

            cont_collector = [f]
            max_mag = m
            t = f
        else:
            cont_collector.append(f)
            max_mag = max_mag if max_mag > m else m
    
    result.append(np.mean(cont_collector))
    result_mag.append(max_mag)
    
    return np.array(list(zip(result, result_mag)))

def notes_recognizer(sound_wave: np.array, window_size=100, ratio=3/2):
    abx = abs(sound_wave)
    abx = abx[:abx.shape[0]-(abx.shape[0]%window_size)]
    chunk_size = int(len(abx)/window_size)
    
    abx = abx.reshape(chunk_size, -1)
    abx = [np.max(window) for window in abx]
    
    eps = 0.00001
    abx = np.array(abx) + eps
    
    d = 1
    r_abx = np.roll(abx, d)
    points_index = np.where(abx-r_abx > 0.2)[0]
    points_values = np.array([abx[p] for p in points_index])
    
    points_index = points_index * window_size
    
    return points_index, points_values

def get_note_of_freq(freq: float) -> str:
    real_note = ''
    smallest_delta = np.inf

    for note in NOTES_FREQS:
        value = NOTES_FREQS[note]
        delta = abs(freq-value)

        if smallest_delta > delta:
            smallest_delta = delta
            real_note = note
    
#     assert len(real_note)
    return real_note

def calculate_wave(xi: np.array, *amplitudes):
    return xi.dot(np.array(amplitudes))

def predict_freqs_and_magnitudes(magnitude_spectrum: np.array, freqs: np.array):
    magnitude_spectrum_normalized = normalize_array_min_max(magnitude_spectrum)
    
    last_mag = np.roll(magnitude_spectrum_normalized, 1)
    next_mag = np.roll(magnitude_spectrum_normalized, -1)
    predicted_freqs = []
    predicted_mag = []

    for f, m, lm, nm, real_m in zip(freqs, magnitude_spectrum_normalized,
                                    last_mag, next_mag, magnitude_spectrum):
        if m > 0.2 and lm < m and nm < m:
            predicted_freqs.append(f)
            predicted_mag.append(real_m)
    
    return np.array(predicted_freqs), np.array(predicted_mag)

def predict_wave_freqs_magnitudes(sound_array: np.array, sr: int, duration: float):
    sound_array_fft = np.fft.fft(sound_array)
    magnitude_spectrum = np.abs(sound_array_fft)
    freqs = np.linspace(0, sr, len(magnitude_spectrum))

    predicted_freqs, predicted_mag = predict_freqs_and_magnitudes(magnitude_spectrum, freqs)

    sine_waves_list = [get_sine_wave(f, duration, sample_rate=sr, amplitude=m/N)
                       for f, m in zip(predicted_freqs, predicted_mag)]

    X = np.array(sine_waves_list)
    X = X.transpose()
    p0 = np.array(predicted_mag)
    
    popt, pcov = curve_fit(calculate_wave, X, sound_array, p0)

    sound_array_pred = X.dot(popt)
    
    return sound_array_pred, predicted_freqs, predicted_mag