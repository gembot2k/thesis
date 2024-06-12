# respiratory.py - Library for respiratory rate measurement using uRAD 24GHz    FMCW radar
# Created by Arisandy Arief - 2024

import numpy as np
import pywt
import scipy.signal as signal

ADC_BITS = 12
ADC_INTERVALS = 2**ADC_BITS
MAX_VOLTAGE = 3.3
NS = 200
DT = 60

WTYPE = "db2"
APPROX = 1


def pre_process(dataset, iqfile, distance, height):
    # Parse dataset into IQ signal data
    iqdata = "../dataset/radar-" + dataset + "/" + iqfile
    data = np.loadtxt(iqdata, delimiter=" ")
    I_data = data[:, :200]
    Q_data = data[:, 200:400]

    # Converting IQ signal to voltage
    I_voltage = (I_data / ADC_INTERVALS) * MAX_VOLTAGE
    Q_voltage = (Q_data / ADC_INTERVALS) * MAX_VOLTAGE

    # Forming complex_signal from IQ voltage
    complex_signal = I_voltage + 1j * Q_voltage

    # Removing DC offset from complex_signal
    complex_signal = complex_signal - np.mean(complex_signal, axis=1, keepdims=True)

    # Applying Hanning window to complex_signal
    hanning_window = np.hanning(NS) * 2 / 3.3
    complex_signal = complex_signal * hanning_window

    return complex_signal


def signal_process(complex_signal):
    # Calculate the phase of the complex signal
    phase_data = np.angle(complex_signal)

    # Calculate mean phase for each sweep
    mean_phase_per_sweep = np.mean(phase_data, axis=1)

    # Perform phase unwrapping
    unwrapped_phase_data = np.unwrap(mean_phase_per_sweep)

    # Choose the type of wavelet
    wavelet = WTYPE

    # Apply wavelet transform
    coeffs = pywt.wavedec(unwrapped_phase_data, wavelet, level=5)

    # Retrieve first approximation coefficients from wavelet transform
    approximation_coeffs = coeffs[APPROX]

    # Apply Hilbert Transform
    analytic_signal = signal.hilbert(approximation_coeffs)

    # Envelope Calculation
    envelope = np.abs(analytic_signal)  # type: ignore

    return envelope


def estimate(envelope, distance, height):
    # Peaks detection on Envelope data
    peaks, _ = signal.find_peaks(envelope, distance=distance, height=height)

    # Calculate the interval between peaks in the sample
    num_samples = len(envelope)
    rr_sample_intervals = np.diff(peaks)

    # Convert the interval between peaks to seconds by dividing by the sampling frequency
    sampling_rate = num_samples / DT
    rr_time_intervals = rr_sample_intervals / sampling_rate

    # Respiratory rate estimation
    if len(rr_time_intervals) > 0:
        rr = DT / np.mean(rr_time_intervals)  # type: ignore
    else:
        rr = 0

    return rr


def feature_extraction(envelope, distance, height):
    peaks, _ = signal.find_peaks(envelope, distance=distance, height=height)

    num_peaks = len(peaks)
    max_peak_distance = np.max(np.diff(peaks)) if num_peaks > 1 else 0
    max_peak_height = np.max(envelope[peaks]) if num_peaks > 0 else 0

    signal_energy = np.sum(envelope**2)

    entropy = -np.sum(
        (envelope / np.sum(envelope)) * np.log(envelope / np.sum(envelope))
    )

    frequencies, psd_values = signal.welch(envelope, fs=1.0, nperseg=256, noverlap=128)
    psd_energy = np.sum(psd_values)

    feature = np.array(
        [
            num_peaks,
            max_peak_distance,
            max_peak_height,
            signal_energy,
            entropy,
            psd_energy,
        ]
    )

    return feature
