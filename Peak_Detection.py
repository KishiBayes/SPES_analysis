"""
Set of algorithms for detecting (and pruning) the peaks in a data array (from raw object).

Optimised for detecting SPES artefacts.

Includes multiple methods, optimised for different PCs.
"""

import mne
import tqdm
import numpy as np

def summed_peaks(data, thresholdRatio=15, proxMin=0.1, sfreq=256):

    # THIS IS THE OLD METHOD
    # Compute the sum of all channels along the channel axis
    sum_data_OneD = np.sum(np.abs(data), axis=0)
    sum_data = np.reshape(sum_data_OneD, (1, len(sum_data_OneD)))

    # Apply a step function to the summed data
    threshold = thresholdRatio * np.std(sum_data_OneD) + np.mean(sum_data_OneD)
    print(f"Threshold for stim channel set at {threshold}")
    peaks = np.where(sum_data >= threshold, 1, 0)
    fixed_sum_data_step = []
    for index, digit in enumerate(peaks[0]):
        if np.sum(peaks[0][index - int(proxMin * sfreq):index]) > 1:
            fixed_sum_data_step.append(0)
        else:
            fixed_sum_data_step.append(digit)
    peaks = np.array(np.reshape(np.array(fixed_sum_data_step), (1, len(fixed_sum_data_step))))
    return peaks

def summed_peaks_simple(data, thresholdRatio=15, proxMin=0.1, sfreq=256):

    # THIS IS THE OLD METHOD
    # Compute the sum of all channels along the channel axis
    sum_data_OneD = np.sum(np.abs(data), axis=0)
    sum_data = np.reshape(sum_data_OneD, (1, len(sum_data_OneD)))

    # Apply a step function to the summed data
    threshold = thresholdRatio * np.std(sum_data_OneD) + np.mean(sum_data_OneD)
    print(f"Threshold for stim channel set at {threshold}")
    peaks = np.where(sum_data >= threshold, 1, 0).flatten()

    N = int(proxMin * sfreq)
    for i in range(N, len(peaks)):
        if 1 in peaks[i - N:i]:
            peaks[i] = 0
    for i in range(N):
        if peaks[i] == 1:
            peaks[i] = 0

    peaks = np.array(np.reshape(np.array(peaks), (1, len(peaks))))
    print(f"There are {np.sum(peaks)} peaks, before removal of paired stimulation")
    return peaks

def summed_windowed_peaks(data, thresholdRatio=30, sfreq=256,
                                   windowSize=50, proxMin=0.5):
    sum_data_OneD = np.sum(np.abs(data), axis=0)

    window = np.lib.stride_tricks.sliding_window_view(sum_data_OneD, windowSize)
    window_mean = np.mean(window, axis=1)
    window_std = np.std(window, axis=1)
    threshold = window_mean + thresholdRatio * window_std
    peaks = np.where(np.abs(sum_data_OneD[windowSize:]) > threshold[:-1], 1, 0)
    peaks = np.array(np.reshape(np.array(peaks), (1, len(peaks))))

    print(peaks)
    print(peaks.shape)
    print(np.sum(peaks))

    fixed_sum_data_step = []
    for index, digit in enumerate(peaks[0]):
        if np.sum(peaks[0][index - int(proxMin * sfreq):index]) > 1:
            fixed_sum_data_step.append(0)
        else:
            fixed_sum_data_step.append(digit)
    peaks = np.array(np.reshape(np.array(fixed_sum_data_step), (1, len(fixed_sum_data_step))))

    print(peaks)
    print(peaks.shape)
    print(np.sum(peaks))

    return peaks

def individual_peak_loops(raw, thresholdRatio=5, saveEpochs=False,
                                   windowSize=100, responseTime=0.2, phaseMeasureWindow=3,
                                   lowPass=4, highPass=0.005, startOffset=0.02, proxMin=0.5,
                                   LastStim=False, bistimDelayMax=2):
    data = raw.get_data()

    peaks = np.zeros((self.n_chans, self.n_samples), dtype=bool)

    for i in range(self.n_chans):
        print(f"Analysing lead {i} of {self.n_chans}")
        for j in tqdm.tqdm(range(windowSize, self.n_samples)):
            window = data[i, j - windowSize:j]
            mean = np.mean(window)
            std = np.std(window)

            if np.any(np.abs(data[i, j]) > mean + thresholdRatio * std):
                peaks[i, j] = True

    return peaks

def individual_peaks_mapped(data, thresholdRatio=20, sfreq=256,
                                   windowSize=256, proxMin=0.5):
    n_chans = data.shape[0]
    n_samples = data.shape[1]
    stacked_rows = np.zeros((n_chans,n_samples - windowSize))
    for i in tqdm.tqdm(range(n_chans)):
        row_data = data[i,:]
        window = np.lib.stride_tricks.sliding_window_view(row_data, windowSize)
        window_mean = np.mean(window, axis=1)
        window_std = np.std(window, axis=1)
        threshold = window_mean + thresholdRatio * window_std
        row_peaks = np.where(np.abs(row_data[windowSize:]) > threshold[:-1], 1, 0)
        stacked_rows[i,:] = row_peaks

    stacked_rows = np.c_[stacked_rows, np.zeros((n_chans, windowSize))]
    print(stacked_rows.shape)

    peaksums = np.sum(stacked_rows, axis = 0).astype(int)
    minimumPeakLeads = 3
    onepeaks = np.where(peaksums>minimumPeakLeads, 1, 0)
    peaks = np.array(np.reshape(np.array(onepeaks), (1, len(onepeaks))))
    print(peaks)
    print(peaks.shape)
    print(np.sum(peaks))

    fixed_sum_data_step = []
    for index, digit in enumerate(peaks[0]):
        if np.sum(peaks[0][index - int(proxMin * sfreq):index]) > 1:
            fixed_sum_data_step.append(0)
        else:
            fixed_sum_data_step.append(digit)
    peaks = np.array(np.reshape(np.array(fixed_sum_data_step), (1, len(fixed_sum_data_step))))

    print(peaks)
    print(peaks.shape)
    print(np.sum(peaks))

    return peaks

if __name__ == "__main__":
    data = np.genfromtxt (r'C:\Users\rohan\PycharmProjects\SPES_analysis\Testing\testingDataSmall.csv', delimiter=",")
    summed_peaks(data)

