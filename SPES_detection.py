import mne
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

def fit_sine(time_series, sampling_rate, min_freq, max_freq):
    # Compute the Fast Fourier Transform (FFT) of the time series
    n = len(time_series)
    frequencies = fftfreq(n, 1 / sampling_rate)
    fft_values = fft(time_series)

    # Filter the FFT values to the desired frequency band
    mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    filtered_fft = fft_values * mask

    # Find the dominant frequency within the frequency band
    dominant_freq_index = np.argmax(np.abs(filtered_fft))
    dominant_freq = frequencies[dominant_freq_index]

    # Estimate the initial phase using the FFT
    dominant_phase = np.angle(fft_values[dominant_freq_index])

    # Fit a sine wave of the dominant frequency to the time series
    def sine_wave(t, amp, phase):
        return amp*np.sin(2 * np.pi * dominant_freq * t + phase)

    time = np.arange(n) / sampling_rate
    popt, _ = curve_fit(sine_wave, time, time_series, p0=[np.max(time_series), dominant_phase])

    fitted_freq = dominant_freq
    phase = popt[1]

    return fitted_freq, phase


def fit_exponential_curve(data):
    x = np.array([i for i in range(len(data))])

    # Define the exponential function to fit
    def exponential_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the exponential curve
    popt, _ = curve_fit(exponential_func, x, y)

    # Generate y-values for the fitted curve
    fitted_y = exponential_func(x, *popt)

    corrected_y = y - fitted_y

    return corrected_y

class SPES_record():
    def __init__(self, file):
        self.file = file
        if self.file.endswith(".edf"):
            self.raw = mne.io.read_raw_edf(file, preload=True)
            self.sfreq = self.raw.info["sfreq"]
            self.data = self.raw.get_data()
            self.analyse_stimulation_events()
        """elif self.file.endswith("_epo.fif"):
            self.epochs = mne.read_epochs(self.file)
            self.sfreq = self.epochs.info["sfreq"]"""

    def analyse_stimulation_events(self, thresholdRatio=20, saveEpochs = True,
                                   windowSize=100, responseTime=2, phaseMeasureWindow=3,
                                   lowPass = 4, highPass = 0.005):
        # Get the data from all the channels
        data = self.raw.get_data()

        # Compute the sum of all channels along the channel axis
        sum_data_OneD = np.sum(np.abs(data), axis=0)
        sum_data = np.reshape(sum_data_OneD, (1, len(sum_data_OneD)))

        # Apply a step function to the summed data
        threshold = thresholdRatio * np.std(sum_data_OneD) + np.average(sum_data_OneD)
        sum_data_step = np.where(sum_data >= threshold, 1, 0)

        # Create a new info object for the summed channel
        sum_info = mne.create_info(["STI"], self.raw.info["sfreq"], ["stim"])
        stim_raw = mne.io.RawArray(sum_data_step, info=sum_info)
        self.raw.add_channels([stim_raw], force_update_info=True)
        self.events = mne.find_events(self.raw, stim_channel=["STI"], initial_event=False)
        self.epochs = mne.epochs.Epochs(raw=self.raw, events=self.events)

        self.stim_times = [x[0] / self.sfreq for x in self.events]

        eventLabels = []
        self.eventData = {}

        for i, event in enumerate(self.events):
            sample = event[0]
            window = data[:, sample - windowSize:sample + windowSize]
            eventAvs = np.mean(abs(window), axis=1)
            quietestLeadIndices = tuple(np.sort(np.argsort(eventAvs)[:2]))
            eventLabels.append(quietestLeadIndices)

        eventLabelSet = set(eventLabels)
        self.eventLabelDict = {}
        for i, eventLabel in enumerate(eventLabelSet):
            self.eventLabelDict[i] = eventLabel
            # Values are a tuple containing the two leads used for stimulation

        keyList = [key for value in eventLabels for key, v in self.eventLabelDict.items() if v == value]

        # Label events in events object with keys from eventLabelDict.
        self.events[:, 2] = keyList
        self.epochs = mne.epochs.Epochs(raw=self.raw, events=self.events)

        if saveEpochs:
            EpochSaveName = str(self.file).split(".")[0] + "_epo.fif"
            self.epochs.save(fname=EpochSaveName, overwrite=True)

        for i, event in enumerate(self.events):
            sample = event[0]
            window = self.data[:, sample - windowSize:sample + windowSize]
            eventAvs = np.mean(abs(window), axis=1)
            quietestLeadIndices = tuple(np.sort(np.argsort(eventAvs)[:2]))
            eventLabels.append(quietestLeadIndices)

            # Find the dominant frequency in the delta band,
            # then fit a sine wave of that frequency to the data before the stim
            # then return the phase

            fitResults = np.apply_along_axis(fit_sine, axis=1,
                                             arr=data[:, int(sample - phaseMeasureWindow * self.sfreq):sample - 1],
                                             sampling_rate=self.sfreq, min_freq=highPass, max_freq=lowPass)

            self.eventData[i] = {"sample": sample,
                                 "stimLeads": quietestLeadIndices,
                                 "polarity": np.sign(np.mean(data[: sample - 2, sample + 2])),
                                 "responses": data[:, sample:int(sample + responseTime * self.sfreq)],
                                 "preStimWindow": data[:, int(sample - phaseMeasureWindow * self.sfreq):sample - 1],
                                 "preStimFreq": fitResults[:, 0],
                                 "preStimPhase": fitResults[:, 1]}

    def correct_stimulus_artefact(self):
        print(self.eventData[3]["responses"][5])
        # File path
        file_path = 'data.csv'

        np.savetxt(file_path, self.eventData[3]["responses"][5], newline=",", delimiter=',')

        for i, event in enumerate(self.events):
            leads = [self.eventData[i]["stimLeads"][0], self.eventData[i]["stimLeads"][1]]

    def make_evoked_and_analyse(self):
        self.evokeds = self.epochs.average(by_event_type=True)
        self.evokedDict = {}
        for i, evoked in enumerate(self.evokeds):
            self.evokedDict[i] = {"data":evoked.data, "evoked":evoked}

if __name__ == "__main__":
    file = r"C:\Users\rohan\PycharmProjects\SPES_analysis\Testing\SPES1.edf"
    a = SPES_record(file)
