import mne
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pywt


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
        return amp * np.sin(2 * np.pi * dominant_freq * t + phase)

    time = np.arange(n) / sampling_rate
    popt, _ = curve_fit(sine_wave, time, time_series, p0=[np.max(time_series), dominant_phase])

    fitted_freq = dominant_freq
    phase = popt[1]

    return fitted_freq, phase


class SPES_record():
    def __init__(self, file, peak_detection_testing_mode=False):
        self.startOffset = None
        self.file = file
        if self.file.endswith(".edf"):
            self.raw = mne.io.read_raw_edf(file, preload=True)
            self.sfreq = self.raw.info["sfreq"]
            print(f"Sampling frequency is {self.sfreq}, so resolution is {1 / self.sfreq}")
            self.data = self.raw.get_data()
            self.analyse_stimulation_events()
            if not peak_detection_testing_mode:
                self.getRelativeShifts()
            if peak_detection_testing_mode:
                self.exportAnnotatedEdf(output_path="Annotatedstims.edf")

    def detect_peaks(self, method=None):
        data = self.raw.get_data()
        from Peak_Detection import individual_peaks_mapped as ipm
        from Peak_Detection import summed_windowed_peaks as swp
        from Peak_Detection import summed_peaks as sp
        from Peak_Detection import summed_peaks_simple as sps
        if method == None:
            return sps(data, sfreq=self.sfreq)
        else:
            return method(data, sfreq=self.sfreq)

    def analyse_stimulation_events(self, saveEpochs: bool=False,
                                   windowSize: int=100, responseTime=0.5, phaseMeasureWindow=3,
                                   lowPass=4, highPass=0.005, startOffset=0.02, bistimDelayMax=2):
        # Get the data from all the channels
        self.startOffset = startOffset

        data = self.raw.get_data()
        originalLeads = self.raw.ch_names
        self.n_chans = data.shape[0]
        self.n_samples = data.shape[1]

        peaks = self.detect_peaks()

        # Create a new info object for the summed channel
        sum_info = mne.create_info(["STI"], self.raw.info["sfreq"], ["stim"])
        stim_raw = mne.io.RawArray(peaks, info=sum_info, verbose=False)
        self.raw.add_channels([stim_raw], force_update_info=True)
        self.events = mne.find_events(self.raw, stim_channel=["STI"], initial_event=False)

        self.stim_times = [x[0] / self.sfreq for x in self.events]

        eventLabels = []
        self.eventData = {}

        for i, event in enumerate(self.events):
            sample = event[0]
            window = data[:, sample - windowSize:sample + windowSize]
            eventAvs = np.mean(abs(window), axis=1)
            quietestLeadIndices = str(np.sort(np.argsort(eventAvs)[:2]))
            eventLabels.append(quietestLeadIndices)

        eventLabelSet = set(eventLabels)
        self.eventLabelDict = {}
        for i, eventLabel in enumerate(eventLabelSet):
            self.eventLabelDict[i] = str(eventLabel)
            # Values are a tuple containing the two leads used for stimulation

        keyList = [key for value in eventLabels for key, v in self.eventLabelDict.items() if v == value]

        # Label events in events object with keys from eventLabelDict.
        self.events[:, 2] = keyList
        self.epochs = mne.epochs.Epochs(raw=self.raw, events=self.events, verbose=False)

        if saveEpochs:
            EpochSaveName = str(self.file).split(".")[0] + "_epo.fif"
            self.epochs.save(fname=EpochSaveName, overwrite=True)

        for i, event in enumerate(tqdm(self.events, desc="Extracting event data...")):
            sample = event[0]
            window = self.data[:, sample - windowSize:sample + windowSize]
            eventAvs = np.mean(abs(window), axis=1)
            quietestLeadIndices = tuple(np.sort(np.argsort(eventAvs)[:2]))
            eventLabels.append(quietestLeadIndices)

            try:
                # Find the dominant frequency in the delta band,
                # then fit a sine wave of that frequency to the data before the stim
                # then return the phase

                fitResults = np.apply_along_axis(fit_sine, axis=1,
                                                 arr=data[:, int(sample - phaseMeasureWindow * self.sfreq):sample - 1],
                                                 sampling_rate=self.sfreq, min_freq=highPass, max_freq=lowPass)
                maxima = np.sqrt(
                    np.max(
                        np.square(data[:, sample:int(sample + startOffset * self.sfreq + responseTime * self.sfreq)]),
                        axis=1))
                maximaIndex = np.argmax(
                    np.abs(data[:, sample:int(sample + startOffset * self.sfreq + responseTime * self.sfreq)]), axis=1)
            except ZeroDivisionError as z:
                print(f"Error in fitting delta sine wave at sample {sample}, {z}")
                maxima = np.zeros((1, self.n_chans))
                maximaIndex = np.zeros((1, self.n_chans))
            self.eventData[i] = {"sample": sample,
                                 "stimLeads": quietestLeadIndices,
                                 "stimLeadNames": [self.raw.ch_names[j] for j in quietestLeadIndices],
                                 "polarity": np.sign(np.mean(data[: sample - 2, sample + 2])),
                                 "responses": data[:, sample:
                                                      int(sample + startOffset * self.sfreq + responseTime * self.sfreq)],
                                 "preStimWindow": data[:, int(sample - phaseMeasureWindow * self.sfreq):sample - 1],
                                 "preStimFreq": fitResults[:, 0],
                                 "preStimPhase": fitResults[:, 1],
                                 "leads": originalLeads,
                                 "earlyResponseAmp": maxima,
                                 "earlyResponseLatency": 1000 * (startOffset * self.sfreq + maximaIndex) / self.sfreq,
                                 "LikelyResponse": np.where(maximaIndex > 1, True, False)}

        # Detect bistim.
        for i, event in enumerate(self.events):
            if i != 0:
                self.eventData[i]["DoubleStim"] = self.eventData[i]["sample"] - self.eventData[i - 1][
                    "sample"] < bistimDelayMax * self.sfreq

        for i, event in enumerate(self.events):  # separate loop so it doesn't break it for the next guy.
            if i < len(self.eventData.keys()):
                if self.eventData[i]["sample"] - self.eventData[i + 1]["sample"] < bistimDelayMax * self.sfreq:
                    del (self.eventData[i])

        # Compute Induced Responses
        for k, v in tqdm(self.eventData.items(), desc="Computing induced responses..."):
            sample = v["sample"]
            inducedDict = self._inducedResponse(sample)
            for resultType, resultArray in inducedDict.items():
                v[resultType] = resultArray

    def getRelativeShifts(self):
        """
        1. Groupby stim leads and polarity
        2. Get the average response amplitude, response latency and average the response array
        3. For each amp, latency and response array, induced responses, subtract these averages
        :return: self, but self.eventData[i] now has 4 new values
        """
        grouped_events = {}
        for event_id, event in tqdm(self.eventData.items(), desc="Computing relative shifts..."):
            stim_leads = event['stimLeads']
            polarity = event['polarity']
            group_key = (stim_leads, polarity)
            if group_key not in grouped_events:
                grouped_events[group_key] = []
            grouped_events[group_key].append(event)

        event_averages = {}

        for group_id, eventGroup in grouped_events.items():
            # Extract the arrays from the dictionaries
            boolArray = np.array([d["LikelyResponse"] for d in eventGroup])
            AmpArray = np.array([d["earlyResponseAmp"] for d in eventGroup])
            LatArray = np.array([d["earlyResponseLatency"] for d in eventGroup])
            ResponseArray = np.array([d["responses"] for d in eventGroup])
            InducedArray = np.array([d["induced_response"] for d in eventGroup])

            # Compute the average of array values where boolArrays is True
            avAmps = np.mean(AmpArray[boolArray], axis=0)
            avLats = np.mean(LatArray[boolArray], axis=0)
            avResps = np.mean(ResponseArray[boolArray], axis=0)
            avInduced = np.mean(InducedArray[boolArray], axis=0)
            event_averages[group_id] = {"meanAmps": avAmps, "meanLats": avLats, "meanResps": avResps,
                                        "meanInduced": avInduced}

        for event_id, event in self.eventData.items():
            stim_leads = event['stimLeads']
            polarity = event['polarity']
            group_key = (stim_leads, polarity)
            self.eventData[event_id].update(event_averages[group_key])
            event_shifts = {
                "shiftAmps": (self.eventData[event_id]["earlyResponseAmp"] - self.eventData[event_id]["meanAmps"]),
                "shiftLats": (self.eventData[event_id]["earlyResponseLatency"] - self.eventData[event_id]["meanLats"]),
                "shiftResps": (self.eventData[event_id]["responses"] - self.eventData[event_id]["meanResps"]),
                "shiftInduced": (self.eventData[event_id]["induced_response"] - self.eventData[event_id]["meanInduced"])
                }
            self.eventData[event_id].update(event_shifts)

    def _inducedResponse(self, sample, window=0.5, sampling_rate=None):
        if sampling_rate == None:
            sampling_rate = self.sfreq
        baselineData = self.data[:, int(sample - window * self.sfreq - 1):int(sample - 1)]
        responseData = self.data[:, int(sample + self.startOffset * self.sfreq):
                                    int(sample + self.startOffset * self.sfreq + window * self.sfreq)]

        signal = self.data[:, int(sample - window * self.sfreq - 1):int(sample + self.startOffset * self.sfreq + window * self.sfreq)]

        # Parameters for the wavelet transform
        wavelet = 'morl'
        frequencies = np.arange(0.5,20,0.5) / self.sfreq
        scales = pywt.frequency2scale('cmor1.5-1.0', frequencies)
        sampling_period = 1/self.sfreq

        # Define a function to apply the wavelet transform to a single row
        def apply_wavelet_transform(row):
            coefficients = pywt.cwt(row, scales, wavelet)[0]

            power_spectrum = (np.abs(coefficients)) ** 2
            # Split the power spectrum back into before and after segments
            power_before = power_spectrum[:, :len(baselineData)]
            power_after = power_spectrum[:, -len(responseData):]
            induced_response = power_after - power_before
            return {"coefficients": coefficients, "frequencies": frequencies, "power_spectrum": power_spectrum,
                    "induced_response": induced_response}

        result = [apply_wavelet_transform(row) for row in signal]

        result_dict = {}
        for dictionary in result:
            for key, value in dictionary.items():
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].append(value)

        return result_dict

    def exportAnnotatedEdf(self, output_path):
        annot_from_events = mne.annotations_from_events(
            events=self.events,
            event_desc=self.eventLabelDict,
            sfreq=self.sfreq,
            orig_time=self.raw.info["meas_date"],
        )
        self.raw.set_annotations(annot_from_events)

        # Save the modified Raw object as an EDF file
        mne.export.export_raw(output_path, raw=self.raw, overwrite=True)

    def samplePlots(self, numberOfSamples=6):
        if numberOfSamples > len(self.epochs.events):
            return KeyError

        for i in range(numberOfSamples):
            ep = np.random.randint(1, len(self.epochs.events))
            epoch = self.epochs[ep]
            evoked = epoch.average()
            evoked.plot()


if __name__ == "__main__":
    sr = SPES_record(r"C:\Users\rohan\PycharmProjects\SPES_analysis\Testing\SPES1.edf",
                     peak_detection_testing_mode=True)

