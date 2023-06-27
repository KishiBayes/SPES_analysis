import SPES_analysis as sd
import os
import pandas as pd
import numpy as np

def getDir(dir):
    files = []
    for file in os.scandir(dir):
        if file.is_dir():
            # Go deeper
            subd_files = getDir(file.path)
            files.append(subd_files)
        else:
            if file.name.endswith(".edf"):
                files.append(file)
    return files

def FolderToDataframe(dir, includeWindows=True, save=True):
    files = getDir(dir)
    folder_df = pd.DataFrame()
    for file in files:
        file_df = pd.DataFrame() #blank per file
        r = sd.SPES_record(file.path)
        resultsDict = r.eventData
        for event in resultsDict.keys():
            data = resultsDict[event]
            eventdf = dataToDataframe(data, includeWindows=includeWindows, sfreq=r.sfreq)
            eventdf["event"] = event
            eventdf["file"] = file.path
            file_df = pd.concat([file_df, eventdf])
        file_df = file_df.reset_index(drop=True)
        folder_df = pd.concat([folder_df, file_df])
        # Reset index
        folder_df = folder_df.reset_index(drop=True)
        if save:
            print("Saving to pickle")
            folder_df.to_pickle("Output.pkl")
    return folder_df

def dataToDataframe(data: dict, includeWindows: bool = True, sfreq: int = 256):
    N = len(data["responses"]) #N = number of leads
    M = len(data["responses"][0]) #M = number of samples in "response" object
    M_ind = len(data["induced_response"][0]) #M_ind = number of elements in induced response array
    M_ind_E = len(data["shiftInduced"][0]) #M_ind_E = number of elements in induced response error array

    if includeWindows:
        # Create a dictionary with the desired column names and values
        df_data = {
            "sample": [data["sample"]] * N,
            "Bistim": [data["DoubleStim"]] * N,
            "time": [data["sample"] * int(sfreq)] * N,
            "stimLeads": [data["stimLeads"]] * N,
            "stimLeadNames": [data["stimLeadNames"]] * N,
            "polarity": [data["polarity"]] * N,
            "lead": data["leads"],
            **{
                f"response {i + 1}": [data["responses"][j][i] for j in range(N)] for i in range(M)
            },
            **{
                f"response error {i + 1}": [data["shiftResps"][j][i] for j in range(N)] for i in range(M)
            },
            **{
                f"preStimWindow {i + 1}": [data["preStimWindow"][j][i] for j in range(N)] for i in range(M)
            },
            **{
                "Induced Response":[data["induced_response"][j] for j in range(N)],
            },
            **{
                "Induced Response Error":[data["shiftInduced"][j] for j in range(N)],
            },
            **{
                "Induced Response Frequencies": [data["frequencies"][j] for j in range(N)],
            },
            "preStimFreq": data["preStimFreq"],
            "preStimPhase": data["preStimPhase"],
            "earlyResponseAmp": data["earlyResponseAmp"],
            "earlyResponseLatency": data["earlyResponseLatency"],
            "LikelyResponse": data["LikelyResponse"],
            "AmplitudeError": data["shiftAmps"],
            "LatencyError": data["shiftLats"]
        }
    else:
        # Create a dictionary with the desired column names and values
        df_data = {
            "sample": [data["sample"]] * N,
            "Bistim": [data["DoubleStim"]] * N,
            "time": [data["sample"] * int(sfreq)] * N,
            "stimLeads": [data["stimLeads"]] * N,
            "stimLeadNames": [data["stimLeadNames"]] * N,
            "polarity": [data["polarity"]] * N,
            "lead": data["leads"],
            "preStimFreq": data["preStimFreq"],
            "preStimPhase": data["preStimPhase"],
            "earlyResponseAmp": data["earlyResponseAmp"],
            "earlyResponseLatency": data["earlyResponseLatency"],
            "LikelyResponse": data["LikelyResponse"],
            "AmplitudeError": data["shiftAmps"],
            "LatencyError": data["shiftLats"]
        }
    return pd.DataFrame(df_data)

if __name__ == "__main__":
    dir = r"C:\Users\rohan\PycharmProjects\SPES_analysis\Testing"
    df = FolderToDataframe(dir)

    import Statistics_and_plotting as SaP

    SaP.split_by_phase(df)