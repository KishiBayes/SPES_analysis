import SPES_analysis as sd
import os
import pandas as pd
from tqdm import tqdm

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

def FolderToDataframe(dir, save=True):
    files = getDir(dir)
    folder_df = pd.DataFrame()
    for index, file in enumerate(files):
        file_df = pd.DataFrame() #blank per file
        r = sd.SPES_record(file.path)
        resultsDict = r.eventData
        for event in resultsDict.keys():
            data = resultsDict[event]
            eventdf = dataToDataframe(data, sfreq=r.sfreq, file_label=f"File{index}")
            eventdf["stimulation"] = event
            eventdf["file_path"] = file.path
            file_df = pd.concat([file_df, eventdf])
        file_df = file_df.reset_index(drop=True)
        folder_df = pd.concat([folder_df, file_df])
        # Reset index
        folder_df = folder_df.reset_index(drop=True)
        if save:
            print("Saving to pickle")
            folder_df.to_pickle("Output.pkl")
    return folder_df

def dataToDataframe(data: dict, sfreq: int = 256, file_label=None):
    N = len(data["responses"]) #N = number of leads

    # Create a dictionary with the desired column names and values
    df_data = {
        "File": [file_label] * N,
        "sample": [data["sample"]] * N,
        "FirstOfTwoStims": [data["FirstOfTwoStims"]] * N,
        "SecondOfTwoStims": [data["SecondOfTwoStims"]] * N,
        "time": [data["sample"] / int(sfreq)] * N,
        "stimLeads": [data["stimLeads"]] * N,
        "stimLeadNames": [data["stimLeadNames"]] * N,
        "polarity": [data["polarity"]] * N,
        "LikelyResponse": data["LikelyResponse"],
        "lead": data["leads"],
        "preStimFreq": data["preStimFreq"],
        "preStimPhase": data["preStimPhase"],
        "earlyResponseAmp": data["earlyResponseAmp"],
        "earlyResponseLatency": data["earlyResponseLatency"],
        "AmplitudeError": data["shiftAmps"],
        "LatencyError": data["shiftLats"],

        **{
            "Hilbert Amplitudes": [data["HilbertEnvelope"][j] for j in range(N)],
        },
        **{
            "Hilbert Phase": [data["HilbertPhase"][j] for j in range(N)],
        },
        **{
            "response": [data["responses"][j] for j in range(N)],
        },
        ** {
            "Induced Response Frequencies": [data["responses"][j] for j in range(N)],
        },
        ** {
            "response error": [data["shiftResps"][j] for j in range(N)],
        },
        **{
            "preStimWindow": [data["preStimWindow"][j] for j in range(N)],
        },
        **{
            "Induced Response": [data["induced_response"][j] for j in range(N)],
        },
        **{
            "Induced Response Error": [data["shiftInduced"][j] for j in range(N)],
        },
        **{
            "Induced Response Frequencies": [data["frequencies"][j] for j in range(N)],
        },
        }
    return pd.DataFrame(df_data)

if __name__ == "__main__":
    dir = r"C:\Users\rohan\PycharmProjects\SPES_analysis\Testing"
    df = FolderToDataframe(dir)