import SPES_detection as sd
import os
import pandas as pd

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

def FolderToDataframe(dir, includeWindows=False):
    files = getDir(dir)
    folder_df = pd.DataFrame()
    for file in files:
        file_df = pd.DataFrame() #blank per file
        r = sd.SPES_record(file.path)
        resultsDict = r.eventData
        for event in resultsDict.keys():
            data = resultsDict[event]
            eventdf = dataToDataframe(data, includeWindows=includeWindows)
            eventdf["event"] = event
            eventdf["file"] = file.path
            file_df = pd.concat([file_df, eventdf])
        file_df = file_df.reset_index(drop=True)
        folder_df = pd.concat([folder_df, file_df])
    # Reset index
    folder_df = folder_df.reset_index(drop=True)
    folder_df.to_csv("Output.csv")
    return folder_df

def dataToDataframe(data, includeWindows = False):
    N = len(data["responses"])
    M = len(data["responses"][0])

    if includeWindows:
        # Create a dictionary with the desired column names and values
        df_data = {
            "stimLeads": [data["stimLeads"]] * N,
            "polarity": [data["polarity"]] * N,
            **{
                f"response {i + 1}": [data["responses"][j][i] for j in range(N)] for i in range(M)
            },
            **{
                f"preStimWindow {i + 1}": [data["preStimWindow"][j][i] for j in range(N)] for i in range(M)
            },
            "preStimFreq": data["preStimFreq"],
            "preStimPhase": data["preStimPhase"],
            "earlyResponseAmp": data["earlyResponseAmp"],
            "earlyResponseLatency": data["earlyResponseLatency"],
            "LikelyResponse": data["LikelyResponse"]
        }
    else:
        # Create a dictionary with the desired column names and values
        df_data = {
            "stimLeads": [data["stimLeads"]] * N,
            "polarity": [data["polarity"]] * N,
            "preStimFreq": data["preStimFreq"],
            "preStimPhase": data["preStimPhase"],
            "earlyResponseAmp": data["earlyResponseAmp"],
            "earlyResponseLatency": data["earlyResponseLatency"],
            "LikelyResponse": data["LikelyResponse"]
        }
    return pd.DataFrame(df_data)

if __name__ == "__main__":
    dir = r"C:\Users\rohan\PycharmProjects\SPES_analysis\Testing"
    df = FolderToDataframe(dir)
