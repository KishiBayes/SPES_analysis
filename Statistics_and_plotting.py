import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ast

def plot_induced_response(induced_response, sampling_rate, frequencies):
    # Plot the induced response
    plt.figure(figsize=(10, 4))
    plt.imshow(induced_response, extent=[0, induced_response.shape[1] / sampling_rate, frequencies[-1], frequencies[0]], aspect='auto', cmap='jet')
    plt.colorbar(label='Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Induced Response')
    plt.show()

def plot_induced_responses(induced_response1, induced_response2, label1, label2, frequencies, sampling_rate):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot for induced_response1
    im1 = ax1.imshow(induced_response1,
                     extent=[0, induced_response1.shape[1] / sampling_rate, frequencies[-1], frequencies[0]],
                     aspect='auto', cmap='jet')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(label1)

    # Plot for induced_response2
    im2 = ax2.imshow(induced_response2,
                     extent=[0, induced_response2.shape[1] / sampling_rate, frequencies[-1], frequencies[0]],
                     aspect='auto', cmap='jet')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(label2)

    # Create a common colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], label='Power')

    plt.show()

def plot_box_and_whisker(arrays, labels=None, Variable=None):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot the box and whisker diagram
    ax.boxplot(arrays)

    # Set labels and title
    if labels:
        ax.set_xticklabels(labels)
    ax.set_xlabel('Data Set')
    ax.set_ylabel(Variable)
    ax.set_title('Box and Whisker Plot')

    # Show the plot
    plt.show()

def split_by_phase(df, removeUnlikely = True, removeBistim = True, shift=True, offset=0):
    if removeUnlikely:
        df = df[df["LikelyResponse"]==True]
    if removeBistim:
        df = df[df["Bistim"]==False]
    posDf = df[df["preStimPhase"] > 0+offset]
    negDf = df[df["preStimPhase"] < 0-offset]
    if shift:
        posIRs = posDf["Induced Response Error"]
        negIRs = negDf["Induced Response Error"]
    else:
        posIRs = posDf["Induced Response"]
        negIRs = negDf["Induced Response"]

    print(f"There are {len(posIRs)} positive phase responses, and {len(negIRs)} negative phase responses")
    frequencies = np.arange(0.5,20,0.5)

    posIRs_3d = np.stack(posIRs.values)
    average_posIR = np.mean(posIRs_3d, axis=0)

    negIRs_3d = np.stack(negIRs.values)
    average_negIR = np.mean(negIRs_3d, axis=0)

    plot_induced_responses(induced_response1=average_posIR, induced_response2=average_negIR, label1="Positive Phase", label2="Negative Phase", sampling_rate=256, frequencies=frequencies)

    posEvAmps = posDf["LatencyError"]
    negEvAmps = negDf["LatencyError"]

    AmpArrays = [list(posEvAmps),list(negEvAmps)]
    labels = ["Positive Phase","Negative Phase"]

    plot_box_and_whisker(AmpArrays,labels=labels, Variable="Amplitude Shift")


if __name__ == "__main__":
    df = pd.read_pickle(r"C:\Users\rohan\PycharmProjects\SPES_analysis\Output.pkl")
    split_by_phase(df=df, shift=False)