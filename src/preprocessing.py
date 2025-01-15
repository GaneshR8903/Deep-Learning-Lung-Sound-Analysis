import os
import shutil
import numpy as np
import scipy.signal as signal
import librosa
import config
import pickle
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import pywt
from PyEMD import EMD


def Filter_Denoised(raw_audio, sample_rate, filter_order, filter_lowcut, filter_highcut, btype="bandpass"):
    b, a = 0.0, 0.0
    if btype == "bandpass":
        b, a = signal.butter(filter_order, [filter_lowcut/(sample_rate/2), filter_highcut/(sample_rate/2)], btype=btype)

    if btype == "highpass":
        b, a = signal.butter(filter_order, filter_lowcut, btype=btype, fs=sample_rate)


    audio = signal.lfilter(b, a, raw_audio)

    return audio



def Wavelet_Denoise(signal, wavelet='db4', level=1, threshold=None, mode='soft'):

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    if threshold is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode=mode)

    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal



def EMD_Denosing(data):
    # Ref: A Lightweight CNN Model for Detecting Respiratory Diseases
    # From Lung Auscultation Sounds Using EMD-CWT-Based Hybrid Scalogram
    emd = EMD()
    emd.emd(data, max_imf=9)
    imfs, res = emd.get_imfs_and_residue()

    goal_index = 0
    correct = -10000
    for imf_i in range(len(imfs)):
        temp = np.correlate(data, imfs[imf_i])[0]
        if temp > correct:
            correct = temp
            goal_index = imf_i
    emd_sig = imfs[goal_index]
    return emd_sig



def Padding(data, sample_rate, respiratory_cycle, padding_more):

    if len(data) == (sample_rate * respiratory_cycle):
        return data  # the duration of data is equal to the length of the demand
    else:
        padding = None
        if padding_more == "zero":
            padding = np.zeros((data.shape[0]))
        elif padding_more == "sample":
            padding = data.copy()

        while True:  # 反复拼接
            data = np.concatenate([data, padding])
            if len(data) > (sample_rate * respiratory_cycle):
                data = data[:int(sample_rate * respiratory_cycle)]

            if len(data) == (sample_rate * respiratory_cycle):
                return data



def Segmentation(audio, label_dir, sample_rate, respiratory_cycle, overlap, padding_more, show_function=False):
    samples = {
        "signal": [],
        "label": [],
    }

    filelabel = open(f"{label_dir}", "r")
    res = filelabel.readlines()
    filelabel.close()

    for i, cur in enumerate(res):
        cur = cur.strip("\n")
        lStart, lEnd, lcrackle, lwheeze = cur.split("\t")
        lStart = float(lStart)
        lEnd = float(lEnd)
        lcrackle = int(lcrackle)
        lwheeze = int(lwheeze)

        # Label Construction
        label = None  # normal - 0, crackle - 1, wheezes - 2, both - 3
        if lcrackle == 0 and lwheeze == 0:
            label = config.normal
        elif lcrackle == 1 and lwheeze == 0:
            label = config.crackle
        elif lcrackle == 0 and lwheeze == 1:
            label = config.wheezes
        elif lcrackle == 1 and lwheeze == 1:
            label = config.both

        # Data Construction
        while lStart < lEnd:
            temp_start = lStart
            temp_end = temp_start + respiratory_cycle
            if temp_end > lEnd:
                temp_end = lEnd

            if (lEnd - lStart) < (config.respiratory_cycle // 2):
                break

            temp_start = int(temp_start * sample_rate)
            temp_end = int(temp_end * sample_rate)

            temp_data = Padding(audio[temp_start:temp_end], sample_rate, respiratory_cycle, padding_more)

            if show_function:
                plt.plot(temp_data)
                plt.show()

            samples["signal"].append(temp_data)
            samples["label"].append(label)

            lStart += overlap

    return samples



def Diagnosis_label(subject, diagnosis_file_dir):
    filelabel = open(f"{diagnosis_file_dir}", "r")
    res = filelabel.readlines()
    filelabel.close()

    diagosis_label_list = {}
    for i, cur in enumerate(res):
        cur = cur.strip("\n")
        subjectID, diagnosis = cur.split("\t")

        # Print for debugging
        print(f"Processing: {subjectID} -> {diagnosis}")

        if diagnosis == "URTI":
            diagnosis = config.URTI
        elif diagnosis == "Healthy":
            diagnosis = config.Healthy
        elif diagnosis == "Asthma":
            diagnosis = config.Asthma
        elif diagnosis == "COPD":
            diagnosis = config.COPD
        elif diagnosis == "LRTI":
            diagnosis = config.LRTI
        elif diagnosis == "Bronchiectasis":
            diagnosis = config.Bronchiectasis
        elif diagnosis == "Pneumonia":
            diagnosis = config.Pneumonia
        elif diagnosis == "Bronchiolitis":
            diagnosis = config.Bronchiolitis

        diagosis_label_list[subjectID] = diagnosis

    # Print the full dictionary for debugging
    print(f"Diagnosis Label List: {diagosis_label_list}")

    if subject not in diagosis_label_list:
        raise KeyError(f"Subject ID {subject} not found in diagnosis file.")

    return diagosis_label_list[subject]




def Preprocessing(dir, preprocessed_dir_savesamples):
    print('start Preprocessing')

    # Make dir
    if os.path.exists(f"{preprocessed_dir_savesamples}") and config.save_for_preprocessing_and_feature_extraction:
        shutil.rmtree(f"{preprocessed_dir_savesamples}")
        os.makedirs(f"{preprocessed_dir_savesamples}")
    else:
        os.makedirs(f"{preprocessed_dir_savesamples}")

    for file_name in os.listdir(dir):
        if ".wav" not in file_name:
            continue

        # path
        data_dir = f"{dir}\\{file_name}"
        label_dir = f"{dir}\\{file_name.split('.')[0]}.txt"

        # load data
        raw_audio, sample_rate = librosa.load(path=data_dir, sr=config.sample_rate)

        # Noise reduction method, filterconfig
        audio_data = Filter_Denoised(raw_audio, sample_rate, config.filter_order,
                                     config.filter_lowcut, config.filter_highcut, btype=config.filter_btype)
        # Candidate method
        # ..........

        # Segmentation - data & label
        samples = Segmentation(audio_data, label_dir, sample_rate, config.respiratory_cycle, config.overlap, config.padding_mode)
        if samples["signal"] == []:
            continue

        # diagnosis label
        subject = file_name[:3]
        diagnosis_label = Diagnosis_label(subject, config.diagnosis_file_dir)

        # Save to the preprocessed_dir_savesamples
        if config.save_for_preprocessing_and_feature_extraction:
            for i in range(len(samples["signal"])):
                save_dir = preprocessed_dir_savesamples + '\\' + file_name.split('.')[0] + f"_{i}.dat"

                temp = {
                    "signal": samples["signal"][i],
                    "label": samples["label"][i],
                    "diagnosis": diagnosis_label
                }
                with open(save_dir, 'wb') as f:
                    pickle.dump(temp, f)

        print(f"{file_name} over")



if __name__ == '__main__':
    Preprocessing(config.raw_dir, config.preprocessed_dir_savesamples)


