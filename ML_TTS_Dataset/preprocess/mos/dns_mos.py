import os
import argparse
import concurrent.futures
import glob
import tqdm, inspect, io, time, json
import argparse
import bisect
import pickle
import logging
import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from requests import session
import yaml
from moviepy.editor import AudioFileClip

import torch, transformers
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    @staticmethod
    def load_model(model_dir: str, **kwargs):
        p808_model_path = os.path.join(model_dir, 'model_v8.onnx')
        primary_model_path = os.path.join(model_dir, 'sig_bak_ovr.onnx')
        return ComputeScore(primary_model_path, p808_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, sampling_rate):
        audio = librosa.to_mono(audio.T)
        # print(audio.shape) 
        fs = SAMPLING_RATE
        audio = librosa.resample(audio, orig_sr = sampling_rate, target_sr = SAMPLING_RATE)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_p808_mos = []
        if num_hops > 1000:
            return None
        # print(hop_len_samples)
        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            p808_oi = {'input_1': p808_input_features}
            # print(p808_input_features.shape)
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            predicted_p808_mos.append(p808_mos)
        return np.mean(predicted_p808_mos)
