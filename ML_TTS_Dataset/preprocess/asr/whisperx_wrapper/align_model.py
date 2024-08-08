from dataclasses import dataclass
from typing import Iterable, Union, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from whisperx.types import AlignedTranscriptionResult, SingleSegment, SingleAlignedSegment, SingleWordSegment
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from whisperx.alignment import (
    PUNKT_ABBREVIATIONS,
    LANGUAGES_WITHOUT_SPACES,
    DEFAULT_ALIGN_MODELS_TORCH,
    DEFAULT_ALIGN_MODELS_HF,
)
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union


def load_align_model(
    language_code, device, 
    model_name=None, model_dir=None,
    pipeline_type: Optional[str] = None,
):
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if pipeline_type == "torchaudio" or model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__.get(model_name, None)
        if bundle is not None:
            align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        else:
            pass
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            # processor = Wav2Vec2Processor.from_pretrained(model_name)
            # align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_dir)
            align_model.eval()
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata
