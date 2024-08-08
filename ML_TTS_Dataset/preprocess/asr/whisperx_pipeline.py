  import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tqdm, inspect, io, time, json
import argparse
import bisect
import pickle
import logging
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import pyannote.audio
import whisperx
import yaml
from moviepy.editor import AudioFileClip

import torch, transformers
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union

from ML_TTS_Dataset.utils.log import get_logger
from ML_TTS_Dataset.preprocess.asr.filter_utils import (
    find_time_interval_idx,
    is_segment_interval_larger_than_threshold,
    return_segment_interval_larger_than_threshold,
    is_segment_less_equal_than_n_speakers,
    return_segment_less_equal_than_n_speakers,
    list_to_segment,
)
from ML_TTS_Dataset.preprocess.mos.dns_mos import ComputeScore
from ML_TTS_Dataset.utils.audio import (
    load_scp, load_audio_list_by_split, create_scp_from_dir, get_output_audio_path, pydub_to_np, split_audio
)
from ML_TTS_Dataset.preprocess.asr.whisperx_wrapper.align_model import load_align_model

logger = get_logger(log_level=logging.INFO)


class AutomaticSpeechRecognitionConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: Optional[str] = "whsiperx",
        model_dir: Optional[str] = None,
        model_size: Optional[str] = "large-v3",
        lang2align_model_yaml_path: Optional[str] = None,
        available_langs: Optional[Sequence[str]] = ["en", "zh"],
        vad_model_path: Optional[str] = None,
        diarize_model_path: Optional[str] = None,
        dns_model_dir: Optional[str] = None,
        precision: Optional[str] = "float16",
        language: Optional[str] = None,
        task: Optional[str] = "transcribe",
        initial_prompt: Optional[str] = None,
        suppress_numerals: Optional[bool] = None,
        target_sr: Optional[int] = 16_000,
        target_format: Optional[str] = "wav",
        chunk_size: Optional[int] = 30,
        batch_size: Optional[int] = None,
        num_threads: Optional[int] = 0,
        num_workers: Optional[int] = 4, # actually num_threads x num_workers
        end_punctuation_list: Optional[Sequence[str]] = None,
        device: Optional[str] = "cuda:0",
        **kwargs
    ):
        if language == "zh" and initial_prompt is None:
            logger.warning("You should prompt '以下是一段语音记录。' to generate punctuations when the language is zh!")
        super().__init__(
            model_type=model_type,
            model_dir=model_dir,
            model_size=model_size,
            lang2align_model_yaml_path=lang2align_model_yaml_path,
            available_langs=available_langs,
            vad_model_path=vad_model_path,
            diarize_model_path=diarize_model_path,
            dns_model_dir=dns_model_dir,
            precision=precision,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            suppress_numerals=suppress_numerals,
            target_sr=target_sr,
            target_format=target_format,
            chunk_size=chunk_size,
            batch_size=batch_size,
            num_threads=num_threads,
            num_workers=num_workers,
            end_punctuation_list=end_punctuation_list,
            device=device,
        )


class  AutomaticSpeechRecognitionPipeline(PreTrainedModel):
    def __init__(
        self,
        config: Union[dict,  AutomaticSpeechRecognitionConfig],
        **kwargs
    ):
        if isinstance(config, dict):
            config =  AutomaticSpeechRecognitionConfig(**config)
        super().__init__(config=config)
        self.model = self.load_model()
    
    def load_model(self, **kwargs):
        pass

    def forward(self, input_audio_path, output_transcription_path: Optional[str] = None, **kwargs):
        pass

    @classmethod
    def build_pipeline(
        cls,
        config: AutomaticSpeechRecognitionConfig,
        **kwargs
    ) -> " AutomaticSpeechRecognitionPipeline":
        if isinstance(config, dict):
            config =  AutomaticSpeechRecognitionConfig(**config)
        if config.model_type == "whisperx":
            return WhisperxPipeline(config=config)
        raise NotImplementedError

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_type', type=str, default="whsiperx")
        parser.add_argument('--model_dir', type=str, default=None)
        parser.add_argument('--model_size', type=str, default="large-v3")
        parser.add_argument('--lang2align_model_yaml_path', type=str, default=None)
        parser.add_argument('--available_langs', type=str, default=None)
        parser.add_argument('--vad_model_path', type=str, default=None)
        parser.add_argument('--diarize_model_path', type=str, default=None)
        parser.add_argument('--dns_model_dir', type=str, default=None)
        parser.add_argument('--dns_mos_threshold', type=float, default=None)
        parser.add_argument('--precision', type=str, default="float16")
        parser.add_argument('--language', type=str, default=None)
        parser.add_argument('--task', type=str, default="transcribe")
        # parser.add_argument('--initial_prompt', type=str, default="以下是一段语音记录。")
        parser.add_argument('--initial_prompt', type=str, default=None)
        parser.add_argument('--suppress_numerals', action="store_true")
        parser.add_argument('--target_sr', type=int, default=16_000)
        parser.add_argument('--target_format', type=str, default="wav")
        parser.add_argument('--chunk_size', type=int, default=30)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--num_threads', type=int, default=0)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--end_punctuation_list', type=str, default=None) #。,？,！
        parser.add_argument('--device', type=str, default="cuda")

        parser.add_argument('--output_info', action="store_true")
        parser.add_argument('--output_transcription', action="store_true")
        parser.add_argument('--output_segment', action="store_true")
        parser.add_argument('--output_audio', action="store_true")
        parser.add_argument('--output_valid_tsv', action="store_true")
        parser.add_argument('--only_reserve_1_speaker', action="store_true")
        parser.add_argument('--segment_interval_threshold', type=float, default=-1.0)
        parser.add_argument('--filter_by_aligned_model', action="store_true")

        parser.add_argument('--error_log_dir', type=str, default=None)
        parser.add_argument('--gpu_list', type=str, default="0") # "0,1,2"
        parser.add_argument('--input_audio_path_list', type=str, default=None)
        parser.add_argument('--input_audio_dir', type=str, default=None)
        parser.add_argument('--input_scp_path', type=str, default=None)
        parser.add_argument('--input_audio_root', type=str, default=None)
        parser.add_argument('--source_format', type=str, default="wav")
        parser.add_argument('--output_root', type=str, default=None)
        parser.add_argument('--resume_from_log', type=str, default=None)
        parser.add_argument('--test_times', type=int, default=-1)
        parser.add_argument('--skip_existed_info', action="store_true")
        parser.add_argument('--reverse_audio_path_list', action="store_true")
        parser.add_argument('--audio_clip_backend', type=str, default="moivepy")
        return parser
    
    @staticmethod
    def multiprocess(
        args: Optional[dict] = None,
    ):
        if args is None:
            parser = AutomaticSpeechRecognitionPipeline.init_parser()
            args = parser.parse_args()
        rank = int(os.environ['LOCAL_RANK'])
        threads_num = int(os.environ['WORLD_SIZE'])
        logger.info("rank {}/{}.".format(rank, threads_num))

        if args.input_audio_path_list is None:
            if args.input_audio_dir is not None:
                input_audio_name_list = os.listdir(args.input_audio_dir)
                input_audio_path_list = sorted([os.path.join(args.input_audio_dir, input_audio_name) for input_audio_name in input_audio_name_list])
            elif args.input_scp_path is not None:
                input_audio_id_list, input_audio_path_list = load_scp(args.input_scp_path)
            elif args.input_audio_root is not None and args.output_root is not None:
                input_lines, split_dict = load_audio_list_by_split(
                    input_audio_root=args.input_audio_root,
                    input_audio_ext_list=args.source_format.split(","),
                    test_times=args.test_times,
                )
                input_audio_path_list = [input_line.split("\t")[1] for input_line in input_lines]
                input_audio_path_list = sorted(input_audio_path_list)
            elif args.resume_from_log is not None:
                input_audio_path_list, _ = resume_from_log(args.resume_from_log)
            else:
                raise ValueError()

        if args.reverse_audio_path_list:
            input_audio_path_list = input_audio_path_list[::-1]
            
        global_recs = input_audio_path_list
        if getattr(args, "test_times", -1) > 0:
            global_recs = global_recs[:getattr(args, "test_times")]
        local_recs = global_recs[rank::threads_num]
        gpu_list = args.gpu_list.split(",")
        local_gpu_id = gpu_list[rank % len(gpu_list)]
        logger.info(f"cuda_available = {torch.cuda.is_available()}")
        available_langs = args.available_langs
        if available_langs is not None:
            available_langs = available_langs.split(",")
        pipeline_config = AutomaticSpeechRecognitionConfig(
            model_type=args.model_type,
            model_dir=args.model_dir,
            model_size=args.model_size,
            lang2align_model_yaml_path=args.lang2align_model_yaml_path,
            available_langs=available_langs,
            vad_model_path=args.vad_model_path,
            diarize_model_path=args.diarize_model_path,
            dns_model_dir=args.dns_model_dir,
            precision=args.precision,
            language=args.language,
            task=args.task,
            initial_prompt=args.initial_prompt,
            suppress_numerals=args.suppress_numerals,
            target_sr=args.target_sr,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            num_workers=args.num_workers,
            end_punctuation_list=args.end_punctuation_list,
            device=f"cuda:{local_gpu_id}",
        )
        pipeline = AutomaticSpeechRecognitionPipeline.build_pipeline(config=pipeline_config)
        for i, input_audio_path in tqdm.tqdm(enumerate(local_recs), total=len(local_recs), desc=f"rank {rank}/{threads_num} at cuda:{local_gpu_id}."):
            if True:
            # try:
                audio_dir = os.path.split(os.path.dirname(input_audio_path))[1]
                pipeline(
                    input_audio_path=input_audio_path,
                    audio_dir_name=audio_dir,
                    output_root=args.output_root,
                    output_info=args.output_info,
                    skip_existed_info=args.skip_existed_info,
                    output_transcription=args.output_transcription,
                    output_segment=args.output_segment,
                    output_audio=args.output_audio,
                    audio_clip_backend=args.audio_clip_backend,
                    output_valid_tsv=args.output_valid_tsv,
                    only_reserve_1_speaker=args.only_reserve_1_speaker,
                    segment_interval_threshold=args.segment_interval_threshold,
                    filter_by_aligned_model=args.filter_by_aligned_model,
                    dns_mos_threshold=args.dns_mos_threshold,
                )
            else:
            # except:
                logger.warning(f"failed at {input_audio_path}, rank {rank}/{threads_num} at cuda:{local_gpu_id}.")
                if args.error_log_dir is not None:
                    os.makedirs(args.error_log_dir, exist_ok=True)
                    with open(os.path.join(args.error_log_dir, f"{rank}.txt"), mode="a+") as f:
                        f.write(f"{input_audio_path} -> {args.output_root}\n")
        logger.info("{}/{}: Complete {} records.".format(rank, threads_num, len(local_recs)))


class WhisperxPipeline(AutomaticSpeechRecognitionPipeline):
    language_code_list = [
        "af", "am", "ar", "as", "az", 
        "ba", "be", "bg", "bn", "bo", 
        "br", "bs", "ca", "cs", "cy", 
        "da", "de", "el", "en", "es", 
        "et", "eu", "fa", "fi", "fo", 
        "fr", "gl", "gu", "ha", "haw", 
        "he", "hi", "hr", "ht", "hu", 
        "hy", "id", "is", "it", "ja", 
        "jw", "ka", "kk", "km", "kn", 
        "ko", "la", "lb", "ln", "lo", 
        "lt", "lv", "mg", "mi", "mk", 
        "ml", "mn", "mr", "ms", "mt", 
        "my", "ne", "nl", "nn", "no", 
        "oc", "pa", "pl", "ps", "pt", 
        "ro", "ru", "sa", "sd", "si", 
        "sk", "sl", "sn", "so", "sq", 
        "sr", "su", "sv", "sw", "ta", 
        "te", "tg", "th", "tk", "tl", 
        "tr", "tt", "uk", "ur", "uz", 
        "vi", "yi", "yo", "zh", "yue",
        None,
    ]

    @staticmethod
    def load_whisperx_model(config, device, device_index: Optional[int] = None, **kwargs):
        if device == "cpu":
            torch_device = "cpu"
        elif device == "cuda":
            if device_index is not None:
                torch_device = f"cuda:{device_index}"
            else:
                torch_device = "cuda"
        vad_model = pyannote.audio.Model.from_pretrained(config.vad_model_path)
        hyperparameters = {
            "onset": 0.500, 
            "offset": 0.363,
            "min_duration_on": 0.1,
            "min_duration_off": 0.1
        }
        vad_pipeline = whisperx.vad.VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(torch_device))
        # vad_pipeline.to(torch.device(torch_device))
        vad_pipeline.instantiate(hyperparameters)

        asr_options = dict(
            initial_prompt=config.initial_prompt,
            suppress_numerals=config.suppress_numerals,
        )
        whisperx_model = whisperx.load_model(
            whisper_arch=config.model_dir if config.model_dir is not None else config.model_size,
            vad_model=vad_pipeline,
            device=device,
            device_index=device_index,
            compute_type=config.precision,
            asr_options=asr_options,
            threads=config.num_threads,
        )
        
        filter_strategy = {}
        if config.lang2align_model_yaml_path is not None:
            with open(config.lang2align_model_yaml_path, mode="r+") as f:
                lang2align_model_dict = yaml.load(f, Loader=yaml.FullLoader)
            align_model_dict = {}
            meta_data_dict = {}
            for lang, align_model_info in lang2align_model_dict.items():
                if lang not in config.available_langs:
                    continue
                # print(666, lang, align_model_info)
                model_a, metadata = load_align_model(
                    language_code=lang, device=torch_device,
                    model_name=align_model_info["model_name"], model_dir=align_model_info["model_dir"],
                    pipeline_type=align_model_info["pipeline_type"],
                )
                align_model_dict[lang] = model_a
                meta_data_dict[lang] = metadata
                filter_strategy[lang] = dict(
                    interval=align_model_info.get("interval", -1.0),
                    confidence=align_model_info.get("confidence", -1.0),
                )
        else:
            align_model_dict = None
            meta_data_dict = None

        diarize_model = None
        if config.diarize_model_path is not None:
            diarize_model = whisperx.DiarizationPipeline(model_name=config.diarize_model_path, device=torch.device(torch_device))
        
        if config.dns_model_dir is not None:
            dns_mos_model = ComputeScore.load_model(model_dir=config.dns_model_dir)
        else:
            dns_mos_model = None
        
        return dict(
            whisperx=whisperx_model,
            align_model_dict=align_model_dict,
            meta_data_dict=meta_data_dict,
            filter_strategy=filter_strategy,
            diarize_model=diarize_model,
            dns_mos=dns_mos_model,
            device=torch_device,
        )

    def load_model(self, **kwargs):
        device = self.config.device
        device_index = None
        if device == "cpu":
            pass
        elif device == "cuda":
            pass
        elif device.startswith("cuda:"):
            device, device_index = device.split(":")
            device_index = int(device_index)
        else:
            device = "cpu"
        model = self.load_whisperx_model(self.config, device, device_index)
        return model

    def forward(
        self,
        input_audio_path: str,
        audio_dir_name: Optional[str] = None,
        output_root: Optional[str] = None,
        output_info: Optional[bool] = None,
        skip_existed_info: Optional[bool] = None,
        output_transcription: Optional[bool] = None,
        output_segment: Optional[bool] = None,
        output_audio: Optional[bool] = None,
        audio_clip_backend: Optional[str] = "moivepy",

        output_valid_tsv: Optional[bool] = None,
        only_reserve_1_speaker: Optional[bool] = None,
        segment_interval_threshold: Optional[float] = -1.0,
        filter_by_aligned_model: Optional[bool] = None,
        dns_mos_threshold: Optional[float] = None,
        align_words: Optional[bool] = None,
        **kwargs
    ):
        # print(input_audio_path)
        audio = whisperx.load_audio(input_audio_path)
        audio_name = os.path.splitext(os.path.split(input_audio_path)[1])[0]
        result = self.model["whisperx"].transcribe(
            audio=audio,
            task=self.config.task,
            language=self.config.language,
            batch_size=self.config.batch_size,
            chunk_size=self.config.chunk_size,
            num_workers=self.config.num_workers,
        )
        language = result["language"]
        if align_words is None:
            if self.model["align_model_dict"] is not None and language in self.model["align_model_dict"]:
                align_words = True
            else:
                align_words = False
        if align_words:
            _result = whisperx.align(result["segments"], self.model["align_model_dict"][language], self.model["meta_data_dict"][language], audio, self.model["device"], return_char_alignments=False)
            for key in result.keys():
                if key not in _result:
                    _result[key] = result[key]
            result = _result

        for i in range(len(result["segments"])):
            result["segments"][i]["index"] = i

        if "diarize_model" in self.model:
            diarize_segments = self.model["diarize_model"](audio)
            language = result["language"]
            result = whisperx.assign_word_speakers(diarize_segments, result)
            if only_reserve_1_speaker:
                segment_idx_list = return_segment_less_equal_than_n_speakers(result["segments"], diarize_segments, 1)
                segments = [result["segments"][idx] for idx in segment_idx_list]
                result["segments"] = segments
        else:
            diarize_segments = None

        audio_file_clip = None
        if audio_clip_backend == "moivepy":
            audio_file_clip = AudioFileClip(str(input_audio_path))
        if align_words and filter_by_aligned_model:
            lang_confidence_dict = {lang: values["confidence"] for lang, values in self.model["filter_strategy"].items()}
            l = result['segments']
            new_l = []
            starts = []
            sids = []
            n = 0
            for seg in l:
                r = [[]]
                if n > 0:
                    r[0].append(float(l[n-1]['end']))
                s = None
                for word in seg['words']:
                    if 'start' in word.keys():
                        if s == None:
                        # if s is None:
                            s = float(word['start'])
                        else:
                            ns = float(word['start'])
                            if ns - s > self.model["filter_strategy"][language]["interval"]:
                                starts.append(ns)
                                r.append([])
                            s = ns
                    else:
                        s = None
                    r[-1].append(word)
                if n+1 < len(l):
                    r[-1].append(float(l[n+1]['start']))
                    ## 以上两行为解决多切到后面句子的开头
                for _segments in r:
                    d = list_to_segment(_segments, language, lang_confidence_dict=lang_confidence_dict)
                    if d:
                        new_l.append(d)
                n += 1
            result['segments'] = new_l
            for i in range(len(result["segments"])):
                result["segments"][i]["index"] = i

        while output_root is not None:
            if audio_dir_name is None:
                output_root = os.path.join(output_root, result["language"], audio_name)
            else:
                output_root = os.path.join(output_root, result["language"], audio_dir_name, audio_name)
            os.makedirs(output_root, exist_ok=True)
            if output_valid_tsv and segment_interval_threshold > 0:
                start_list = diarize_segments["start"].tolist()
                end_list = diarize_segments["end"].tolist()
                speaker_list = diarize_segments["speaker"].tolist()
                valid_idx_list = return_segment_interval_larger_than_threshold(
                    segments=result["segments"],
                    diarize_segments=diarize_segments,
                    # threshold=2.0,
                    threshold=segment_interval_threshold,
                )
                result["segments"] = [result["segments"][idx] for idx in valid_idx_list]
            valid_list = []
            for segment in result["segments"]:
                # print(666, segment.keys())
                seg_index = segment["index"]
                speaker = segment.get("speaker", None)
                if speaker is None:
                    output_dir = output_root
                else:
                    output_dir = os.path.join(output_root, speaker)
                if output_audio:
                    output_audio_path = os.path.join(output_dir, f"{seg_index}.{self.config.target_format}")
                    if audio_file_clip is not None:
                        audio = audio_file_clip
                    else:
                        audio = AudioFileClip(str(input_audio_path))
                    # cut = audio.subclip(segment["start"], segment["end"])
                    cut = audio.subclip(min(max(0, segment["start"]), audio.duration), max(0, min(segment["end"], audio.duration)))
                    desired_fs = cut.fps
                    audio_array = cut.to_soundarray()
                    if self.model["dns_mos"] is not None:
                        dnsmos_score = self.model["dns_mos"](audio_array, desired_fs)
                        dnsmos_score = float(dnsmos_score)
                        segment["dns_mos"] = dnsmos_score
                        # print(type(dnsmos_score), dnsmos_score)
                        if dns_mos_threshold is not None and dnsmos_score < dns_mos_threshold:
                            continue
                    os.makedirs(output_dir, exist_ok=True)
                    cut.write_audiofile(output_audio_path, verbose=False, logger=None)
                os.makedirs(output_dir, exist_ok=True)
                if output_transcription:
                    output_transcription_path = os.path.join(output_dir, f"{seg_index}.txt")
                    with open(output_transcription_path, mode="w+") as f:
                        f.write(segment["text"])
                if output_segment:
                    output_segment_path = os.path.join(output_dir, f"{seg_index}.json")
                    segment["segment_id"] = seg_index
                    segment["language"] = result["language"]
                    with open(output_segment_path, mode="w+") as f:
                        json.dump(segment, f, ensure_ascii=False)
                if output_valid_tsv:
                    if True:
                        flag = is_segment_less_equal_than_n_speakers(
                            start_time=segment["start"],
                            end_time=segment["end"],
                            start_list=start_list,
                            end_list=end_list,
                            speaker_list=speaker_list,
                            num_speakers=1,
                        )
                        if not flag:
                            # print(f"666 seg_index = {seg_index}, invalid")
                            continue
                    valid_list.append([seg_index, output_audio_path, segment["text"]])
            if output_valid_tsv:
                columns = ["segment_index", "audio_path", "transcript"]
                df = pd.DataFrame(data=valid_list, columns=columns)
                df.set_index(keys="segment_index", inplace=True)
                valid_tsv_path = os.path.join(output_root, f"valid.tsv")
                # print(f"666 valid_tsv_path = {valid_tsv_path}")
                df.to_csv
            if output_info:
                output_seg_path = os.path.join(output_root, f"segments.json")
                if skip_existed_info and os.path.exists(output_seg_path):
                        break
                with open(output_seg_path, mode="w+") as f:
                    json.dump(result, f, ensure_ascii=False)
                if diarize_segments is not None:
                    output_diarize_seg_path = os.path.join(output_root, f"diarize_segments.csv")
                    diarize_segments.to_csv(output_diarize_seg_path, sep="\t")
            break
        return {
            "segments": result,
            "diarize_segments": diarize_segments,
        }


if __name__ == "__main__":
    AutomaticSpeechRecognitionPipeline.multiprocess()
