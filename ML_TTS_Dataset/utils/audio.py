import os, tqdm, subprocess
from moviepy.editor import AudioFileClip
import librosa
import numpy as np
import pydub
from pydub import AudioSegment
from collections import OrderedDict
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union


def split_audio(input_audio_path, start_seconds, end_seconds, target_sr, output_audio_path, **kwargs):
    backend = kwargs.get("backend", "moviepy")
    if backend == "moviepy":
        return split_audio_moviepy(input_audio_path, start_seconds, end_seconds, target_sr, output_audio_path, **kwargs)
    elif backend == "ffmpeg":
        return split_audio_ffmpeg(input_audio_path, start_seconds, end_seconds, target_sr, output_audio_path, **kwargs)
    return NotImplementedError()


def split_audio_ffmpeg(input_audio_path, start_seconds, end_seconds, target_sr, output_audio_path, **kwargs):
    """使用ffmpeg按给定的开始和结束时间戳切分视频。"""
    command = [
        'ffmpeg',
        '-loglevel', 'quiet',
        '-y',  # 自动覆盖输出文件
        '-i', input_audio_path,  # 输入文件
        '-ss', str(start_seconds),  # 切分开始时间
        '-to', str(end_seconds),  # 切分结束时间
        # '-c:a', 'pcm_s16le',  # 设置音频编码为PCM 16位小端格式
        '-ar', str(target_sr),
        '-ac', '1',  # 设置音频通道数为1
        '-vn',  # 不处理视频流
        output_audio_path  # 输出文件名
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def split_audio_moviepy(input_audio_path, start_seconds, end_seconds, target_sr, output_audio_path, **kwargs):
    audio_file_clip = kwargs.get("audio_file_clip", None)
    if audio_file_clip is not None:
        audio = audio_file_clip
    else:
        audio = AudioFileClip(str(input_audio_path))
    cut = audio.subclip(start_seconds, end_seconds)
    cut.write_audiofile(output_audio_path, verbose=False, logger=None)


def load_scp(scp_path: str):
    audio_id_list, audio_path_list = [], []
    with open(scp_path, mode="r+") as f:
        for line in f.readlines():
            line = line.strip()
            audio_id, audio_path = line.split("\t")
            audio_id_list.append(audio_id)
            audio_path_list.append(audio_path)
    return audio_id_list, audio_path_list


def load_audio_list_by_split(
    input_audio_root: str,
    input_audio_ext_list: Union[str, Sequence[str]],
    test_times: Optional[int] = -1,
):
    splits = os.listdir(input_audio_root)
    split_dict = OrderedDict()
    for split in splits:
        split_dict[split] = []
    input_lines = []
    tot = 0
    for root, dirs, files in tqdm.tqdm(os.walk(input_audio_root), desc=f"wakling in {input_audio_root}"):
        for file in files:
            for input_audio_ext in input_audio_ext_list:
                if file.endswith(input_audio_ext):
                    file_path = os.path.join(root, file)
                    speech_id = os.path.splitext(file)[0]
                    input_line = f"{speech_id}\t{file_path}"
                    input_lines.append(input_line)
                    _split = None
                    for split in splits:
                        if file_path.startswith(os.path.join(input_audio_root, split)):
                            _split = split
                            break
                    if _split is None:
                        raise ValueError(f"split {split} is not existed!")
                    split_dict[_split].append(file_path)
                    tot += 1
                    break
            if 0 < test_times <= tot:
                break
        if 0 < test_times <= tot:
            break
    return input_lines, split_dict


def create_scp_from_dir(
    input_audio_root: str,
    input_audio_ext_list: Union[str, Sequence[str]],
    output_audio_root: str,
    input_scp_path: Optional[str] = None,
    output_scp_path: Optional[str] = None,
    output_audio_ext: Optional[str] = None,
    test_times: Optional[int] = -1,
):
    if isinstance(input_audio_ext_list, str):
        input_audio_ext_list = [input_audio_ext_list]
        if output_audio_ext is None:
            output_audio_ext = input_audio_ext_list[0]
    input_lines = []
    output_lines = []
    tot = 0
    for root, dirs, files in tqdm.tqdm(os.walk(input_audio_root), desc=f"wakling in {input_audio_root}"):
        for file in files:
            for input_audio_ext in input_audio_ext_list:
                if file.endswith(input_audio_ext):
                    file_path = os.path.join(root, file)
                    speech_id = os.path.splitext(file)[0]
                    input_line = f"{speech_id}\t{file_path}"
                    input_lines.append(input_line)
                    output_line = f"{speech_id}\t{file_path.replace(input_audio_root, output_audio_root)[:-len(input_audio_ext)] + output_audio_ext}"
                    output_lines.append(output_line)
                    tot += 1
                    break
        if 0 < test_times <= tot:
            break
    if input_scp_path is not None:
        os.makedirs(os.path.dirname(input_scp_path), exist_ok=True)
        with open(input_scp_path, mode="w+") as f:
            f.write("\n".join(input_lines))
    if output_scp_path is not None:
        os.makedirs(os.path.dirname(output_scp_path), exist_ok=True)
        with open(output_scp_path, mode="w+") as f:
            f.write("\n".join(output_lines))
    return input_lines, output_lines


def get_output_audio_path(
    input_audio_path: str,
    input_audio_root: str,
    output_audio_root: str,
    make_output_audio_dir: Optional[bool] = False,
):
    output_audio_path = input_audio_path.replace(input_audio_root, output_audio_root)
    if make_output_audio_dir:
        output_audio_dir = os.path.dirname(output_audio_path)
        os.makedirs(output_audio_dir, exist_ok=True)
    return output_audio_path


def pydub_to_np(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


if __name__ == "__main__":
    subset = "cn-comedy-podcasts"
    input_audio_root = f"/mnt/northcn3/cbu-tts/data_ori/Chinese/podcast/{subset}"
    audio_ext = ".m4a"
    input_scp_path = f"/mnt/guiyang1/cbu-tts/data_process/scp/Chinese/podcast/{subset}/input.scp"
    output_audio_root = f"/mnt/guiyang1/cbu-tts/data_process/resample/Chinese/podcast/{subset}"
    output_scp_path = f"/mnt/guiyang1/cbu-tts/data_process/scp/Chinese/podcast/{subset}/output.scp"
    create_scp_from_dir(
        input_audio_root=input_audio_root,
        audio_ext=audio_ext,
        input_scp_path=input_scp_path,
        output_audio_root=output_audio_root,
        output_scp_path=output_scp_path,
        # test_times=10,
    )
