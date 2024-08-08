import os, logging
import warnings
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union
warnings.filterwarnings("ignore")

def get_logger(fpath=None, log_level=logging.INFO):
    formatter = logging.Formatter(
        f"[{os.uname()[1].split('.')[0]}]"
        f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.basicConfig(
        level=log_level,
        format=f"[{os.uname()[1].split('.')[0]}]"
               f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("Pyobj, f")
    if fpath is not None:
        # Dump log to file
        fh = logging.FileHandler(fpath)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def gather_log_files(log_dir: str):
    pass

def resume_from_log_file(log_file_path: str, split_symbol: Optional[str] = " -> "):
    input_audio_path_list, output_audio_path_list = [], []
    with open(log_file_path, mode="r+") as f:
        for line in f.readlines():
            line = line.strip()
            input_audio_path, output_audio_path = line.split(split_symbol, 1)
            input_audio_path.strip()
            output_audio_path.strip()
            input_audio_path_list.append(input_audio_path)
            output_audio_path_list.append(output_audio_path)
    return input_audio_path_list, output_audio_path_list


def resume_from_log(log_path: str, split_symbol: Optional[str] = " -> "):
    if os.path.isfile(log_path):
        return resume_from_log_file(log_file_path=log_path, split_symbol=split_symbol)
