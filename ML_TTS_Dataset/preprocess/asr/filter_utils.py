import bisect
import pandas as pd
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union


def find_time_interval_idx(
    start_time: float,
    end_time: float,
    start_list: Sequence = [],
    end_list: Sequence = [],
):
    start_idx = bisect.bisect_right(start_list, start_time) - 1
    while 0 <= start_idx < len(start_list) and start_time > end_list[start_idx]:
        start_idx += 1
    end_idx = bisect.bisect_left(end_list, end_time)
    while 0 <= end_idx < len(end_list) and end_time < start_list[end_idx]:
        end_idx -= 1
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(end_list) - 1)
    if start_idx > end_idx:
        start_idx = end_idx
    return start_idx, end_idx


def is_segment_interval_larger_than_threshold(
    start_time: float,
    end_time: float,
    start_list: Sequence = [],
    end_list: Sequence = [],
    threshold: Optional[float] = -1.0,
    **kwargs
):
    start_idx, end_idx = find_time_interval_idx(start_time, end_time, start_list, end_list)
    interval_list = []
    for i in range(start_idx + 1, end_idx + 1):
        interval = start_list[i] - end_list[i - 1]
        if 0.0 < threshold <= interval:
            return False
    return True


def return_segment_interval_larger_than_threshold(
    segments: Sequence,
    diarize_segments: pd.DataFrame,
    threshold: Optional[float] = -1.0,
    **kwargs,
):
    valid_idx_list = []
    start_list = diarize_segments["start"].tolist()
    end_list = diarize_segments["end"].tolist()
    for idx, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        if is_segment_interval_larger_than_threshold(start_time, end_time, start_list, end_list, threshold):
            valid_idx_list.append(idx)
    return valid_idx_list


def is_segment_less_equal_than_n_speakers(
    start_time: float,
    end_time: float,
    start_list: Sequence = [],
    end_list: Sequence = [],
    speaker_list: Sequence = [],
    num_speakers: Optional[int] = 1,
    **kwargs
):
    start_idx, end_idx = find_time_interval_idx(start_time, end_time, start_list, end_list)
    speakers = []
    for i in range(start_idx, end_idx + 1):
        speaker = speaker_list[i]
        speakers.append(speaker)
    # print(666, start_time, end_time, start_list[start_idx], end_list[end_idx], speakers)
    return len(set(speakers)) <= num_speakers


def return_segment_less_equal_than_n_speakers(
    segments: Sequence,
    diarize_segments: pd.DataFrame,
    num_speakers: Optional[int] = 1,
    **kwargs,
):
    valid_idx_list = []
    start_list = diarize_segments["start"].tolist()
    end_list = diarize_segments["end"].tolist()
    speaker_list = diarize_segments["speaker"].tolist()
    for idx, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        if is_segment_less_equal_than_n_speakers(start_time, end_time, start_list, end_list, speaker_list, num_speakers):
            valid_idx_list.append(idx)
    return valid_idx_list


# align with wav2vec2, which will take more time
def list_to_segment_v1(
    segment_list: Sequence,
    language: Optional[str] = "en",
    lang_confidence_dict: Optional[Dict] = {},
):
    """
    params:
        lang_confidence_dict:
            # 对于每种语言初步识别的结果，平均单词置信度低于多少则进行过滤，可调整改动
    """
    if lang_confidence_dict is None:
        lang_confidence_dict = dict(
            en=0.70,
            zh=0.95,
        )
    score = 0
    n = 0
    # print(type(segment_list), segment_list)
    for word in segment_list:
        if 'score' in word.keys():
            score += word['score']
            n += 1
    score = score / max(n, 1)
    if score < lang_confidence_dict[language]:
        return None
    sid = None
    for word in segment_list:
        if 'speaker' in word.keys():
            if sid == None:
                sid = word['speaker']
            # elif sid != word['speaker']:
            #     return None
    # if sid == None:
    #     return None
    try:
        if segment_list[0]['start'] > segment_list[-1]['end']:
            return None
        # print("seg", segment_list[0]['start'], segment_list[-1]['end'])
        # start = segment_list[0]['start']
        # end = segment_list[-1]['end']
        start = segment_list[0]['end'] - 0.5
        end = segment_list[-1]['start'] + 0.5
        if start > end:
            return None

        speaker = sid
        text = ''
        for word in segment_list:
            if language == 'zh':
                text += word['word']
            else:
                text += word['word'] + ' '
        print(score, n, lang_confidence_dict[language], start, end, segment_list)
        return {'start': start, 'end': end, 'text': text, 'speaker': speaker, "score": score, 'words': segment_list}
    except:
        return None


def list_to_segment(mylist, language='en', **kwargs):
    punctuation_list = [
        '。', '，', '、', '；', '：', '？', '！', '‘', '’', '“', '”',
        '《', '》', '（', '）', '——', '……', '-', #以上为中文标点符号
    ]
    next_start = float('inf')
    if isinstance(mylist[-1], float):
        next_start = mylist[-1]
        mylist.pop()
        ## 以上四行为解决多切到后面句子的开头，next_start在后文中涉及到的使用也是为了这个目的
    last_end = 0
    if isinstance(mylist[0], float):
        last_end = mylist[0]
        mylist.pop(0)
        ## 以上四行为解决多切到后面句子的开头，next_start在后文中涉及到的使用也是为了这个目的
    try:
        text = ''
        for word in mylist:
            if language == 'zh':
                text += word['word']
            else:
                text += word['word']+' '    
    except:
        return None
    mylist = [item for item in mylist if item['word'] not in punctuation_list]
    score_line = { 
        'en': 0.70,
        'zh': 0.95
    }# 这里代表每种语言，对于初步识别的结果，平均单词置信度低于多少则进行过滤，可调整改动
    last_score_line = {
        'en': 0.50,
        'zh': 0.88
    }
    begin_score_line = last_score_line
    min_score_line = {
        'en': 0.30,
        'zh': 0.70
    }
    score = 0
    n = 0
    last_score = 0
    last_n = 0
    min_score = 1
    if len(mylist)==1:
        return None
    for word in mylist:
        if 'score' in word.keys():
            if word['score'] < min_score:
                min_score = word['score']
            score += word['score']
            n += 1
    score = score / max(n,1) 
    if score < score_line[language]:
        return None
    if min_score < min_score_line[language]:
        return None
    for word in reversed(mylist):
        if 'score' in word.keys():
            last_score += word['score']
            last_n += 1
        if last_n >= 3:
            break
    last_score = last_score/max(last_n,1)
    if last_score < last_score_line[language]:
        return None
    
    begin_score = 0
    begin_n = 0
    for word in (mylist):
        if 'score' in word.keys():
            begin_score += word['score']
            begin_n += 1
        if begin_n >= 3:
            break
        
    begin_score = begin_score/max(begin_n,1)
    if begin_score < begin_score_line[language]:
        return None
    sid = None
    for word in mylist:
        if 'speaker' in word.keys():
            if sid == None:
                sid = word['speaker']
            elif sid != word['speaker']:
                return None
    if sid == None:
        return None
    try:
        if mylist[0]['start'] > mylist[-1]['end']:
            return None
        start = max(mylist[0]['end']-0.5,last_end)
        end = min(mylist[-1]['start']+0.5, next_start) ## 改行min的添加为解决多切到后面句子的开头
        if start > end:
            return None
        speaker = sid

        return {'start':start, 'end':end, 'text':text,'score':score, 'speaker': speaker, 'audio_path':None, 'words':mylist}
    except:
        return None 
