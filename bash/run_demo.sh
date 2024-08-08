#! /bin/bash
# ps -aux | grep '[p]ipeline' | awk '{print $2}' | xargs kill -9

model_type=whisperx
model_dir=/mnt/northcn3/cbu-tts/checkpoint/faster_whisper/faster-whisper-large-v3
model_size=large-v3
lang2align_model_yaml_path=/mnt/yuyin1/cbu-tts/model/ML_TTS_Dataset/examples/bash/asr_whisperx/lang2align_model.yaml
# available_langs=en,zh
available_langs=zh
vad_model_path=/mnt/yuyin1/cbu-tts/checkpoint/whisperx/vad/pytorch_model.bin
diarize_model_path=/mnt/yuyin1/cbu-tts/checkpoint/pyannote/speaker-diarization-3.1/config.yaml
dns_model_dir=/mnt/yuyin1/cbu-tts/checkpoint/DNSMOS/DNSMOS
dns_mos_threshold=3.8
precision=float16
task=transcribe
initial_prompt=以下是一段语音记录。
# initial_prompt="The following is a record of speech."
target_sr=16000
target_format=wav
audio_clip_backend=moivepy
chunk_size=12
batch_size=8
num_threads=6
num_workers=0

device=cuda
# gpu_list=0,1,2,3,4
gpu_list=1,2,3

input_audio_root=/mnt/yuyin1/cbu-tts/data_process/demo/noise_suppression
output_root=/mnt/yuyin1/cbu-tts/data_process/demo/asr
# input_audio_root=/mnt/yuyin1/cbu-tts/data_process/demo/noise_suppression_small
# output_root=/mnt/yuyin1/cbu-tts/data_process/demo/asr_small
error_log_dir=/mnt/yuyin1/cbu-tts/data_process/demo/log/asr
end_punctuation_list=。,.

echo 'input:' $input_audio_root
echo 'output:' $output_root

master_port=2300

# test_times=2
# n_process=2
# output_root=/mnt/yuyin1/cbu-tts/data_process/audio_test/${split}
test_times=-1
n_process=3



cd /mnt/yuyin1/cbu-tts/model/Data_Process_Piple/ML_TTS_Dataset/ML_TTS_Dataset/preprocess/asr

# --batch_size $batch_size --num_threads $num_threads --num_workers $num_workers 
# --initial_prompt $initial_prompt
# --skip_existed_info
# --output_valid_tsv --only_reserve_1_speaker --segment_interval_threshold 1.5 \
# --filter_by_aligned_model
torchrun --nproc_per_node=$n_process --master_port=$master_port whisperx_pipeline.py \
    --model_type $model_type --model_dir $model_dir --model_size $model_size \
    --lang2align_model_yaml_path $lang2align_model_yaml_path --available_langs $available_langs \
    --vad_model_path $vad_model_path --diarize_model_path $diarize_model_path \
    --dns_model_dir $dns_model_dir --dns_mos_threshold $dns_mos_threshold \
    --precision $precision --batch_size $batch_size --num_threads $num_threads --num_workers $num_workers \
    --task $task --end_punctuation_list $end_punctuation_list --initial_prompt $initial_prompt --suppress_numerals \
    --target_sr $target_sr --target_format $target_format \
    --chunk_size $chunk_size \
    --device $device --gpu_list $gpu_list \
    --test_times $test_times \
    --error_log_dir $error_log_dir \
    --input_audio_root $input_audio_root --output_root $output_root \
    --source_format wav --target_format wav --audio_clip_backend $audio_clip_backend \
    --output_info --output_transcription --output_segment --output_audio --filter_by_aligned_model
    
