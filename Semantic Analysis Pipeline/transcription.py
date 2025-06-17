import os
import subprocess
from datetime import timedelta
import whisper
import torch
import json

def format_seconds_to_hms(seconds: float) -> str:
    """将浮点数秒转换为 'HH:MM:SS.ms' 格式的字符串"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def process_video_to_transcript(video_path: str, output_audio_name: str, output_json_name: str):
    """从视频中提取音频，再提取文字。"""

    print(f" step 1: 从视频 '{video_path}' 中提取音频...")

    # 使用您指定的输出音频文件名
    extracted_audio_path = output_audio_name

    command = [
        'ffmpeg', '-i', video_path, '-y', '-vn',
        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        extracted_audio_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f" 音频提取成功 -> '{extracted_audio_path}'")
    except subprocess.CalledProcessError as e:
        print(f" 错误: 音频提取失败。FFmpeg 返回错误:\n{e.stderr}")
        return

    print("\n step 2: 开始音频转写...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = whisper.load_model("medium", device=device)
    print("Whisper 'medium' 模型加载完毕。")

    print(f"正在转写音频文件...")
    result = model.transcribe(extracted_audio_path, language='Chinese', word_timestamps=True)
    print(" 音频转写完成。")

    formatted_segments = []
    for segment in result['segments']:
        formatted_segments.append({
            "start_time": format_seconds_to_hms(segment['start']), 
            "end_time": format_seconds_to_hms(segment['end']),     
            "text": segment['text'].strip()
        })

    # 保存为 JSON 文件
    with open(output_json_name, 'w', encoding='utf-8') as f:
        json.dump(formatted_segments, f, ensure_ascii=False, indent=4)

    print("\n 任务全部完成！")
    print("最终产出文件如下:")
    print(f"  1. 音频文件: {output_audio_name}")
    print(f"  2. 文稿文件: {output_json_name}")


video_file_name = "interview.mp4"       # <--- 根据实际视频名称改
output_audio_name = "extracted_audio.wav" # <--- 可改
output_json_name = "transcript.json"      # <--- 可改

colab_video_path = f"/{video_file_name}" # <--- 根绝实际视频路径改

# 检查视频文件是否存在
if os.path.exists(colab_video_path):
    process_video_to_transcript(colab_video_path, output_audio_name, output_json_name)
else:
    print(f"错误：找不到视频文件 '{colab_video_path}'")
