import os
import json
import subprocess
from datetime import timedelta

from explanation_generator import generate_clip_explanation

def timedelta_to_seconds(time_str: str) -> float:
    """一个健壮的时间字符串转秒数的函数"""
    if '.' in time_str:
        parts = time_str.split('.')
        main_part = parts[0]
        ms_part = parts[1]
    else:
        main_part = time_str
        ms_part = "0"
        
    h, m, s = map(int, main_part.split(':'))
    return float(h * 3600 + m * 60 + s + float(f"0.{ms_part}"))

#def cut_clip(input_video, start_time, end_time, output_path):
#    cmd = ["ffmpeg", "-i", input_video, "-ss", str(start_time), "-to", str(end_time), "-c", "copy", output_path, "-y"]
#    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 使用这个新版本的 cut_clip 函数
def cut_clip(input_video, start_time, end_time, output_path):
    """使用ffmpeg剪切视频片段（重新编码以确保音视频同步）。"""
    cmd = [
        "ffmpeg",
        "-i", input_video,      # 输入文件
        "-ss", str(start_time),  # 开始时间
        "-to", str(end_time),    # 结束时间
        "-c:v", "libx264",       # 视频使用x264编码器
        "-c:a", "aac",           # 音频使用aac编码器
        "-strict", "experimental", # 兼容某些aac版本
        output_path,             # 输出文件
        "-y"                     # 覆盖已存在的文件
    ]
    # 使用DEVNULL来抑制ffmpeg的大量输出
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def find_representative_frame(clip_start_time, clip_end_time, frames_dir):
    start_sec = timedelta_to_seconds(clip_start_time)
    end_sec = timedelta_to_seconds(clip_end_time)
    middle_sec = start_sec + (end_sec - start_sec) * 0.33
    frame_index = int(middle_sec) + 1
    frame_filename = f"{frame_index:06d}.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)
    if not os.path.exists(frame_path):
        frame_index = int(start_sec) + 1
        frame_filename = f"{frame_index:06d}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
    return frame_path if os.path.exists(frame_path) else None

def export_final_clips(combined_scores_path, original_video_path, frames_dir, output_dir, top_k, qwen_api_key):
    if not os.path.exists(combined_scores_path):
        print(f"错误：找不到综合评分文件 {combined_scores_path}")
        return

    with open(combined_scores_path, "r", encoding='utf-8') as f:
        all_segments = json.load(f)

    top_segments = all_segments[:top_k]
    print(f"\n准备导出排名前 {len(top_segments)} 的高光片段...")

    for i, seg in enumerate(top_segments):
        clip_name = f"highlight_{i+1:03d}.mp4"
        clip_path = os.path.join(output_dir, clip_name)
        print(f"[{i+1}/{len(top_segments)}] 正在导出片段: {clip_name} (总分: {seg['total_score']})")
        
        try:
            cut_clip(original_video_path, seg["start"], seg["end"], clip_path)
            print(f" 视频剪辑成功: {clip_path}")
        except Exception as e:
            print(f"视频剪辑失败: {e}")
            continue

        rep_frame_path = find_representative_frame(seg["start"], seg["end"], frames_dir)
        if not rep_frame_path:
            print(" 警告: 未能找到该片段的代表帧，无法生成智能解释。")
            continue

        explanation = generate_clip_explanation(seg, rep_frame_path, qwen_api_key)
        
        explanation_file = os.path.join(output_dir, f"highlight_{i+1:03d}_explanation.json")
        with open(explanation_file, "w", encoding='utf-8') as f:
            json.dump(explanation, f, indent=2, ensure_ascii=False)
        print(f" 智能解释已保存: {explanation_file}")

    print(f"\n全部导出任务完成！请查看 '{output_dir}' 文件夹。")
