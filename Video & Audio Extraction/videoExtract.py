import os
import subprocess

# 视频帧提取，将视频以一秒为单位转换成照片
def extract_frames(videos_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg',
        '-i', videos_path,
        "-vf", f"fps={fps}",
        f"{output_dir}/%06d.jpg",
    ]
    subprocess.run(cmd, check=True)
    print("帧提取完成")

# 音频转存为.wav 文件
def extract_audio(videos_path, output_audio_path):
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    cmd = [
        'ffmpeg',
        '-i', videos_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_audio_path
    ]
    subprocess.run(cmd, check=True)
    print("音频提取完成")

# 预处理视频
def preprocess_video(video_path):
    extract_frames(video_path, "frames")
    extract_audio(video_path, "audio/audio.wav")

