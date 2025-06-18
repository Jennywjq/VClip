import os
import subprocess
import sys

def extract_frames(video_path, output_dir="frames", fps=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f" 视频文件不存在：{video_path}")

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        f"{output_dir}/%06d.jpg"
    ]
    subprocess.run(cmd, check=True)
    print(f" 帧提取完成，保存目录：{output_dir}")

def extract_audio(video_path, output_audio_path="audio/audio.wav"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f" 视频文件不存在：{video_path}")

    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",                # 标准无损音频采集格式 （PCM）
        "-ar", "44100",                        # 采样率 44100khz
        "-ac", "2",                            # 双声道（立体声）
        output_audio_path
    ]
    subprocess.run(cmd, check=True)
    print(f" 音频提取完成，输出路径：{output_audio_path}")

def preprocess_video(video_path, frame_dir="frames", audio_path="audio/audio.wav"):
    print(f"开始预处理：{video_path}")
    extract_frames(video_path, frame_dir)
    extract_audio(video_path, audio_path)
    print("视频预处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python videoExtract.py your_video.mp4")      # “your_video.mp4” 替换为实际文件名称。
    else:
        preprocess_video(sys.argv[1])
