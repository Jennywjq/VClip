import os
import json
import subprocess

VIDEO_FILE = "input_video.mp4"  # 原始完整视频路径
SCORE_FILE = "score/combined_scores.json"  # 包含每段时间戳和分数
OUTPUT_DIR = "highlights"  # 输出目录（包含剪辑视频与说明）
TOP_K = 5  # 保留五个项目

def load_scores(score_file):
    with open(score_file, "r") as f:
        segments = json.load(f)

    # 按总分排序，保留前K段
    top_segments = sorted(segments, key=lambda x: x.get("total_score", 0), reverse=True)[:TOP_K]
    return top_segments

def cut_clip(input_video, start_time, end_time, output_path):
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ss", start_time,
        "-to", end_time,
        "-c", "copy",
        output_path,
        "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_placeholder_explanation(i, seg):
    return {
        "title": f"Highlight Clip {i+1}",
        "keywords": ["auto-generated"],
        "split_reason": "This clip was selected based on a high propagation score.",
        "score": seg.get("total_score", 0)
    }

def export_clips():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    segments = load_scores(SCORE_FILE)
    for i, seg in enumerate(segments):
        clip_name = f"clip_{i+1:03d}.mp4"
        clip_path = os.path.join(OUTPUT_DIR, clip_name)
        cut_clip(VIDEO_FILE, seg["start"], seg["end"], clip_path)
        print(f"已导出至  {clip_name}")

        # 生成解释说明文件
        explanation = generate_placeholder_explanation(i, seg)
        explanation_file = os.path.join(OUTPUT_DIR, f"clip_{i+1:03d}_explanation.json")
        with open(explanation_file, "w") as f:
            json.dump(explanation, f, indent=2, ensure_ascii=False)

    print(f" 已导出前 {TOP_K} 的高光至 '{OUTPUT_DIR}'")

if __name__ == "__main__":
    export_clips()
