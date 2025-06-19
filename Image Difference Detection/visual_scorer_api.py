
import os
import json
import dashscope
from dashscope import MultiModalConversation

dashscope.api_key = "sk-0a0eefabc3f9421399d0f5981904326b"

FRAME_DIR = "frames"
SEGMENT_FILE = "segments/scene_changes.json"
OUTPUT_FILE = "score/visual_scores.json"
API_ENABLED = True

PROMPT = (
    "请判断以下画面中人物的情绪强度（0-1）、画面吸引力（0-1）和信息密度（0-1），"
    "并说明是否有字幕、关键人物或图表。输出格式为：emotion=数值, impact=数值, info=数值"
)

def call_qwen_vl(image_path, prompt):
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        print(f"Image not found: {abs_path}")
        return "emotion=0.0, impact=0.0, info=0.0"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": abs_path}
        ]
    }]

    try:
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages
        )
        print(" Raw response:", response)
        if response and "output" in response and "choices" in response["output"]:
            content_list = response["output"]["choices"][0]["message"]["content"]
            if isinstance(content_list, list):
                content_text = "".join([part.get("text", "") for part in content_list])
            else:
                content_text = str(content_list)
            return content_text
        else:
            return "emotion=0.0, impact=0.0, info=0.0"

    except Exception as e:
        print(" API调用异常：", e)
        return "emotion=0.0, impact=0.0, info=0.0"



def extract_scores_from_text(text):
    import re
    pattern = r"emotion=([0-9.]+).*impact=([0-9.]+).*info=([0-9.]+)"
    match = re.search(pattern, text.replace(" ", ""))
    if match:
        return {
            "emotion_score": float(match.group(1)),
            "impact_score": float(match.group(2)),
            "info_density": float(match.group(3)),
        }
    return {}

def timedelta_str_to_seconds(tstr):
    t = tstr.split(":")
    return int(t[0]) * 3600 + int(t[1]) * 60 + float(t[2])

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(SEGMENT_FILE, "r") as f:
        segments = json.load(f)

    frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])
    results = []

    for segment in segments:
        start_sec = int(timedelta_str_to_seconds(segment["start"]))
        end_sec = int(timedelta_str_to_seconds(segment["end"]))
        middle_idx = (start_sec + end_sec) // 2
        if middle_idx >= len(frame_files):
            middle_idx = len(frame_files) - 1
        frame_path = os.path.join(FRAME_DIR, frame_files[middle_idx])

        response_text = call_qwen_vl(frame_path, PROMPT)
        scores = extract_scores_from_text(response_text)
        scores.update({
            "start": segment["start"],
            "end": segment["end"],
            "total_visual_score": round(
                0.4 * scores.get("emotion_score", 0) +
                0.4 * scores.get("impact_score", 0) +
                0.2 * scores.get("info_density", 0), 3)
        })
        results.append(scores)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f" Visual scores saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
