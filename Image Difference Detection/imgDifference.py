import os
import cv2
import json
from datetime import timedelta

def calc_hist(img1, img2):
    # 利用视频所提取的.jpg图片转化为hsv以比较直方图差异
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)     # 数值越低相似值越低差异越大

def detect_scene_changes(frame_dir, fps=1, threshold=0.8):
    frames_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frames_files:
        print("错误：frames 文件夹为空或不存在 .jpg 图片")
        return

    scene_segments = []
    last_scene_start = 0
    prev_image = None

    for idx, f in enumerate(frames_files):
        img_path = os.path.join(frame_dir, f)
        img = cv2.imread(img_path)
        if prev_image is not None:
            sim = calc_hist(prev_image, img)
            if sim < threshold:
                scene_segments.append({
                    "start": str(timedelta(seconds=last_scene_start)),
                    "end": str(timedelta(seconds=idx))
                })
                last_scene_start = idx
        prev_image = img

    scene_segments.append({
        "start": str(timedelta(seconds=last_scene_start)),
        "end": str(timedelta(seconds=len(frames_files))),
    })

    os.makedirs("segments", exist_ok=True)
    with open("segments/scene_changes.json", "w") as f:
        json.dump(scene_segments, f, indent=2, ensure_ascii=False)

    print(f"\n 场景切换检测完成，共生成 {len(scene_segments)} 个片段：")
    for i, seg in enumerate(scene_segments):
        print(f"片段 {i+1}: {seg['start']} --> {seg['end']}")

if __name__ == "__main__":
    detect_scene_changes("frames", fps=1, threshold=0.8)
