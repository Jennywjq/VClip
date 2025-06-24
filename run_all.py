import os
import sys
import json
from datetime import timedelta

sys.path.append(os.path.abspath('Video & Audio Extraction'))
sys.path.append(os.path.abspath('Image Difference Detection'))
sys.path.append(os.path.abspath('Semantic Analysis Pipeline'))

from videoExtract import extract_frames
from imgDifference import detect_scene_changes
from visual_scorer_api import run_visual_scoring_pipeline
from clip_export import export_final_clips 

from transcription import process_video_to_transcript
from segment_text import semantic_segment_final
from scoring_pipeline import run_scoring_pipeline

from explanation_generator import generate_clip_explanation 


def timedelta_to_seconds(time_str: str) -> float:
    if '.' in time_str:
        parts = time_str.split('.')
        main_part = parts[0]
        ms_part = parts[1]
    else:
        main_part = time_str
        ms_part = "0"
    h, m, s = map(int, main_part.split(':'))
    return float(h * 3600 + m * 60 + s + float(f"0.{ms_part}"))

#动态归一化方法，融合视觉与文本分数
def fuse_scores(visual_scores_path, text_scores_path, output_path, w_visual=0.6, w_text=0.4):
    with open(visual_scores_path, 'r', encoding='utf-8') as f:
        visual_segments = json.load(f)
    with open(text_scores_path, 'r', encoding='utf-8') as f:
        text_segments = json.load(f)

    all_text_scores = [seg.get('final_text_score', 0) for seg in text_segments]
    if not all_text_scores:
        print("警告: 文本分数列表为空。")
        return

    max_text_score = max(all_text_scores)
    min_text_score = min(all_text_scores)
    score_range = max_text_score - min_text_score
    

    # 处理所有分数都相同的特殊情况，避免除以零
    if score_range == 0:
        print("警告: 所有文本分数都相同，将所有文本归一化分数设为0.5。")

    combined_results = []
    
    for v_seg in visual_segments:
        v_start_sec = timedelta_to_seconds(v_seg['start'])
        v_end_sec = timedelta_to_seconds(v_seg['end'])
        v_score = v_seg.get('total_visual_score', 0)

        overlapping_raw_scores = []
        overlapping_text_info = []
        for t_seg in text_segments:
            t_start_sec = timedelta_to_seconds(t_seg['start_time'])
            t_end_sec = timedelta_to_seconds(t_seg['end_time'])
            
            if max(v_start_sec, t_start_sec) < min(v_end_sec, t_end_sec):
                overlapping_raw_scores.append(t_seg.get('final_text_score', 0))
                overlapping_text_info.append({
                    "text": t_seg.get('paragraph_text', ''),
                    "golden_quote": t_seg.get('analysis', {}).get('golden_quote_text', '')
                })

        # --- 动态归一化应用 ---
        if overlapping_raw_scores:
            avg_raw_text_score = sum(overlapping_raw_scores) / len(overlapping_raw_scores)
            if score_range == 0:
                normalized_text_score = 0.5 
            else:
                # 应用 Min-Max Scaling 公式
                normalized_text_score = (avg_raw_text_score - min_text_score) / score_range
        else:
            avg_raw_text_score = 0
            normalized_text_score = 0
            
        total_score = w_visual * v_score + w_text * normalized_text_score

        combined_results.append({
            "start": v_seg['start'],
            "end": v_seg['end'],
            "total_score": round(total_score, 4),
            "details": {
                "visual_score": v_score,
                "avg_text_score_raw": round(avg_raw_text_score, 4),
                "normalized_text_score": round(normalized_text_score, 4),
                "text_info": overlapping_text_info
            }
        })

    combined_results = sorted(combined_results, key=lambda x: x['total_score'], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    print(f"[融合模块] 分数融合完成，已保存至 {output_path}")
    return output_path


# --- 主函数 ---
def main():
    # 全局配置
    VIDEO_INPUT_PATH = "/Users/xiaohei/Documents/2025intern/interview.mp4" # <--- 【重要】请将这里替换为你的视频文件路径
    DEEPSEEK_API_KEY = "sk-984f91a660ca40ab9427e513a97f67ca" 
    QWEN_API_KEY = "sk-0a0eefabc3f9421399d0f5981904326b"
    TOP_K_CLIPS = 5 

    # 定义输出路径
    OUTPUT_BASE = "output"
    FRAMES_DIR = os.path.join(OUTPUT_BASE, "frames")
    AUDIO_PATH = os.path.join(OUTPUT_BASE, "audio", "extracted_audio.wav")
    TRANSCRIPT_PATH = os.path.join(OUTPUT_BASE, "segments", "transcript.json")
    SCENE_SEGMENTS_PATH = os.path.join(OUTPUT_BASE, "segments", "scene_segments.json")
    SEMANTIC_SEGMENTS_PATH = os.path.join(OUTPUT_BASE, "segments", "semantic_segments.json")
    VISUAL_SCORES_PATH = os.path.join(OUTPUT_BASE, "score", "visual_scores.json")
    TEXT_SCORES_PATH = os.path.join(OUTPUT_BASE, "score", "text_scores.json")
    COMBINED_SCORES_PATH = os.path.join(OUTPUT_BASE, "score", "combined_scores.json")
    HIGHLIGHTS_DIR = os.path.join(OUTPUT_BASE, "highlights")

    # 创建输出目录
    for path in [FRAMES_DIR, os.path.dirname(AUDIO_PATH), os.path.dirname(TRANSCRIPT_PATH), os.path.dirname(VISUAL_SCORES_PATH), HIGHLIGHTS_DIR]:
        os.makedirs(path, exist_ok=True)

    # 完整流水线开始
    print("========== 阶段 1: 视频预处理 ==========")
    extract_frames(VIDEO_INPUT_PATH, FRAMES_DIR)
    process_video_to_transcript(VIDEO_INPUT_PATH, AUDIO_PATH, TRANSCRIPT_PATH)

    print("\n========== 阶段 2: 视觉与文本分析 ==========")
    print("--- 视觉分析流 ---")
    detect_scene_changes(FRAMES_DIR, SCENE_SEGMENTS_PATH)
    run_visual_scoring_pipeline(api_key=QWEN_API_KEY, frame_dir=FRAMES_DIR, segment_file=SCENE_SEGMENTS_PATH, output_file=VISUAL_SCORES_PATH)
    
    print("\n--- 文本分析流 ---")
    semantic_segment_final(TRANSCRIPT_PATH, SEMANTIC_SEGMENTS_PATH, api_key=DEEPSEEK_API_KEY)
    run_scoring_pipeline(SEMANTIC_SEGMENTS_PATH, TEXT_SCORES_PATH, api_key=DEEPSEEK_API_KEY)

    print("\n========== 阶段 3: 核心分数融合 ==========")
    fuse_scores(VISUAL_SCORES_PATH, TEXT_SCORES_PATH, COMBINED_SCORES_PATH)

    print("\n========== 阶段 4: 导出高光片段与智能解释 ==========")
    export_final_clips(
        combined_scores_path=COMBINED_SCORES_PATH,
        original_video_path=VIDEO_INPUT_PATH,
        frames_dir=FRAMES_DIR,
        output_dir=HIGHLIGHTS_DIR,
        top_k=TOP_K_CLIPS,
        qwen_api_key=QWEN_API_KEY
    )

    print("\n========== 所有任务已完成！ ==========")

if __name__ == "__main__":
    main()
