import os
import sys
import json
from datetime import timedelta

import statistics 

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


def select_dynamic_highlights(all_scored_segments, min_duration, std_dev_factor, max_cap):
    """根据一组动态规则来筛选高光片段。"""

    # 规则 1: 过滤掉时长过短的片段
    long_enough_segments = []
    for seg in all_scored_segments:
        start_sec = timedelta_to_seconds(seg['start'])
        end_sec = timedelta_to_seconds(seg['end'])
        if (end_sec - start_sec) >= min_duration:
            long_enough_segments.append(seg)

    print(f"  -> 规则1 (时长过滤): {len(all_scored_segments)} 个片段中，有 {len(long_enough_segments)} 个时长超过 {min_duration} 秒。")

    if not long_enough_segments:
        print("  -> 没有足够长的片段可选，流程终止。")
        return []

    # 规则 2: 只选择分数足够高的精英片段
    scores = [seg['total_score'] for seg in long_enough_segments]
    if len(scores) > 1:
        mean_score = statistics.mean(scores)
        stdev_score = statistics.stdev(scores)
        score_threshold = mean_score + std_dev_factor * stdev_score
    else:
        # 如果只有一个片段，它就是唯一的选择
        score_threshold = scores[0] * 0.99

    elite_segments = [
        seg for seg in long_enough_segments if seg['total_score'] >= score_threshold
    ]

    print(f"  -> 规则2 (分数过滤): 平均分 {mean_score:.2f}，标准差 {stdev_score:.2f}，筛选门槛 {score_threshold:.2f}。")
    print(f"     有 {len(elite_segments)} 个精英片段脱颖而出。")

    # 规则 3: 应用数量上限
    final_selection = elite_segments[:max_cap]
    print(f"  -> 规则3 (数量上限): 最终选定 {len(final_selection)} 个高光片段进行导出。")

    return final_selection



def align_boundaries_to_semantics(highlight_clips, semantic_segments):
    """寻找与视觉高光重叠度最高的完整语义片段，并采用其边界。"""
    
    aligned_clips = []

    # 如果没有语义片段，无法对齐，直接返回原始片段
    if not semantic_segments:
        print("  -> 警告: 未找到任何语义片段，跳过边界对齐。")
        return highlight_clips

    for clip in highlight_clips:
        clip_start_sec = timedelta_to_seconds(clip['start'])
        clip_end_sec = timedelta_to_seconds(clip['end'])
        
        best_match_segment = None
        max_overlap = -1

        # 遍历所有句子，找到和当前高光片段重叠时间最长的那一句
        for sem_seg in semantic_segments:
            sem_start_sec = timedelta_to_seconds(sem_seg['start_time'])
            sem_end_sec = timedelta_to_seconds(sem_seg['end_time'])
            
            # 计算重叠时长
            overlap_start = max(clip_start_sec, sem_start_sec)
            overlap_end = min(clip_end_sec, sem_end_sec)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_match_segment = sem_seg
        
        # 如果找到了最佳匹配的句子，就采用它的边界
        if best_match_segment:
            new_start_sec = timedelta_to_seconds(best_match_segment['start_time'])
            new_end_sec = timedelta_to_seconds(best_match_segment['end_time'])
            
            new_start_str = str(timedelta(seconds=new_start_sec)).split('.')[0] + '.' + f"{int((new_start_sec % 1) * 1000):03d}"
            new_end_str = str(timedelta(seconds=new_end_sec)).split('.')[0] + '.' + f"{int((new_end_sec % 1) * 1000):03d}"
            
            aligned_clip = clip.copy()
            aligned_clip['start'] = new_start_str
            aligned_clip['end'] = new_end_str
            aligned_clip['original_start'] = clip['start']
            aligned_clip['original_end'] = clip['end']
            aligned_clip['aligned_text'] = best_match_segment['paragraph_text']
            
            aligned_clips.append(aligned_clip)
            print(f"  -> 片段 [{clip['start']} -> {clip['end']}]")
            print(f"     最大重叠句子为: '{best_match_segment['paragraph_text'][:20]}...'")
            print(f"     最终对齐为 -> [{new_start_str} -> {new_end_str}]")
        else:
            # 如果完全没有重叠的句子，保留原始片段
            aligned_clips.append(clip)
            print(f"  -> 片段 [{clip['start']} -> {clip['end']}] 未找到重叠句子，保留原始边界。")

    return aligned_clips


# --- 主函数 ---
def main():
    # 全局配置
    VIDEO_INPUT_PATH = "/Users/xiaohei/Documents/2025intern/interview.mp4" # <--- 【重要】请将这里替换为你的视频文件路径
    DEEPSEEK_API_KEY = "sk-984f91a660ca40ab9427e513a97f67ca" 
    QWEN_API_KEY = "sk-0a0eefabc3f9421399d0f5981904326b"

    # --- 动态高光片段设置 ---
    MIN_CLIP_DURATION = 10.0  # (秒) 任何高光片段都不能短于这个时长
    SCORE_STD_DEV_FACTOR = 0.5 # 筛选分数线：只保留高于“平均分 + 0.5个标准差”的片段。
    MAX_CLIPS_CAP = 7        # 就算精英片段很多，最终输出的片段数量也不要超过这个上限

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

    print("\n========== 阶段 2: 视觉与文本分析 (并行) ==========")
    print("--- 视觉分析流 ---")
    detect_scene_changes(FRAMES_DIR, SCENE_SEGMENTS_PATH)
    run_visual_scoring_pipeline(api_key=QWEN_API_KEY, frame_dir=FRAMES_DIR, segment_file=SCENE_SEGMENTS_PATH, output_file=VISUAL_SCORES_PATH)
    
    print("\n--- 文本分析流 ---")
    semantic_segment_final(TRANSCRIPT_PATH, SEMANTIC_SEGMENTS_PATH, api_key=DEEPSEEK_API_KEY)
    run_scoring_pipeline(SEMANTIC_SEGMENTS_PATH, TEXT_SCORES_PATH, api_key=DEEPSEEK_API_KEY)

    print("\n========== 阶段 3: 核心分数融合 ==========")
    all_segments_path = fuse_scores(VISUAL_SCORES_PATH, TEXT_SCORES_PATH, COMBINED_SCORES_PATH)
    
    with open(all_segments_path, 'r', encoding='utf-8') as f:
        all_scored_segments = json.load(f)

    print("\n========== 阶段 4: 动态筛选高光片段 ==========")
    candidate_highlight_segments = select_dynamic_highlights(
        all_scored_segments=all_scored_segments,
        min_duration=MIN_CLIP_DURATION,
        std_dev_factor=SCORE_STD_DEV_FACTOR,
        max_cap=MAX_CLIPS_CAP
    )

    # 如果在筛选后没有任何片段，就提前结束
    if not candidate_highlight_segments:
        print("未能根据筛选规则找到任何合适的高光片段。")
        print("\n========== 所有任务已完成！ ==========")
        return

    
    print("\n========== 阶段 5: 语义边界对齐 ==========")
    try:
        with open(SEMANTIC_SEGMENTS_PATH, 'r', encoding='utf-8') as f:
            semantic_segments = json.load(f)
        
        # 调用边界对齐函数
        aligned_final_clips = align_boundaries_to_semantics(
            candidate_highlight_segments, 
            semantic_segments
        )
    except FileNotFoundError:
        print(f"警告：找不到语义分段文件 {SEMANTIC_SEGMENTS_PATH}，将跳过边界对齐步骤。")
        aligned_final_clips = candidate_highlight_segments

    print("\n========== 阶段 6: 导出高光片段与智能解释 ==========")
    export_final_clips(
        segments_to_export=aligned_final_clips, 
        original_video_path=VIDEO_INPUT_PATH,
        frames_dir=FRAMES_DIR,
        output_dir=HIGHLIGHTS_DIR,
        qwen_api_key=QWEN_API_KEY
    )

    print("\n========== 所有任务已完成！ ==========")

if __name__ == "__main__":
    main()
