import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import json
import time
import requests
import statistics 

# -----------------------------------------------------------------------------
# SECTION 1: 初始化与配置
# -----------------------------------------------------------------------------

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

# -----------------------------------------------------------------------------
# SECTION 2: 核心工作流函数 (分析 -> 平滑 -> 决策 -> 后处理 -> 执行)
# -----------------------------------------------------------------------------

def analyze_video_scenes(input_path: str, analysis_interval, main_face_area_threshold):
    """【第一步：分析】(已调整主角脸阈值)生成一份更丰富、包含主角/配角信息的场景报告。"""
    print(f"分析阶段: 开始快速分析视频场景 (间隔: {analysis_interval}s) -> {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("错误: 无法打开视频。")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("错误: 无法读取视频的FPS，将使用默认值30。")
        fps = 30

    scenes = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps

        if frame_count % int(fps * analysis_interval) == 0:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = results.detections if results.detections else []

            face_areas = [(d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height) for d in detections]
            main_face_count = sum(1 for area in face_areas if area > main_face_area_threshold)
            max_area = max(face_areas) if face_areas else 0.0

            scene_info = {
                "time": round(current_time, 2),
                "total_face_count": len(detections),
                "main_face_count": main_face_count,
                "max_face_area": round(max_area, 4)
            }
            scenes.append(scene_info)
            print(f"  - 时间点 {scene_info['time']}s: 检测到 {scene_info['total_face_count']} 张脸 (其中 {scene_info['main_face_count']} 个是主角)...")


        frame_count += 1

    cap.release()
    print("分析阶段: 完成！")
    return scenes

def smooth_scene_report(scenes: list, window_seconds):
    """【第二步：平滑处理】(已适配)使用滑动窗口平滑【主角脸数量】，消除抖动和毛刺。"""
    print(f"\n平滑处理阶段: 开始平滑分析报告 (窗口大小: {window_seconds}s)...")
    if not scenes:
        return []

    smoothed_scenes = json.loads(json.dumps(scenes))
    num_scenes = len(scenes)

    for i in range(num_scenes):
        current_time = scenes[i]['time']
        window_start_time = current_time - window_seconds / 2
        window_end_time = current_time + window_seconds / 2

        main_face_counts_in_window = []
        for j in range(num_scenes):
            if window_start_time <= scenes[j]['time'] <= window_end_time:
                main_face_counts_in_window.append(scenes[j]['main_face_count'])

        if main_face_counts_in_window:
            try:
                mode_main_face_count = statistics.mode(main_face_counts_in_window)
                if smoothed_scenes[i]['main_face_count'] != mode_main_face_count:
                    print(f"  - 平滑修正: 时间点 {scenes[i]['time']}s, 原始主角脸数 {scenes[i]['main_face_count']}, 修正为 -> {mode_main_face_count}")
                    smoothed_scenes[i]['main_face_count'] = mode_main_face_count
            except statistics.StatisticsError:
                pass

    print("平滑处理阶段: 完成！")
    return smoothed_scenes


def get_edit_plan_from_qwen(scenes_report: list):
    """
    【第三步：决策】(已升级)
    将【更丰富】的场景报告和【更明确的规则】发送给Qwen。
    """
    print("\n决策阶段: 正在生成增强型Prompt并请求Qwen API获取剪辑建议...")

    prompt_text = (
        "你是一位顶级的短视频剪辑师，擅长制作快节奏、抓人眼球的竖屏内容。\n"
        "请根据以下视频场景分析报告，为我制定一个9:16的竖屏剪辑计划。报告中包含了'主角脸数量'和'总脸数'，你的决策应主要基于'主角脸数量'。\n\n"
        "【剪辑规则】:\n"
        "1.  **单人特写优先 (CLOSE_UP):**\n"
        "    - 当'主角脸数量'为 1 时，这是**首选**。\n"
        "    - 即使'主角脸数量'可能因为检测误差而暂时为 0，但如果'总脸数'为 1，且持续一段时间，也应考虑给予特写，因为这很可能就是主角。\n"
        "    - 特写能突出核心人物的情绪和表达。\n"
        "2.  **双人分屏 (SPLIT_SCREEN):** 仅当'主角脸数量'稳定为 2 时使用。这适用于对话、互动或对峙场景。\n"
        "3.  **大场面/远景 (WIDE_SHOT_BLUR):** 当'主角脸数量'长时间为 0 (且'总脸数'也较少，说明是空旷场景) 或 '主角脸数量' > 2 (多人场景不适合特写或分屏) 时使用。这用来交代环境或展示宏大场面。\n\n"
        "【视频分析报告】:\n"
    )

    for scene in scenes_report:
        prompt_text += (
            f"- 时间点 {scene['time']}秒: 有 {scene['main_face_count']} 个主角脸 "
            f"(总共检测到 {scene['total_face_count']} 张脸), "
            f"最大脸面积占画面的 {scene['max_face_area']:.2%}。\n"
        )

    prompt_text += "\n请严格按照以下JSON格式返回你的剪辑计划 (Edit Decision List)，不要包含任何JSON格式之外的额外说明文字:\n{\"edit_plan\": [{\"start_time\": float, \"end_time\": float, \"shot_type\": \"SHOT_TYPE\", \"description\": \"你的剪辑理由(请简洁)\"}]}"
    
    api_key = "sk-0a0eefabc3f9421399d0f5981904326b" 
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "qwen-max", "input": {"prompt": prompt_text}, "parameters": {}}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        qwen_response_json = response.json()

        if "output" in qwen_response_json and "text" in qwen_response_json["output"]:
            generated_text = qwen_response_json["output"]["text"]
            print("决策阶段: 成功获取Qwen的剪辑建议！")
            if "```json" in generated_text:
                json_part = generated_text.split("```json")[1].split("```")[0].strip()
            else:
                json_part = generated_text[generated_text.find('{'):generated_text.rfind('}')+1]
            return json.loads(json_part)
        else:
            print(f"错误: Qwen返回的数据格式不正确。返回内容: {qwen_response_json}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"错误: API请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 解析Qwen返回的JSON失败: {e}. 返回的文本内容是: \n{generated_text}")
        return None


def refine_edit_plan(edit_plan: dict, min_duration):
    """【新增！第四步：决策后处理】(采纳建议3)优化AI返回的剪辑计划，合并过于零碎的片段，提升流畅度。"""

    print(f"\n决策后处理阶段: 开始优化剪辑计划 (最短片段时长: {min_duration}s)...")
    plan = edit_plan.get("edit_plan")
    if not plan or len(plan) < 2:
        print("决策后处理阶段: 无需优化。")
        return edit_plan

    refined_plan = []
    temp_plan = json.loads(json.dumps(plan)) 
    
    i = 0
    while i < len(temp_plan):
        current_clip = temp_plan[i]
        duration = current_clip["end_time"] - current_clip["start_time"]

        # 检查当前片段是否过短
        if duration < min_duration and i < len(temp_plan) - 1:
            next_clip = temp_plan[i+1]
            print(f"  - 发现短片段: 时长 {duration:.2f}s, 类型 {current_clip['shot_type']}. 尝试与下一片段合并...")
            # 将当前短片段的结束时间直接拉到下一个片段的结束时间，实现合并
            # 并将短片段的类型和描述更新为下一个片段的，以长的为主
            next_clip['start_time'] = current_clip['start_time']
            # 我们实际上是“跳过”了当前这个短片段，让下一个片段把它“吃掉”
            i += 1 
            continue
        
        refined_plan.append(current_clip)
        i += 1

    if not refined_plan:
        print("决策后处理阶段: 优化后计划为空，返回原始计划。")
        return edit_plan

    # 重新链接所有片段的时间戳，确保无缝连接
    for i in range(len(refined_plan) - 1):
        refined_plan[i]['end_time'] = refined_plan[i+1]['start_time']

    # 确保最后一个片段的结束时间与原始计划的最后一个片段一致
    if refined_plan and plan:
        refined_plan[-1]['end_time'] = plan[-1]['end_time']
    
    edit_plan["edit_plan"] = refined_plan
    print("决策后处理阶段: 完成！")
    return edit_plan


#def execute_edit_plan(input_path: str, output_path: str, edit_plan: dict):
#    """【第五步：执行】根据【优化后】的剪辑计划，逐帧生成最终视频。"""
#    print("\n执行阶段: 开始按照优化后的剪辑计划处理视频...")

#    plan = edit_plan.get("edit_plan")
#    if not plan:
#        print("错误: 剪辑计划无效。")
#        return

#    cap = cv2.VideoCapture(input_path)
#    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#    fps = cap.get(cv2.CAP_PROP_FPS)
#    if fps == 0: fps = 30

#    target_aspect_ratio = (9, 16)
#    target_w_ratio, target_h_ratio = target_aspect_ratio
#    target_width = original_width
#    target_height = int(target_width * target_h_ratio / target_w_ratio)

#    temp_silent_output_path = output_path.replace('.mp4', '_silent_temp.mp4')
#    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#    out = cv2.VideoWriter(temp_silent_output_path, fourcc, fps, (target_width, target_height))

#    frame_count = 0
#    plan_index = 0
#    last_known_crop_x_single, last_known_crop_x_split1, last_known_crop_x_split2 = None, None, None

#    while cap.isOpened():
#        ret, frame = cap.read()
#        if not ret: break

#        current_time = frame_count / fps

#        if plan_index + 1 < len(plan) and current_time >= plan[plan_index + 1]["start_time"]:
#            plan_index += 1
#            print(f"  - 时间点 {current_time:.2f}s: 切换到 -> {plan[plan_index]['shot_type']} (理由: {plan[plan_index]['description']})")

#        shot_type = plan[plan_index]["shot_type"]
#        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#        detections = results.detections if results.detections else []

#        if shot_type == "WIDE_SHOT_BLUR":
#            new_frame = create_blurred_background_frame(frame, target_width, target_height)
#            last_known_crop_x_single, last_known_crop_x_split1, last_known_crop_x_split2 = None, None, None

#        elif shot_type == "CLOSE_UP":
#            if detections:
#                main_subject = sorted(detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)[0]
#                new_frame, last_known_crop_x_single = create_pan_and_scan_frame(frame, main_subject, target_width, target_height, last_known_crop_x_single, 0.08)
#            else:
#                new_frame = crop_frame_at(frame, last_known_crop_x_single, target_width, target_height) if last_known_crop_x_single is not None else create_blurred_background_frame(frame, target_width, target_height)

#        elif shot_type == "SPLIT_SCREEN":
#            if len(detections) >= 2:
#                new_frame, last_known_crop_x_split1, last_known_crop_x_split2 = create_split_screen_frame(frame, detections, target_width, target_height, last_known_crop_x_split1, last_known_crop_x_split2, 0.08)
#            elif len(detections) == 1:
#                new_frame, last_known_crop_x_single = create_pan_and_scan_frame(frame, detections[0], target_width, target_height, last_known_crop_x_single, 0.08)
#            else:
#                new_frame = create_blurred_background_frame(frame, target_width, target_height)

#        else:
#            new_frame = create_blurred_background_frame(frame, target_width, target_height)
        
                 # ======================== 【【【 新增诊断代码 】】】 ========================
#        if new_frame is None:
#            print(f"错误：在时间点 {current_time:.2f}s, new_frame 为空！将写入黑帧。")
             # 创建一个黑色的帧来填充，防止程序崩溃
#            new_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

         # 检查帧的尺寸是否与 VideoWriter 期望的尺寸完全一致
#        if new_frame.shape[0] != target_height or new_frame.shape[1] != target_width:
#            print(f"警告：在时间点 {current_time:.2f}s, 帧尺寸不匹配！正在强制拉伸。")
#            print(f"  - 期望尺寸 (W, H): ({target_width}, {target_height})")
#            print(f"  - 实际尺寸 (W, H): ({new_frame.shape[1]}, {new_frame.shape[0]})")
             # 强制将帧缩放/拉伸到正确的尺寸，这是保证写入成功的关键
#            new_frame = cv2.resize(new_frame, (target_width, target_height))
         # =======================================================================
       
#        out.write(new_frame)
#        frame_count += 1

#    print("执行阶段: 画面处理完成。")
#    cap.release()
#    out.release()

#    print("\n最后一步: 正在合并音频...")
#    merge_audio(temp_silent_output_path, input_path, output_path)



def execute_edit_plan(input_path: str, output_path: str, edit_plan: dict):
    """【最终版：执行】根据剪辑计划，逐帧生成视频，并使用FFmpeg管道写入，绕过cv2.VideoWriter。"""
    print("\n执行阶段: 开始按照优化后的剪辑计划处理视频 (使用 FFmpeg 管道)...")

    plan = edit_plan.get("edit_plan")
    if not plan:
        print("错误: 剪辑计划无效。")
        return

    cap = cv2.VideoCapture(input_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    target_aspect_ratio = (9, 16)
    target_w_ratio, target_h_ratio = target_aspect_ratio
    target_width = original_width
    target_height = int(target_width * target_h_ratio / target_w_ratio)

    # 【【【 最终修复：确保宽高都为偶数 】】】
    # H.264 编码器 (尤其是使用 yuv420p 像素格式时) 要求视频的宽和高都必须是偶数。
    if target_width % 2 != 0:
        print(f"警告: 目标宽度 {target_width} 是奇数，自动减 1 以满足编码器要求。")
        target_width -= 1
    if target_height % 2 != 0:
        print(f"警告: 目标高度 {target_height} 是奇数，自动减 1 以满足编码器要求。")
        target_height -= 1
    
    temp_silent_output_path = output_path.replace('.mp4', '_silent_temp.mp4')

    # --- FFmpeg 子进程设置 ---
    # 我们将直接向 FFmpeg 的 stdin 管道写入原始视频帧 (rawvideo)
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{target_width}x{target_height}',  
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        temp_silent_output_path
    ]

    # 启动 FFmpeg 子进程
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_count = 0
    plan_index = 0
    last_known_crop_x_single, last_known_crop_x_split1, last_known_crop_x_split2 = None, None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps

        if plan_index + 1 < len(plan) and current_time >= plan[plan_index + 1]["start_time"]:
            plan_index += 1
            print(f"  - 时间点 {current_time:.2f}s: 切换到 -> {plan[plan_index]['shot_type']} (理由: {plan[plan_index]['description']})")

        shot_type = plan[plan_index]["shot_type"]
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = results.detections if results.detections else []

        # --- 生成帧的逻辑保持不变 ---
        if shot_type == "WIDE_SHOT_BLUR":
            new_frame = create_blurred_background_frame(frame, target_width, target_height)
            last_known_crop_x_single, last_known_crop_x_split1, last_known_crop_x_split2 = None, None, None
        elif shot_type == "CLOSE_UP":
            if detections:
                main_subject = sorted(detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)[0]
                new_frame, last_known_crop_x_single = create_pan_and_scan_frame(frame, main_subject, target_width, target_height, last_known_crop_x_single, 0.08)
            else:
                new_frame = crop_frame_at(frame, last_known_crop_x_single, target_width, target_height) if last_known_crop_x_single is not None else create_blurred_background_frame(frame, target_width, target_height)
        elif shot_type == "SPLIT_SCREEN":
            if len(detections) >= 2:
                new_frame, last_known_crop_x_split1, last_known_crop_x_split2 = create_split_screen_frame(frame, detections, target_width, target_height, last_known_crop_x_split1, last_known_crop_x_split2, 0.08)
            elif len(detections) == 1:
                new_frame, last_known_crop_x_single = create_pan_and_scan_frame(frame, detections[0], target_width, target_height, last_known_crop_x_single, 0.08)
            else:
                new_frame = create_blurred_background_frame(frame, target_width, target_height)
        else:
            new_frame = create_blurred_background_frame(frame, target_width, target_height)
        
        try:
            process.stdin.write(new_frame.tobytes())
        except (IOError, BrokenPipeError) as e:
            print(f"错误：写入 FFmpeg 管道失败: {e}")
            break
        
        frame_count += 1

    print("执行阶段: 画面处理完成，正在关闭FFmpeg管道...")
    cap.release()

    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"❌ 错误：FFmpeg 在处理视频帧时返回了错误。")
        print(f"FFmpeg Stderr:\n{stderr.decode('utf-8', errors='ignore')}")
    
    print("\n最后一步: 正在合并音频...")
    if os.path.exists(temp_silent_output_path) and os.path.getsize(temp_silent_output_path) > 0:
        merge_audio(temp_silent_output_path, input_path, output_path)
    else:
        print(f"❌ 错误：无声视频文件 '{temp_silent_output_path}' 未能成功生成，跳过音频合并。")


# -----------------------------------------------------------------------------
# SECTION 4: 辅助与渲染函数
# -----------------------------------------------------------------------------
def merge_audio(silent_video_path, original_video_path, final_output_path):
    try:
        command = ['ffmpeg', '-y', '-i', silent_video_path, '-i', original_video_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', final_output_path]
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"成功！最终文件已保存至: {final_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误：音频合并失败。FFmpeg返回了错误。")
        print(f"FFmpeg Stderr: {e.stderr}")
    except Exception as e:
        print(f"❌ 错误：音频合并失败。错误: {e}")
    finally:
        if os.path.exists(silent_video_path):
            os.remove(silent_video_path)

def get_top_two_faces(detections):
    detections.sort(key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)
    return detections[:2]

def calculate_target_crop_x(frame, detection, target_w, target_h):
    h, w, _ = frame.shape
    box_data = detection.location_data.relative_bounding_box
    center_x = box_data.xmin + box_data.width / 2
    crop_w = int(h * target_w / target_h)
    return int((center_x * w) - (crop_w / 2))

def crop_frame_at(frame, crop_x, target_w, target_h):
    h, w, _ = frame.shape
    crop_w = int(h * target_w / target_h)
    crop_x = max(0, min(crop_x, w - crop_w)) if crop_x is not None else (w - crop_w) // 2
    cropped_frame = frame[:, crop_x:crop_x+crop_w]
    return cv2.resize(cropped_frame, (target_w, target_h))

def create_blurred_background_frame(frame, target_w, target_h):
    bg = cv2.resize(frame, (target_w, target_h))
    bg = cv2.GaussianBlur(bg, (155, 155), 0)
    h, w, _ = frame.shape
    scale = target_w / w
    fg_h, fg_w = int(h * scale), target_w
    fg = cv2.resize(frame, (fg_w, fg_h))
    y_offset = (target_h - fg_h) // 2
    bg[y_offset:y_offset+fg_h, 0:target_w] = fg
    return bg

def create_pan_and_scan_frame(frame, detection, target_w, target_h, previous_crop_x, smoothing_factor):
    target_crop_x = calculate_target_crop_x(frame, detection, target_w, target_h)
    new_crop_x = int(previous_crop_x * (1.0 - smoothing_factor) + target_crop_x * smoothing_factor) if previous_crop_x is not None else target_crop_x
    final_frame = crop_frame_at(frame, new_crop_x, target_w, target_h)
    return final_frame, new_crop_x

def create_split_screen_frame(frame, detections, target_w, target_h, prev_x1, prev_x2, smoothing_factor):
    top_two = get_top_two_faces(detections)
    target1_x = calculate_target_crop_x(frame, top_two[0], target_w, target_h // 2)
    target2_x = calculate_target_crop_x(frame, top_two[1], target_w, target_h // 2)
    if prev_x1 is not None and prev_x2 is not None:
        if abs(target1_x - prev_x1) + abs(target2_x - prev_x2) > abs(target1_x - prev_x2) + abs(target2_x - prev_x1):
            target1_x, target2_x = target2_x, target1_x
    if target1_x > target2_x:
        target1_x, target2_x = target2_x, target1_x
    new_x1 = int(prev_x1 * (1.0 - smoothing_factor) + target1_x * smoothing_factor) if prev_x1 is not None else target1_x
    new_x2 = int(prev_x2 * (1.0 - smoothing_factor) + target2_x * smoothing_factor) if prev_x2 is not None else target2_x
    frame1 = crop_frame_at(frame, new_x1, target_w, target_h // 2)
    frame2 = crop_frame_at(frame, new_x2, target_w, target_h // 2)
#    return np.vstack((frame1, frame2)), new_x1, new_x2
    # 将两个半屏堆叠起来
    stacked_frame = np.vstack((frame1, frame2))
    
    # 【【【 这就是最终的、一劳永逸的修复方案 】】】
    # 这几行代码的作用就像一个“尺寸保险”。它在堆叠完两个半屏后，会再检查一次总尺寸。
    # 如果因为奇数问题导致有1个像素的偏差，它会用 cv2.resize 做一次极其微小的、
    # 肉眼无法察觉的拉伸，确保最终输出的帧和 VideoWriter 期望的尺寸完全一致。
    if stacked_frame.shape[0] != target_h or stacked_frame.shape[1] != target_w:
        stacked_frame = cv2.resize(stacked_frame, (target_w, target_h))
        
    return stacked_frame, new_x1, new_x2

def reframe_to_vertical_video(input_video_path: str, output_video_path: str):
    """
    接收一个视频输入路径，经过AI分析和处理，最终输出一个9:16的竖屏视频。
    这是一个完整的工作流，包含了分析、平滑、决策、优化和执行五个步骤。

    :param input_video_path: 输入的视频文件路径。
    :param output_video_path: 处理后输出的9:16视频文件路径。
    :return: 成功时返回输出路径，失败时返回None。
    """
    print("\n--- 开始执行智能竖屏转换工作流 ---")
    
    # 定义工作流中的参数
    analysis_interval = 0.25         # 视频分析间隔（秒）
    main_face_area_threshold = 0.008 # 主角脸部面积阈值
    smoothing_window = 1.5           # 平滑处理的时间窗口（秒）
    min_clip_duration = 1.5          # 最终剪辑计划的最短片段时长（秒）

    # 1. 分析视频
    raw_scenes_report = analyze_video_scenes(input_video_path, analysis_interval, main_face_area_threshold)
    
    # 2. 对报告进行平滑处理
    smoothed_report = smooth_scene_report(raw_scenes_report, window_seconds=smoothing_window)
    
    if not smoothed_report:
        print("错误: 无法生成场景报告，智能转换终止。")
        return None
        
    # 3. 从Qwen获取剪辑计划
    edit_plan = get_edit_plan_from_qwen(smoothed_report)
    
    if not edit_plan:
        print("错误: 无法从Qwen获取剪辑计划，智能转换终止。")
        return None

    # 4. 优化剪辑计划，合并过短片段
    refined_edit_plan = refine_edit_plan(edit_plan, min_duration=min_clip_duration)
        
    # 5. 严格按照优化后的计划执行剪辑
    execute_edit_plan(input_video_path, output_video_path, refined_edit_plan)

    print(f"--- 智能竖屏转换工作流完成 ---")

    # 检查输出文件是否存在，确保成功
    if os.path.exists(output_video_path):
        return output_video_path
    else:
        print(f"错误: 最终输出文件 {output_video_path} 未能成功生成。")
        return None
