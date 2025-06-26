import os
import json
from openai import OpenAI
import sys

def semantic_segment_final(transcript_path: str, output_path: str, api_key: str) -> bool:
    """使用 OpenAI 库调用 DeepSeek API 进行语义分段。"""

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文稿文件 '{transcript_path}'")
        return False
    
    formatted_text = ""

    for i, segment in enumerate(transcript_data):
        line_number = i + 1
        clean_text = segment['text'].replace("`", "").replace("#", "")
        formatted_text += f"{line_number}: {clean_text}\n"
     
    system_prompt = """
你是一位专业的视频内容分析师。你的任务是将以下按行号标记的视频文稿，切分成若干个在语义上连贯且完整的“内容段落”。

**核心规则**:
1.  每个段落应聚焦于一个独立的话题、一个完整的故事、或一个明确的问答对。
2.  段落之间应存在明显的逻辑分割点或话题转换。

**输出格式要求 (必须严格遵守)**:
1.  你的回答**必须是、且仅能是**一个完整的JSON数组（JSON Array）。
2.  整个输出**必须**以 `[` 开始，并以 `]` 结束。
3.  **绝对不能**在JSON数组的前后添加任何解释性文字、注释或Markdown代码块（例如 ```json ... ```）。

**示例格式**:
[
  {"start_line": 1, "end_line": 5},
  {"start_line": 6, "end_line": 12},
  {"start_line": 13, "end_line": 25}
]

请根据以上规则，处理我接下来提供的文稿。
"""
     
    print("正在连接 DeepSeek API...")
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com")
         
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_text},
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.1,
        )
        llm_output_str = response.choices[0].message.content
        print("API 请求成功，已收到模型返回的分段计划。")
         
    except Exception as e:
        print(f" 错误: 调用 API 失败: {e}")
        return False

    try:
        parsed_data = json.loads(llm_output_str)
        segmentation_plan = []
        if isinstance(parsed_data, list):
            segmentation_plan = parsed_data
        elif isinstance(parsed_data, dict):
            for key in parsed_data:
                if isinstance(parsed_data[key], list):
                    segmentation_plan = parsed_data[key]
                    break
        if not segmentation_plan:
            print(" 错误: 模型返回的数据格式不正确，无法找到分段列表。")
            print(f"收到的原始数据: {llm_output_str}")
            return False

        final_paragraphs = []
        for plan_item in segmentation_plan:
            start_line, end_line = plan_item['start_line'], plan_item['end_line']
            start_index, end_index = start_line - 1, end_line - 1
            if start_index < 0 or end_index >= len(transcript_data):
                continue
            paragraph_text = " ".join([transcript_data[i]['text'] for i in range(start_index, end_index + 1)])
            final_paragraphs.append({
                "start_time": transcript_data[start_index]['start_time'],
                "end_time": transcript_data[end_index]['end_time'],
                "paragraph_text": paragraph_text
            })

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir) and output_dir != '': 
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_paragraphs, f, ensure_ascii=False, indent=4)
         
        print(f"\n 全部完成！分段后的文稿已保存至: {output_path}")
        return True

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f" 错误: 解析或处理模型返回的数据时失败: {e}")
        print(f"收到的原始数据: {llm_output_str}")
        return False


if __name__ == "__main__":
    my_api_key = "sk-984f91a660ca40ab9427e513a97f67ca"   # <--- 可以改 API 密钥 

    google_drive_base_path = "/content/drive/My Drive/"    # <--- 改路径
    colab_output_folder = os.path.join(google_drive_base_path, "Colab_Output") 

    os.makedirs(colab_output_folder, exist_ok=True)

    input_transcript = os.path.join(colab_output_folder, "transcript.json") 
    output_segmented = os.path.join(colab_output_folder, "segmented_transcript.json")

    if os.path.exists(input_transcript):
        semantic_segment_final(
            transcript_path=input_transcript,
            output_path=output_segmented,
            api_key=my_api_key
        )
    else:
        print(f"错误: 找不到输入文件 '{input_transcript}'。请确保阶段一的脚本已成功运行，并且 'transcript.json' 文件存在于 Google Drive 的 '{colab_output_folder}' 路径下。")
