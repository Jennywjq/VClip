from transformers import pipeline
import torch

emotion_analyzer_pipeline = None

def initialize_emotion_model():
    global emotion_analyzer_pipeline
    if emotion_analyzer_pipeline is None:
        try:
            model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
          
            emotion_analyzer_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print(" 模型加载成功！")
        except Exception as e:
            print(f" 模型加载失败: {e}")


def analyze_emotion(text: str) -> float:
    """分析单段文本的情感，并返回一个从 -1.0 (非常消极) 到 +1.0 (非常积极) 的分数。"""
    if emotion_analyzer_pipeline is None:
        print(" 模型尚未初始化，请先调用 initialize_emotion_model()")
        return 0.0
    if not text or not isinstance(text, str):
        return 0.0

    try:
        result = emotion_analyzer_pipeline(text)[0]
        label = result['label']
        score = result['score']

        if 'positive' in label.lower() or 'pos' in label.lower():
            return score
        else:
            return -score

    except Exception as e:
        print(f"情感分析出错: {e}")
        return 0.0


if __name__ == "__main__":
    initialize_emotion_model()

    if emotion_analyzer_pipeline:
        text1 = "这部剧的男主角演技太神了，剧情也很紧凑，yyds！"
        # text2 = "等了一星期更新，结果剧情超拖戏，看到快睡着，好失望。"
        text2 = "嗯，只能说我这种普通观众的审美水平还是有限，暂时还欣赏不来这么先锋的艺术"
        text3 = "就这？就这？我还以为是什么惊天动地的神作呢，看来是我期待太高了。"

        emotion_score1 = analyze_emotion(text1)
        emotion_score2 = analyze_emotion(text2)
        emotion_score3 = analyze_emotion(text3)

        print(f"文本: '{text1}'\n情感分数: {emotion_score1:.4f}")
        print("-" * 20)
        print(f"文本: '{text2}'\n情感分数: {emotion_score2:.4f}")
        print("-" * 20)
        print(f"文本: '{text3}'\n情感分数: {emotion_score3:.4f}")
