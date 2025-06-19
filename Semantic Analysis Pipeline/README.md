# 🎵 VClip - Semantic Analysis Pipeline

This project provides a **xxx-stage pipeline** for processing video content.  

---

## ⚙️ Setup

Install all necessary Python packages using the `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🧠 Stage 1: Audio Transcription (transcription.py)

‼️ Core Task: It uses the Whisper model to transcribe audio from a video file.

📌 Note: Please manually update the video path inside the script or provide it as input.

🎯 What it does:
Extracts audio from your video
Transcribes the audio using Whisper
Generates a timestamped transcript stored in the output/ folder

📝 Output: A structured .json file with accurate timestamps and corresponding transcribed text.

```bash
python transcription.py
```

---


### 🧠 Stage 2: Initial Segmentation (segment_text.py)

‼️ Core Task: It reads the transcript from Stage 1 and uses the DeepSeek model to segment it into logical and semantically meaningful paragraphs.

📌 Note:Set up your own DeepSeek API key. Modify the script to include your API configuration.

📝 Output: JSON output showing how many segments were created, including start and end timestamps for each paragraph.

```bash
python segment_text.py
```

---

### 🧠 Stage 3: Highlight Scoring

‼️ Core Task ：This stage analyzes each semantic paragraph generated in Stage 2 and scores it across three dimensions: Emotion, Keyword Density and Golden Quote Detection.

#### 🪄 1. Emotion Analysis (analyze_emotion.py)

The model used is the Erlangshen series RoBERTa model: IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment.
Compared to the standard BERT model, it has been trained on a more diverse range of data types, including a large volume of social media-style text. This makes it more capable of handling internet slang and flexible sentence structures commonly found in TV show or variety show reviews.
(However, it may still misinterpret certain sarcastic or passive-aggressive expressions.)


Logic: Load the model → Input the text → Output the emotion (e.g., positive/negative) and confidence score → Convert the result into a numerical score (e.g., +1 for positive, -1 for negative).


#### 🪄 2. Keyword Density Analysis (analyze_keywords.py)

Two-Stage Refinement (TF-IDF + DeepSeek)

Initial Screening (by TF-IDF):
We use the TF-IDF algorithm as an efficient “pre-selection” tool to quickly generate a shortlist of 5–10 candidate keywords from the paragraph.

Final Review (by DeepSeek):
Then, we feed the original paragraph along with the candidate list to DeepSeek, which acts as a final judge.
Leveraging its deep language understanding, it selects the most essential and irreplaceable keywords from the shortlist.

#### 🪄 3. Golden Quote Detection (analyze_golden_quote)

Uses a targeted prompt to determine whether a paragraph contains a "Golden Quote"—a line that is insightful, emotionally powerful, or easily shareable.

Logic :Use a clearly defined prompt with the DeepSeek API → input the paragraph text → obtain the scoring result.

