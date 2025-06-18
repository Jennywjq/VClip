# 🎬 VClip - Semantic Analysis Pipeline

This project provides a **xxx-stage pipeline** for processing video content.  

---

## ⚙️ Setup

Install all necessary Python packages using the `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### 🧠 Stage 1: Audio Transcription

This stage is executed by the transcription.py script.

Core Task: It uses the Whisper model to transcribe audio from a video file.

📌 Note: Please manually update the video path inside the script or provide it as input.

🎯 What it does:

Extracts audio from your video

Transcribes the audio using Whisper

Generates a timestamped transcript stored in the output/ folder

📝 Output: A structured .json file with accurate timestamps and corresponding transcribed text.

```bash
python transcription.py
```


### ✂️ Stage 2: Initial Segmentation (初步分段)

This stage is executed by the segment_text.py script.

Core Task: It reads the transcript from Stage 1 and uses the DeepSeek model to segment it into logical and semantically meaningful paragraphs.

🔐 Prerequisite:

Set up your own DeepSeek API key. Modify the script to include your API configuration

📊 Parsed Result:

JSON output showing how many segments were created

Start and end timestamps for each paragraph

```bash
python segment_text.py
```
