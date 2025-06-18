Image Difference Detection

imgDifference.py
This script analyzes visual differences between consecutive video frames and detects scene transitions based on histogram similarity. 
It's used to segment a video into visually distinct chunks (e.g., change of setting, camera angle, or subject).

Run videoExtract.py beforehand to extract frames from a video (e.g., 1 frame per second).
All frame images should be placed in a frames/ folder, named sequentially (e.g., 000001.jpg, 000002.jpg, etc.).

Each frame is converted to HSV color space.
A 2D histogram is computed for hue and saturation.
Similarity between two consecutive frames is measured using correlation-based histogram comparison.
When similarity drops below a defined threshold (default 0.8), a scene cut is recorded.

