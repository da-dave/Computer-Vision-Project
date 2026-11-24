# VeinyVizion
VeinyVizion is a prototype that uses near-infrared imaging and computer vision to highlight superficial veins in a live camera feed. It was built to make veins easier to see during blood draws.

## How it works
1. Capture near-infrared video. A modified $25 security camera captures near-infrared light. Veins appear darker than the surrounding skin in the resulting frames.
2. Highlight veins in real time. A custom computer vision model identifies and highlights veins in the feed. The project uses Python and OpenCV.
3. Display the results. The processed feed streams to a Meta Quest 3, where it can be viewed alongside the arm. A conversational Gemini interface lets users ask questions about the vein data.

## Technology stack
|Component | Technology|
|-----------|----------|
|Image capture |	Modified security camera and near-infrared imaging|
|Computer vision	| Python, OpenCV, and a custom vein-detection model|
|Backend |	Flask|
|Headset | display	Meta Quest 3, Unity, C#, and OpenXR|
|Conversational interface	| Google Gemini|

## What's next
- Align the highlighted vein map directly with the arm in the headset.
- Improve image quality with a higher-quality near-infrared camera.

## Team
Krish Iyer, Minh Nguyen, Calvin Zheng, Xavier Engelbrecht, Pradeep Nallipogu
