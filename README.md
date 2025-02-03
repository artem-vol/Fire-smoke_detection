# Motion detection
![GitHub top language](https://img.shields.io/github/languages/top/artem-vol/Fire-smoke_detection)

Fire and smoke object detection, using YOLOV8

## Installation
```
git clone git@github.com:artem-vol/Fire-smoke_detection.git
```
## Usage
Run inference.py to perform fire and smoke detection on a video.
  
You can change the following parameters of ```process_video_with_tracking function```:
- **model** (YOLO): A YOLO model object for detecting and tracking objects. 
- **input_video_path** (str): Path to the input video file.
- **show_video** (bool, True by default): Whether to display the annotated video.
- **save_video** (bool, False by default): Whether to save the annotated video.
- **output_video_path** (str, default is "output_video.mp4"): The path to the output video file if save_video is set to True.

## Example

![output_video_1 - Trim](https://github.com/user-attachments/assets/5046d34b-aa57-4924-bda3-2725fef9039d)






