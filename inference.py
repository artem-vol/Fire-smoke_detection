import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def process_video_with_tracking(model, input_video_path, show_video=True, save_video=False,
                                output_video_path="output_video.mp4"):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get input video frame rate and dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()  # read the next frame
        if not ret:  # Exit if frame is not read
            break

        # Start bounding box tracking on current frame
        results = model.track(frame, iou=0.1, conf=0.0, persist=True, imgsz=768, verbose=False, tracker="bytetrack.yaml")

        annotator = Annotator(frame, line_width=2, example=str(model.model.names))

        # Draw bounding boxes and labels for tracked objects
        if results[0].boxes.id is not None:  # this will ensure that id is not None -> exist tracks
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy().astype(float)

            for box, cls, cnf in zip(boxes, classes, confs):
                if cls == 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 165, 255)
                label = f"{model.names[cls]} {round(cnf, 2)}"
                annotator.box_label((box[0], box[1], box[2], box[3]), label, color)

        if save_video:  # Write annotated frame to output video
            out.write(frame)

        if show_video:  # Show annotated frame to output video
            cv2.imshow('1', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # The condition of closing a window
            break

    # Release the input video capture and output video writer
    cap.release()
    if save_video:
        out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Example usage:
nn_model = YOLO("runs/weights/best.pt")
nn_model.fuse()
process_video_with_tracking(nn_model, "test_videos/car_fire.mp4", show_video=True, save_video=False)
