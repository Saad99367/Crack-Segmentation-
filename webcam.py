
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


# GPU Configuration
device = 0 if torch.cuda.is_available() else "cpu"

# Load Model
model = YOLO("models/best.pt")


def predict_video(source, width=300, height=400):

    # Convert webcam source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Create output folder
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    # Video settings
    width = int(width)
    height = int(height)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    # Output path
    name = "webcam" if isinstance(source, int) else Path(source).stem
    output_path = output_dir / f"{name}_result.mp4"

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video")

    # Fixed window size
    window_name = "YOLO GPU Segmentation"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    print("Press Q to stop")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # GPU Inference
            results = model.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                device=device,
                verbose=False
            )

            # Draw Segmentation Masks
            annotated_frame = results[0].plot()

            # Resize Frame
            annotated_frame = cv2.resize(
                annotated_frame,
                (width, height)
            )

            # Display
            cv2.imshow(window_name, annotated_frame)

            # Save
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    source = input(
        "Enter source (0 or video path): "
    ).strip().strip('"')

    predict_video(
        source=source,
        width=640,
        height=480
    )
