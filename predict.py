
from pathlib import Path
from ultralytics import YOLO


# Load Model
model = YOLO("models/best.pt")


def predict(image_path):
    # Create output folder
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Run Segmentation
    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=640,
        save=False
    )

    # Save Result
    for result in results:
        output_path = output_dir / Path(image_path).name
        result.save(filename=str(output_path))

        print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip().strip('"')
    predict(image_path)
