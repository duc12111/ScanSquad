from os.path import join

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-tumor.pt")

    # image_dir = "brain-tumor/valid/images"
    # image_dir = "luca/Series-018 sag"
    image_dir = "luca/Series-018 sag"

    # quick and dirty
    # model.predict(source=image_dir, conf=0.8, save=True)

    # one pic at a time
    for f in image_dir:

        image_path = join(image_dir, f)
        results = model(image_path)
        for result in results:
            boxes = result.boxes.xyxy  # [x1, y1, x2, y2] format
            confs = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class IDs
            print(f"Detected {len(boxes)} objects in {image_path}")


if __name__ == '__main__':
    main()
