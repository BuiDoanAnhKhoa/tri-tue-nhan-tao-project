import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
from .camera_io import read_video


def export(video, weights, output, device):
    model = YOLO(weights)
    names = {"car", "truck", "bus", "bicycle", "motor", "motorcycle"}
    classes = [i for i, name in model.names.items() if name in names]
    with Path(output).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow("frame_id,timestamp_s,track_id,class,x1,y1,x2,y2,confidence".split(","))
        for index, seconds, frame in read_video(video):
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1,
                                 classes=classes, device=device, verbose=False)[0]
            boxes = result.boxes.cpu()
            if boxes.id is None:
                continue
            for identity, cls, xyxy, score in zip(boxes.id, boxes.cls, boxes.xyxy, boxes.conf):
                name = model.names[int(cls)]
                writer.writerow([index, seconds, int(identity), "motor" if name == "motorcycle" else name,
                                 *xyxy.tolist(), float(score)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("weights")
    parser.add_argument("output")
    parser.add_argument("--device", default="0")
    export(**vars(parser.parse_args()))
