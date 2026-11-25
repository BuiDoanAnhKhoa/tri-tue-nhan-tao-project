"""Run in lane-det: python benchmark.py INPUT_MP4 BEST_PT OUTPUT_DIR"""
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

source, weights, output = map(Path, sys.argv[1:])
measurements = []
for device in ["0", "cpu"]:
    model = YOLO(weights)
    capture = cv2.VideoCapture(str(source))
    ok, first = capture.read()
    assert ok
    options = dict(device=device, imgsz=640, conf=0.1, half=False, verbose=False)
    for _ in range(10):
        model.predict(first, **options)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    gpu = device == "0"
    if gpu:
        writer = cv2.VideoWriter(str(output / "VisDrone_intersection_tracked.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), capture.get(cv2.CAP_PROP_FPS),
                                 (first.shape[1], first.shape[0]))
        assert writer.isOpened()
        track_file = (output / "intersection_tracks.csv").open("w", newline="")
        tracks = csv.writer(track_file)
        tracks.writerow("frame,track_id,class,x1,y1,x2,y2,confidence".split(","))
    frame = 0
    times = []
    start = perf_counter()
    while True:
        ok, image = capture.read()
        if not ok:
            break
        t = perf_counter()
        result = model.track(image, persist=True, tracker="bytetrack.yaml", **options)[0]
        if gpu:
            torch.cuda.synchronize()
        elapsed = perf_counter() - t
        frame += 1
        if frame > 10:
            times.append(elapsed)
        if gpu:
            writer.write(result.plot(line_width=1, font_size=8))
            boxes = result.boxes.cpu()
            if boxes.id is not None:
                for identity, cls, coords, confidence in zip(boxes.id, boxes.cls, boxes.xyxy, boxes.conf):
                    tracks.writerow([frame, int(identity), model.names[int(cls)], *coords.tolist(), float(confidence)])
    capture.release()
    if gpu:
        writer.release()
        track_file.close()
    duration = perf_counter() - start
    measurements.append(dict(device=device, frames=frame, measured_frames=len(times),
                             tracking_fps=len(times)/sum(times), median_ms=1000*np.median(times),
                             p95_ms=1000*np.percentile(times,95), wall_seconds=duration,
                             wall_fps=frame/duration, includes_output=gpu))
(output / "benchmark.json").write_text(json.dumps(measurements, indent=2))
print(measurements)
