"""Estimate speed from synchronized tracks in a measured, fixed-camera road region."""
import argparse
import csv
import json
from collections import defaultdict, deque
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np

from .camera_io import read_video


def geometry(config):
    points = np.float32(config["image_xy"])
    meters = np.float32(config["road_xy_m"])
    for array in (points, meters):
        if array.shape != (4, 2) or not np.isfinite(array).all() or not cv2.isContourConvex(array):
            raise ValueError("Supply four corresponding convex corners in perimeter order.")
    return points, cv2.getPerspectiveTransform(points, meters)


def estimate(history, seconds, position):
    if history and (seconds <= history[-1][0] or seconds - history[-1][0] > 0.25):
        history.clear()
    history.append((seconds, position))
    while len(history) > 1 and seconds - history[1][0] >= 0.5:
        history.popleft()
    elapsed = seconds - history[0][0]
    if elapsed < 0.5:
        return None
    return 3.6 * np.linalg.norm(position - history[0][1]) / elapsed


def run(video, tracks, calibration, outdir):
    config = json.loads(Path(calibration).read_text())
    roi, matrix = geometry(config)
    histories = defaultdict(deque)
    smoothed = defaultdict(lambda: deque(maxlen=5))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    size = tuple(config["frame_size"])
    writer = cv2.VideoWriter(str(out / "speed_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError("Cannot create speed overlay")
    try:
        with open(tracks, newline="") as src, (out / "speeds.csv").open("w", newline="") as dest:
            reader = csv.DictReader(src)
            result = csv.DictWriter(dest, fieldnames=[*reader.fieldnames, "raw_kmh", "speed_kmh"])
            result.writeheader()
            groups = groupby(reader, lambda r: int(r["frame_id"]))
            pending = next(groups, None)
            for index, ts, frame in read_video(video):
                if frame.shape[1::-1] != size:
                    raise ValueError("Calibration resolution differs from the video")
                if pending and pending[0] < index:
                    raise ValueError("Tracks must be sorted by frame")
                if pending and pending[0] == index:
                    for row in pending[1]:
                        if abs(ts - float(row["timestamp_s"])) > 0.01:
                            raise ValueError("Track timestamps differ from the video")
                        identity = int(row["track_id"])
                        x1, y1, x2, y2 = (float(row[k]) for k in ("x1","y1","x2","y2"))
                        anchor = ((x1+x2)/2, y2)
                        raw = speed = None
                        h = histories[identity]
                        if cv2.pointPolygonTest(roi, anchor, False) >= 0:
                            pos = cv2.perspectiveTransform(np.float32([[anchor]]), matrix)[0,0]
                            raw = estimate(h, ts, pos)
                            if raw is not None:
                                smoothed[identity].append(raw)
                                speed = np.median(smoothed[identity])
                        else:
                            h.clear()
                        if raw is None:
                            smoothed[identity].clear()
                        result.writerow(dict(row, raw_kmh=raw, speed_kmh=speed))
                        label = f"{row['class']} #{identity}"
                        if speed is not None:
                            label += f" {speed:.1f} km/h"
                        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(40,220,50),2)
                        cv2.putText(frame,label,(int(x1),max(16,int(y1)-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,220,50),1)
                    pending = next(groups, None)
                cv2.polylines(frame,[roi.astype(int)],True,(0,175,255),2)
                writer.write(frame)
            if pending:
                raise ValueError("Track rows exceed the source clip")
    finally:
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video","tracks","calibration","outdir"):
        parser.add_argument(name)
    run(**vars(parser.parse_args()))
