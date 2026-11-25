"""Run from project root after downloading the official train and val archives."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import gc
import time
import torch

from PIL import Image
from ultralytics import YOLO
import yaml


def prepare():
    root = Path("data/traffic").resolve()
    # Original VisDrone annotation IDs are one-based.
    mapping = {3: 0, 4: 1, 6: 2, 9: 3, 10: 4}
    for split, expected in [("train", 6471), ("val", 548)]:
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        with ZipFile(f"data/VisDrone2019-DET-{split}.zip") as archive:
            files = [f for f in archive.namelist() if "/images/" in f and f.endswith(".jpg") and not Path(f).name.startswith(".")]
            if len(files) != expected:
                raise ValueError(f"Unexpected image count {len(files)} in {split}")
            for f in files:
                contents = archive.read(f)
                with Image.open(BytesIO(contents)) as image:
                    width, height = image.size
                rows = []
                annotation = f.replace("/images/", "/annotations/")[:-4] + ".txt"
                for line in archive.read(annotation).decode().splitlines():
                    x, y, w, h, valid, cls = map(int, line.split(",")[:6])
                    if valid and cls in mapping:
                        left, top = max(x, 0), max(y, 0)
                        right, bottom = min(width, x + w), min(height, y + h)
                        if right > left and bottom > top:
                            rows.append(f"{mapping[cls]} {(left+right)/(2*width)} {(top+bottom)/(2*height)} {(right-left)/width} {(bottom-top)/height}")
                (images / Path(f).name).write_bytes(contents)
                (labels / Path(annotation).name).write_text("\n".join(rows))
    config = root / "traffic.yaml"
    config.write_text(yaml.safe_dump(dict(path=str(root), train="images/train", val="images/val",
                                          names=["bicycle", "car", "truck", "bus", "motor"])))
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    config = prepare()
    settings = dict(device=0, workers=2, project=str(Path("runs").resolve()), data=str(config))
    candidates = []
    for batch, size in [(args.batch, args.imgsz), (4, 640), (8, 704)]:
        # Time steady-state training batches and include a validation pass in each trial.
        starts, elapsed = [], []
        model = YOLO("yolo11n.pt")
        def start(trainer):
            torch.cuda.synchronize()
            starts.append(time.perf_counter())
        def finish(trainer):
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - starts[-1])
        model.add_callback("on_train_batch_start", start)
        model.add_callback("on_train_batch_end", finish)
        try:
            model.train(**settings, batch=batch, imgsz=size, fraction=0.015, epochs=1,
                        name=f"trial-{batch}-{size}", plots=False, save=False)
            if len(elapsed) > 3:
                candidates.append((batch * (len(elapsed)-3) / sum(elapsed[2:-1]), batch, size))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"Insufficient VRAM for batch={batch}, imgsz={size}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print("Trial results (images/s, batch, image size)", candidates, flush=True)
    _, batch, size = max(candidates) if candidates else (0, 2, 640)
    model = YOLO("yolo11n.pt")
    deadline = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).replace(hour=23, minute=30, second=0, microsecond=0).timestamp()

    def check_deadline(trainer):
        # Leave time to validate the saved checkpoint and run inference today.
        if time.time() + trainer.epoch_time >= deadline:
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", check_deadline)
    model.train(**settings, batch=batch, imgsz=size, epochs=250, patience=30, name="traffic")
