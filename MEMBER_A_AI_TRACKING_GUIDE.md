# Thành viên A — AI / Detection / Tracking

> **Context cho AI assistant trên máy training.**
> Đọc file này để hiểu toàn bộ nhiệm vụ, môi trường, và các bước cần thực hiện.

---

## 1. Tổng quan dự án

Đồ án cuối kỳ môn Trí tuệ nhân tạo: **Đo vận tốc xe máy từ một camera cố định** (đặt trên cầu vượt bộ hành, nhìn xuống đường).

### Pipeline tổng thể

```
Video MP4/MOV → Đọc frame + timestamp (PTS)
    → YOLO11n phát hiện xe (motorcycle, car, bus, truck)
    → ByteTrack gán ID theo dõi xuyên frame
    → Lấy bottom-center bbox → Homography → tọa độ mét thực
    → Cửa sổ trượt + Lọc trung vị → Vận tốc km/h
    → Vẽ kết quả lên video + Ghi CSV
```

### Vai trò thành viên A (chính là ta)

Phụ trách **toàn bộ phần AI**: phát hiện phương tiện (YOLO) + theo dõi đa đối tượng (ByteTrack). Cụ thể:

1. Tìm dataset phù hợp, fine-tune YOLO cho góc nhìn từ cầu vượt
2. Viết module đọc video + trích timestamp — `src/camera_io.py`
3. Viết module detection + tracking — `src/detector_tracker.py`
4. Tạo file CSV mẫu 20 dòng cho B và C code song song
5. Đo benchmark FPS (GPU và CPU)

### Dữ liệu đầu ra (Data Contract) — B và C nhận từ A

Mỗi frame, module của A xuất danh sách đối tượng theo format:

```
frame_id, timestamp_s, track_id, class, x1, y1, x2, y2, confidence
```

Ví dụ:
```csv
frame_id,timestamp_s,track_id,class,x1,y1,x2,y2,confidence
1,0.000,1,motorcycle,320,180,380,250,0.92
2,0.033,1,motorcycle,325,182,385,252,0.91
3,0.067,1,motorcycle,331,185,391,255,0.89
```

---

## 2. Môi trường máy training

| Thông số | Giá trị |
|---|---|
| **GPU** | NVIDIA GTX 1650 SUPER (4GB VRAM) |
| **OS** | Ubuntu 24.04 LTS |
| **Python env** | Anaconda |
| **Có thể train xuyên đêm** | Có |

---

## 3. Setup môi trường

```bash
# Tạo conda env
conda create -n speed-cam python=3.10 -y
conda activate speed-cam

# Cài PyTorch (CUDA 12.x — phổ biến trên Ubuntu 24.04)
# Kiểm tra CUDA version: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài Ultralytics (bao gồm YOLO11)
pip install ultralytics

# Cài thêm
pip install opencv-python-headless numpy pandas lap

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Lưu ý**: Nếu CUDA version khác (ví dụ 11.8), đổi `cu121` thành `cu118`. Kiểm tra bằng `nvidia-smi`.

---

## 4. Dataset: VisDrone-DET (Lựa chọn số 1)

### Vì sao chọn VisDrone?

- Ảnh chụp từ **drone nhìn xuống** — rất giống góc từ cầu vượt
- Có class **`motor`** (motorcycle) — đúng mục tiêu
- Các class khác: car, van, truck, bus, bicycle, pedestrian
- Hàng nghìn ảnh, cảnh đô thị đông đúc tương tự Việt Nam
- **Ultralytics hỗ trợ native** — chỉ cần trỏ config là tự download

### VisDrone class mapping

```
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor          ← xe máy, class quan trọng nhất
```

### Các class ta cần giữ cho đồ án

| VisDrone class | ID | Tương đương COCO |
|---|---|---|
| `motor` | 9 | motorcycle |
| `car` | 3 | car |
| `bus` | 8 | bus |
| `truck` | 5 | truck |
| `van` | 4 | (gom vào car hoặc giữ riêng) |

### Download

**Cách 1: Ultralytics tự download (đơn giản nhất)**

Ultralytics sẽ tự tải VisDrone khi train. Chỉ cần dùng `data="VisDrone.yaml"`.

**Cách 2: Tải thủ công từ GitHub**

```bash
# Clone repo chính thức
git clone https://github.com/VisDrone/VisDrone-Dataset.git

# Hoặc tải trực tiếp từ Hugging Face (đã format YOLO)
# https://huggingface.co/datasets/banu4prasad/VisDrone-Dataset
```

**Cách 3: Kaggle**

Tìm "VisDrone" trên Kaggle, tải ZIP về.

---

## 5. Training YOLO11n trên VisDrone

### 5.1. Train cơ bản (Ultralytics tự download dataset)

```python
from ultralytics import YOLO

# Load model pretrained
model = YOLO("yolo11n.pt")  # nano ~6MB, nhanh, phù hợp 1650 SUPER

# Fine-tune trên VisDrone
results = model.train(
    data="VisDrone.yaml",    # Ultralytics tự download nếu chưa có
    epochs=150,              # Chạy xuyên đêm, nhiều epoch
    patience=20,             # Early stopping: dừng nếu 20 epoch không cải thiện
    imgsz=640,               # Resolution chuẩn, 1650 SUPER chạy tốt
    batch=16,                # Batch size phù hợp 4GB VRAM
    device=0,                # GPU 0
    workers=4,               # Dataloader workers
    project="runs/visdrone", # Thư mục lưu kết quả
    name="yolo11n-visdrone",
    
    # Augmentation (đã có default tốt, có thể tùy chỉnh)
    mosaic=1.0,              # Mosaic augmentation
    mixup=0.1,               # MixUp nhẹ
    copy_paste=0.1,          # Copy-paste augmentation
    
    # Optimizer
    optimizer="AdamW",
    lr0=0.001,               # Learning rate ban đầu
    lrf=0.01,                # Learning rate cuối = lr0 * lrf
    
    # Logging
    save=True,
    save_period=10,          # Lưu checkpoint mỗi 10 epoch
    plots=True,              # Tạo biểu đồ training
)
```

### 5.2. Train với custom YAML (nếu muốn lọc chỉ vài class)

Tạo file `visdrone_filtered.yaml`:

```yaml
# visdrone_filtered.yaml
# Chỉ giữ 5 class phương tiện giao thông cần thiết

path: ../datasets/VisDrone  # sẽ được Ultralytics populate
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

> **Mẹo**: Nếu chỉ muốn detect phương tiện (bỏ pedestrian, people, bicycle, tricycle), cần viết script lọc label file trước khi train. Hoặc train đầy đủ 10 class rồi khi inference chỉ filter output theo class cần.

### 5.3. Train với resolution cao hơn (nếu xe máy quá nhỏ)

Xe máy nhìn từ trên xuống có thể rất nhỏ (10-20 pixel). Nếu mAP cho `motor` thấp:

```python
model.train(
    data="VisDrone.yaml",
    imgsz=960,       # Tăng resolution
    batch=8,         # Giảm batch vì VRAM hạn chế
    # ... các tham số khác giữ nguyên
)
```

> **Chú ý VRAM**: GTX 1650 SUPER chỉ có 4GB.
> - `imgsz=640, batch=16` → ~3.5GB VRAM ✅
> - `imgsz=960, batch=8` → ~3.8GB VRAM ✅ (sát giới hạn)
> - `imgsz=1280, batch=4` → có thể OOM ⚠️ (thử trước)

### 5.4. Inference sau khi train

```python
from ultralytics import YOLO

# Load model đã fine-tune
model = YOLO("runs/visdrone/yolo11n-visdrone/weights/best.pt")

# Chạy trên video
results = model.predict(
    source="data/IMG_9471.MOV",
    conf=0.35,                    # Ngưỡng confidence
    classes=[3, 4, 5, 8, 9],      # Chỉ car, van, truck, bus, motor
    save=True,
    save_txt=True,
)

# Hoặc inference từng frame trong code
results = model(frame, conf=0.35, classes=[3, 4, 5, 8, 9])
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()      # [x1, y1, x2, y2]
    confs = r.boxes.conf.cpu().numpy()       # confidence scores
    cls_ids = r.boxes.cls.cpu().numpy()      # class IDs
```

---

## 6. ByteTrack — Theo dõi đa đối tượng

### ByteTrack là gì?

YOLO xử lý **từng frame độc lập** — nó không biết xe ở frame 1 và frame 2 là cùng 1 xe. ByteTrack giải quyết vấn đề này bằng cách **gán ID duy nhất** cho mỗi xe và duy trì ID xuyên suốt video.

### Tích hợp ByteTrack với Ultralytics (cách đơn giản nhất)

Ultralytics đã tích hợp sẵn ByteTrack:

```python
from ultralytics import YOLO

model = YOLO("runs/visdrone/yolo11n-visdrone/weights/best.pt")

# Track = Detect + ByteTrack
results = model.track(
    source="data/IMG_9471.MOV",
    tracker="bytetrack.yaml",     # Dùng ByteTrack
    conf=0.35,
    classes=[3, 4, 5, 8, 9],
    persist=True,                 # Giữ track xuyên frame
    save=True,
)

# Lấy dữ liệu tracking mỗi frame
for r in results:
    if r.boxes.id is not None:
        track_ids = r.boxes.id.cpu().numpy()     # Track IDs
        boxes = r.boxes.xyxy.cpu().numpy()       # Bounding boxes
        confs = r.boxes.conf.cpu().numpy()        # Confidence
        cls_ids = r.boxes.cls.cpu().numpy()       # Class IDs
```

### Custom ByteTrack config

Tạo file `bytetrack_custom.yaml` nếu cần tinh chỉnh:

```yaml
# bytetrack_custom.yaml
tracker_type: bytetrack
track_high_thresh: 0.5      # Ngưỡng cao để tạo track mới
track_low_thresh: 0.1       # Ngưỡng thấp để recover track bị mất
new_track_thresh: 0.6       # Ngưỡng tạo track hoàn toàn mới
track_buffer: 30            # Số frame giữ track khi xe bị mất (1 giây ở 30fps)
match_thresh: 0.8           # Ngưỡng matching IoU
```

---

## 7. Module cần viết

### 7.1. `src/camera_io.py` — Đọc video + timestamp

```python
"""
Module đọc video và trích xuất timestamp chính xác.
Thành viên A phụ trách.
"""
import cv2

class VideoReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Không mở được video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0
        while True:
            # LẤY TIMESTAMP TRƯỚC KHI READ — quan trọng!
            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_id, timestamp_ms / 1000.0, frame
            frame_id += 1

    def release(self):
        self.cap.release()
```

> **CẢNH BÁO QUAN TRỌNG**: Luôn dùng `CAP_PROP_POS_MSEC` để lấy timestamp thực của video. **KHÔNG** dùng `frame_id / FPS_inference` — vì tốc độ inference GPU khác tốc độ quay video, dẫn đến vận tốc tính ra sai.

### 7.2. `src/detector_tracker.py` — YOLO + ByteTrack

```python
"""
Module phát hiện phương tiện (YOLO) + theo dõi (ByteTrack).
Thành viên A phụ trách.
"""
import numpy as np
from ultralytics import YOLO

# VisDrone class names
VISDRONE_CLASSES = {
    3: "car", 4: "van", 5: "truck", 8: "bus", 9: "motor"
}
# Class IDs phương tiện cần detect
VEHICLE_CLASSES = list(VISDRONE_CLASSES.keys())

class DetectorTracker:
    def __init__(self, model_path: str, conf: float = 0.35,
                 tracker_config: str = "bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker_config = tracker_config

    def process_frame(self, frame: np.ndarray):
        """
        Nhận 1 frame, trả về list các detection đã có track ID.
        
        Returns:
            list[dict]: Mỗi dict chứa:
                - track_id (int)
                - class_name (str): "motor", "car", "bus", "truck", "van"
                - bbox (tuple): (x1, y1, x2, y2)
                - confidence (float)
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            classes=VEHICLE_CLASSES,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes.id is None:
                continue
            for box, track_id, conf, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
            ):
                detections.append({
                    "track_id": int(track_id),
                    "class_name": VISDRONE_CLASSES.get(cls_id, "unknown"),
                    "bbox": tuple(box.tolist()),  # (x1, y1, x2, y2)
                    "confidence": float(conf),
                })
        return detections
```

---

## 8. File CSV mẫu 20 dòng (Mock Data)

**Mục đích**: Cho B và C có dữ liệu để code module của họ ngay từ tuần 1, không cần chờ YOLO train xong.

Tạo file `data/sample_tracks.csv`:

```csv
frame_id,timestamp_s,track_id,class,x1,y1,x2,y2,confidence
0,0.000,1,motor,320.0,180.0,380.0,250.0,0.92
1,0.033,1,motor,325.5,182.0,385.5,252.0,0.91
2,0.067,1,motor,331.0,185.0,391.0,255.0,0.89
3,0.100,1,motor,337.0,188.0,397.0,258.0,0.90
4,0.133,1,motor,343.5,191.0,403.5,261.0,0.88
5,0.167,1,motor,350.0,195.0,410.0,265.0,0.91
6,0.200,1,motor,357.0,198.0,417.0,268.0,0.87
7,0.233,1,motor,364.0,202.0,424.0,272.0,0.90
8,0.267,1,motor,371.5,206.0,431.5,276.0,0.89
9,0.300,1,motor,379.0,210.0,439.0,280.0,0.92
10,0.333,1,motor,387.0,214.0,447.0,284.0,0.88
11,0.367,1,motor,395.5,219.0,455.5,289.0,0.90
0,0.000,2,car,600.0,300.0,720.0,400.0,0.95
1,0.033,2,car,604.0,301.5,724.0,401.5,0.94
2,0.067,2,car,608.0,303.0,728.0,403.0,0.93
3,0.100,2,car,612.5,304.5,732.5,404.5,0.95
4,0.133,2,car,617.0,306.0,737.0,406.0,0.94
5,0.167,2,car,621.5,308.0,741.5,408.0,0.93
6,0.200,2,car,626.0,309.5,746.0,409.5,0.92
7,0.233,2,car,631.0,311.0,751.0,411.0,0.94
```

> File này mô phỏng: xe máy (track 1) di chuyển từ trái sang phải, ô tô (track 2) di chuyển song song. Tọa độ thay đổi đều để B có thể test Homography + tính vận tốc.

---

## 9. Benchmark FPS

Sau khi train xong, chạy benchmark:

```python
from ultralytics import YOLO
import time, cv2

model = YOLO("runs/visdrone/yolo11n-visdrone/weights/best.pt")
cap = cv2.VideoCapture("data/IMG_9471.MOV")

# Warm up
for _ in range(10):
    ret, frame = cap.read()
    model(frame, verbose=False)

# Đo
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
times = []
for i in range(100):
    ret, frame = cap.read()
    if not ret: break
    t0 = time.perf_counter()
    model(frame, conf=0.35, classes=[3, 4, 5, 8, 9], verbose=False)
    times.append(time.perf_counter() - t0)

fps = 1.0 / (sum(times) / len(times))
print(f"Avg inference: {sum(times)/len(times)*1000:.1f} ms | FPS: {fps:.1f}")
cap.release()
```

Kỳ vọng trên GTX 1650 SUPER:
- `yolo11n` (imgsz=640): **60-80 FPS**
- `yolo11s` (imgsz=640): **40-55 FPS**
- `yolo11n` (imgsz=960): **30-45 FPS**

---

## 10. Cấu trúc thư mục dự án

```
cuoi-ki-ttnt-prj/
├── configs/
│   ├── calibration.json          # [B] Ma trận H + 4 điểm mốc
│   └── config.yaml               # [Nhóm] Cấu hình chung
├── data/
│   ├── IMG_9471.MOV              # Video thực nghiệm
│   └── sample_tracks.csv         # [A] Mock data 20 dòng cho B, C
├── src/
│   ├── camera_io.py              # [A] Đọc video + timestamp
│   ├── detector_tracker.py       # [A] YOLO + ByteTrack
│   ├── calibration.py            # [B] Tool chọn 4 điểm, tính H
│   ├── speed_estimator.py        # [B] Tính vận tốc km/h
│   ├── visualizer.py             # [C] Vẽ bbox, ID, badge km/h
│   ├── evaluate.py               # [C] Tính MAE, RMSE
│   └── app.py                    # [C] Chương trình chính
├── tests/
│   └── test_geometry.py          # [B] Unit test toán
├── runs/                         # Kết quả training YOLO
├── outputs/                      # Video + CSV kết quả
└── de_cuong_do_toc_do_xe_may_3_thanh_vien.pdf  # Đề cương
```

---

## 11. Checklist thành viên A

- [ ] Tạo conda env + cài PyTorch + Ultralytics
- [ ] Kiểm tra GPU nhận (`torch.cuda.is_available()`)
- [ ] Tạo file `data/sample_tracks.csv` (20 dòng mock) → gửi B và C
- [ ] Chạy thử YOLO11n pretrained (COCO) trên `IMG_9471.MOV` → xem nhận diện thế nào
- [ ] Download VisDrone dataset
- [ ] Fine-tune YOLO11n trên VisDrone (xuyên đêm, early stopping)
- [ ] Đánh giá mAP, đặc biệt class `motor`
- [ ] Nếu mAP `motor` thấp → thử `imgsz=960` hoặc `yolo11s`
- [ ] (Tùy chọn) Annotate thêm vài trăm frame từ `IMG_9471.MOV`
- [ ] Viết `src/camera_io.py`
- [ ] Viết `src/detector_tracker.py`
- [ ] Tích hợp ByteTrack, test ID ổn định
- [ ] Đo benchmark FPS (GPU + CPU)
- [ ] Export model `best.pt` cho C tích hợp vào `app.py`

---

## 12. Lưu ý đặc biệt

### Về timestamp (CỰC KỲ QUAN TRỌNG)

```python
# ✅ ĐÚNG: Lấy timestamp thực từ video
timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

# ❌ SAI: Lấy thời gian inference
timestamp_s = frame_id / inference_fps  # SAI! Vận tốc sẽ bị lệch!
```

### Về VisDrone class vs COCO class

VisDrone và COCO đánh số class **khác nhau**. Nếu fine-tune trên VisDrone thì dùng VisDrone class IDs. Nếu dùng pretrained COCO thì class motorcycle = 3 (COCO), nhưng trên VisDrone motor = 9.

### Về Git branch

Làm việc trên nhánh `feature/ai-A`. Không commit thẳng vào `main`.

```bash
git checkout -b feature/ai-A
# ... code ...
git add . && git commit -m "feat(A): YOLO11n + ByteTrack integration"
git push origin feature/ai-A
```
