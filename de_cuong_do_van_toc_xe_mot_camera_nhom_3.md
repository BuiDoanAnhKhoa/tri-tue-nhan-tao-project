# ĐỀ CƯƠNG NGHIÊN CỨU & KẾ HOẠCH TRIỂN KHAI

## Ước lượng vận tốc phương tiện giao thông từ một camera cố định

**Tên tiếng Anh:** *Vehicle Speed Estimation Using a Fixed Monocular Camera: Detection, Tracking, and Geometric Calibration*  
**Phiên bản:** 1.0 — 22/11/2026  
**Quy mô:** Nhóm 3 thành viên; đồ án nghiên cứu và demo  
**Phần cứng dự kiến:** Camera laptop / camera Jetson, laptop hoặc PC Ubuntu, GTX 1650 Super 4GB để thử nghiệm/fine-tune (nếu có), Jetson Nano 2GB là mục tiêu mở rộng.

> **Định hướng chốt:** Hoàn thành hệ thống **camera cố định → phát hiện xe → tracking → hiệu chuẩn mặt đường → ước lượng km/h → đánh giá sai số**, chạy ổn định trên laptop/PC trước. Jetson Nano là phần **tùy chọn**, không phải điều kiện để hoàn thành đồ án.

---

## 1. Mục tiêu, câu hỏi nghiên cứu và phạm vi

### 1.1. Bài toán

Từ video của **một camera cố định** nhìn xuống một đoạn đường, xác định vận tốc (km/h) của từng xe khi đi qua vùng quan sát. Không sử dụng radar, cảm biến đo tốc độ hay camera thứ hai trong **quá trình suy luận**. Có thể dùng khoảng cách vật lý được đo trước để hiệu chuẩn, và dùng nguồn đo tốc độ độc lập để **đánh giá**.

### 1.2. Mục tiêu bắt buộc (MVP)

1. Nhận diện ít nhất ô tô; nếu điều kiện cho phép, mở rộng xe máy/xe tải/xe buýt.
2. Gán ID theo dõi cho các xe liên tiếp trong video.
3. Ước lượng vị trí xe trên **mặt đường** và quy đổi từ pixel sang mét bằng phép hiệu chuẩn.
4. Tính tốc độ theo thời gian thực của video, hiển thị ID + km/h trên từng xe.
5. Xuất video có chú thích, bảng CSV kết quả và các chỉ số đánh giá.
6. Có ít nhất một phép so sánh thực nghiệm giữa phương án đơn giản và phương án đề xuất.

### 1.3. Câu hỏi nghiên cứu

- So với hệ số pixel-to-meter cố định, **homography** cải thiện sai số vận tốc bao nhiêu?
- Sử dụng vị trí bottom-center của bounding box và làm mượt quỹ đạo có ổn định không khi xe xa camera hoặc bị che khuất?
- Tần suất xử lý 5/10/15/30 FPS ảnh hưởng thế nào đến MAE và tốc độ chạy hệ thống?
- Mô hình chạy trên laptop/PC đạt tốc độ bao nhiêu; có thể tối ưu xuống Jetson Nano 2GB hay không?

### 1.4. Ngoài phạm vi ban đầu

- Camera gắn trên phương tiện đang di chuyển; hệ thống phạt nguội hoặc thiết bị đo tốc độ đạt chuẩn pháp lý; bảo đảm đo chính xác mọi góc nhìn; tự hiệu chuẩn hoàn toàn không có tham chiếu kích thước; train end-to-end từ đầu.

---

## 2. Vì sao một camera vẫn đo được vận tốc?

Công thức:

\[
 v_{\mathrm{km/h}} = 3.6\,\frac{\Delta s_{\mathrm{m}}}{\Delta t_{\mathrm{s}}}.
\]

Camera cho vị trí ảnh `(u, v)` đơn vị **pixel**, chứ không cho trực tiếp mét. Khi camera nghiêng, xe gần và xe xa camera có **tỷ lệ pixel/mét khác nhau**. Vì vậy, việc lấy một hằng số `meters_per_pixel` dùng cho cả ảnh thường tạo sai số có hệ thống.

**Cách làm thực tế:** Cố định camera; xác định ≥4 điểm **không thẳng hàng** cùng thuộc mặt phẳng mặt đường và biết tọa độ tương ứng trong hệ mét; tính homography `H`; chuyển điểm đại diện mặt đường của xe thành `(X, Y)` tính bằng mét. Sau đó tính dịch chuyển theo timestamp thực tế.

Với tọa độ thuần nhất:

\[
\lambda \begin{bmatrix}X\\Y\\1\end{bmatrix}
=H\begin{bmatrix}u\\v\\1\end{bmatrix}.
\]

> Không thể suy ra tốc độ tuyệt đối km/h đáng tin cậy từ pixel của camera đơn **nếu không có ràng buộc về tỷ lệ thực** (mốc đo, kích thước đã biết, hiệu chuẩn hoặc giả định bổ sung). Homography cũng chỉ mô tả một **mặt phẳng**, không tự động xác định tỷ lệ mét nếu các điểm đích chỉ là tọa độ tùy ý.

**Điểm lấy trên xe:** Điểm giữa cạnh dưới bounding box là một xấp xỉ tiện lợi nhưng có sai số do bounding box, góc nhìn, vật cản và bóng đổ. Hãy nêu rõ sai số này; nếu đủ thời gian có thể thử segmentation/ước lượng điểm tiếp xúc bánh xe với mặt đường.

---

## 3. Kiến trúc hệ thống

```text
Camera USB/CSI hoặc MP4
        │
        ▼
Frame reader + timestamps
        │
        ▼
YOLO pretrained (vehicle detection)
        │
        ▼
ByteTrack (persistent track IDs)
        │
        ▼
Bottom-center / ground-contact proxy
        │
        ▼
Homography → world coordinates (meters)
        │
        ▼
Track history + Δdistance/Δtime
        │
        ▼
Moving average / robust smoothing
        │
        ▼
Speed estimate by ID (km/h)
        ├── Annotated video/live display
        ├── CSV (id, timestamps, speed, class...)
        └── Evaluation (MAE, RMSE, bias, FPS)
```

**Khuyến nghị công nghệ:** Python 3.x, PyTorch, Ultralytics YOLO phiên bản nano tương thích môi trường, OpenCV, NumPy, pandas, ByteTrack (qua cấu hình tracker của Ultralytics), matplotlib. Cố định phiên bản trong `requirements.txt`; tránh đổi model/framework giữa chừng.

**Phân biệt:** Supervised learning nằm chủ yếu ở **mô hình phát hiện xe** (pretrained hoặc fine-tune). ByteTrack, homography, phép tính vận tốc **không cần train supervised**. Train mô hình hồi quy vận tốc là một nhánh nghiên cứu mở rộng, **không bắt buộc**.

---

## 4. Dataset: chọn cái nào, dùng vào việc gì?

| Dataset | Đặc điểm đã công bố | Vai trò phù hợp | Điểm cần kiểm tra |
|---|---|---|---|
| **BrnoCompSpeed** | 18 video Full HD, khoảng một giờ/video; 6 địa điểm; hơn 20.000 lượt xe có tham chiếu vận tốc, dữ liệu hiệu chuẩn | **Benchmark chính** về tốc độ, thử nghiệm homography | Quyền truy cập, kích thước tải về, định dạng ground truth; không mặc định có nhãn YOLO theo từng frame |
| **ForeSpeed** | 322 video CCTV với điều kiện camera/định dạng khác nhau, có tốc độ tham chiếu | **Kiểm tra tổng quát hóa** hoặc phân tích FPS/độ nén | Một số video dùng variable frame rate; cần timestamp chính xác; xem cấu trúc/điều kiện dataset |
| **VS13** | 400 clip, 13 loại xe, một lượt xe/clip, tốc độ đã biết | Nhánh mở rộng hồi quy hoặc thử bài toán nhìn ngang | Không phải lựa chọn số 1 để benchmark hệ thống nhìn từ trên cao, nhiều xe, có homography |
| **Video tự quay** | Camera laptop / camera Jetson, đặt cố định, đã đo các mốc mặt đường | **Demo thực tế + validation phụ** | Cần tốc độ tham chiếu độc lập/tương đối độc lập; quyền riêng tư & an toàn |

Nguồn gốc dataset:

- BrnoCompSpeed: https://github.com/JakubSochor/BrnoCompSpeed ; công bố https://www.fit.vut.cz/research/result/c147193/.en
- ForeSpeed: https://github.com/ampedsoftware/forespeed ; bài báo https://www.mdpi.com/2313-433X/12/10/488
- VS13: https://slobodan.ucg.ac.me/science/vs13/ ; bài liên quan https://arxiv.org/abs/2212.01651

**Lưu ý cực quan trọng:** *Speed ground truth* là nhãn **vận tốc**, còn *object detection labels* là nhãn **hộp bao**. Không thể lấy bảng tốc độ gắn vào tham số `data=...` rồi fine-tune YOLO detector. Để fine-tune YOLO phải có bộ nhãn bounding box tương ứng (có thể chọn dataset detection riêng hoặc tự gán nhãn một tập con).

**Đề xuất thực dụng:** chạy pipeline trên một **phần nhỏ BrnoCompSpeed** trước (1 video/đoạn cắt, một góc nhìn). Không tải toàn bộ dữ liệu chỉ để dựng prototype. Khi benchmark hãy chia theo video/camera/location; tránh dùng các frame rất gần nhau ở cả train và test.

---

## 5. Phương pháp triển khai theo mức độ

### Baseline A — pixel scaling

- YOLO → ByteTrack → vị trí xe `(u, v)` → dịch chuyển pixel → nhân tỷ lệ mét/pixel cố định → km/h.
- **Chỉ để đối chứng**, vì sai số phối cảnh sẽ lớn khi xe chạy ở vùng gần/xa khác nhau.

### Phương pháp B — homography

1. Chọn 4+ mốc cùng trên mặt đường; biết khoảng cách/kích thước thực để xây dựng tọa độ mét.
2. Chỉ định cặp tọa độ `(pixel_x, pixel_y) ↔ (X_m, Y_m)`.
3. Tính `H` bằng `cv2.getPerspectiveTransform` (4 cặp) hoặc `cv2.findHomography` (nhiều cặp, có kiểm tra outlier).
4. Theo từng track, lấy điểm tiếp xúc mặt đường xấp xỉ; dùng `cv2.perspectiveTransform` để lấy `(X, Y)`.
5. Với cửa sổ thời gian `τ` (ví dụ 0,5–1,0 s), tính:

\[
\hat v_i(t)=3.6\,\frac{\lVert \mathbf p_i(t)-\mathbf p_i(t-\tau)\rVert_2}{\tau}.
\]

6. Chỉ xuất kết quả nếu track đủ tuổi, đủ điểm và chuyển động đáng tin cậy.

### Phương pháp C — homography + smoothing (đề xuất chính)

- Moving median/average trên tốc độ hoặc lọc Kalman trên **tọa độ mặt đường**.
- Loại chuyển động bất thường, nhảy ID, detection mất quá lâu; đánh dấu độ tin cậy thấp thay vì xuất tốc độ vô lý.
- Lưu cả tốc độ tức thời được làm mượt và tốc độ trung bình trên vùng quan sát.

**Timestamp:** Với video MP4, ưu tiên timestamp frame thực hoặc PTS do decoder cung cấp. Với camera trực tiếp, dùng đồng hồ đơn điệu tại thời điểm frame được nhận và cân nhắc độ trễ hàng đợi. **Không được lấy `frame_index / inference_FPS`** vì FPS suy luận khác FPS thu hình. Với video VFR, không dùng mặc định `frame_index / nominal_FPS`.

**Không dùng hai mốc A–B đã dùng làm toàn bộ cơ sở quy đổi làm phép kiểm chứng độc lập duy nhất.** Nếu dùng cùng mốc để tính homography và tham chiếu tốc độ, cần ghi rõ phụ thuộc này; tốt hơn dùng nguồn tốc độ độc lập hoặc các mốc kiểm định khác được đo riêng.

---

## 6. Thiết kế thực nghiệm & thước đo

### 6.1. Các thí nghiệm tối thiểu

| ID | Thí nghiệm | Thay đổi | Đo lường |
|---|---|---|---|
| E1 | Baseline vs homography | A → B | MAE, RMSE, bias (km/h) |
| E2 | Smoothing | B → C | MAE và độ dao động vận tốc |
| E3 | Tần suất inference | 5, 10, 15, 30 FPS mục tiêu hoặc lấy mẫu tương ứng | Sai số + FPS xử lý + tỉ lệ track hợp lệ |
| E4 | Khoảng cách đến camera | Gần / giữa / xa | Sai số theo vùng mặt đường |
| E5 | Video thực tế | Tối thiểu 10–20 lượt xe nếu khả thi | Sai số từng lượt và ví dụ thành công/thất bại |

**Các tiêu chí:**

\[
\operatorname{MAE}=\frac1N\sum_{i=1}^{N}|\hat v_i-v_i|,
\qquad
\operatorname{RMSE}=\sqrt{\frac1N\sum_{i=1}^{N}(\hat v_i-v_i)^2}.
\]

\[
\operatorname{Bias}=\frac1N\sum_{i=1}^{N}(\hat v_i-v_i).
\]

- Chấm **theo sự kiện xe/lượt chạy** nếu nhãn tham chiếu ở cấp lượt xe; đừng tính mọi frame của cùng một xe như mẫu độc lập.
- Ghi số lượt dự đoán hợp lệ, tỷ lệ bỏ sót, tỷ lệ match với ground truth; không chỉ báo cáo MAE của những xe dễ đo nhất.
- Báo cáo thời gian xử lý trung bình/frame, FPS, CPU/RAM, GPU VRAM và độ trễ hiển thị (nếu demo realtime).
- Không tự đặt ra các số liệu MAE/FPS như kết quả thật trước khi đo.

### 6.2. Tự quay an toàn

- Cố định camera chắc chắn ở vị trí **được phép**, tránh đặt thiết bị gây nguy hiểm/che khuất giao thông.
- Bắt đầu bằng xe đồ chơi, xe đạp hoặc xe chạy chậm trong khu vực an toàn; sau đó mới thử phương tiện thật ở vị trí phù hợp.
- Chụp/quay ảnh mặt đường để đo tọa độ các mốc; ghi độ dài đo thực, độ cao/góc camera, độ phân giải và timestamp.
- Đồng hồ xe máy **chỉ là tham chiếu sơ bộ**, không mặc định là ground truth chính xác. Có thể đánh giá bằng thời gian đi qua hai mốc đo **độc lập với mốc hiệu chuẩn** hoặc dùng thiết bị ghi vận tốc độc lập với độ chính xác đã biết.
- Hạn chế ghi thông tin nhận dạng như biển số/khuôn mặt; tránh chia sẻ dữ liệu cá nhân không cần thiết.

---

## 7. Phân công nhóm 3 người — chia theo interface, không chia kiểu ai làm gì cũng được

### Thành viên A — AI Detection & Tracking (**chủ phần inference**)

**Nhiệm vụ:**
- Đọc video/camera; lựa chọn model nhẹ (ví dụ YOLO nano) và cấu hình inference.
- Tích hợp ByteTrack; lưu mỗi xe theo `track_id`, `class_id`, bbox, confidence, `timestamp`.
- Đánh giá detection/tracking trên clip thử; theo dõi FPS, độ ổn định ID; cân nhắc fine-tune **chỉ khi cần**.

**Bàn giao:** `detector_tracker.py`, `camera_io.py`, một video demo có ID, log `tracks.csv`, benchmark FPS.

**Hợp đồng dữ liệu (output tối thiểu):** `frame_index,timestamp_s,track_id,class_name,x1,y1,x2,y2,confidence`.

### Thành viên B — Camera Calibration & Speed Estimation (**chủ thuật toán nghiên cứu**)

**Nhiệm vụ:**
- Nghiên cứu homography, phép đo tỷ lệ thực và sai số hình học.
- Làm giao diện/script chọn điểm hiệu chuẩn, lưu `calibration.json`.
- Tính world coordinates, tốc độ theo cửa sổ thời gian, lọc nhiễu, từ chối điểm bất thường.
- Viết test với quỹ đạo tổng hợp đã biết tốc độ, kiểm chứng đơn vị m/s ↔ km/h.

**Bàn giao:** `calibration.py`, `speed_estimator.py`, bộ test, ví dụ bảng tốc độ có `track_id`.

**Hợp đồng dữ liệu (input):** log track từ A + `calibration.json` + timestamps.

### Thành viên C — Dataset, Evaluation & Integration (**chủ thí nghiệm và sản phẩm**)

**Nhiệm vụ:**
- Chuẩn bị clip BrnoCompSpeed, xác minh cấu trúc metadata và ground truth; quản lý version/chia tập.
- Tạo script ghép dự đoán với ground truth, tính MAE/RMSE/bias; vẽ biểu đồ, bảng so sánh A/B/C.
- Tích hợp live/video/CSV, kiểm tra tính tái lập; tổ chức quay thử và lập báo cáo/slides.
- Tạo `README.md`, hướng dẫn setup, ghi rõ các giới hạn và lỗi đã gặp.

**Bàn giao:** `evaluate.py`, `results/*.csv`, biểu đồ, demo hoàn chỉnh, báo cáo.

### Nguyên tắc để tránh một người gánh cả dự án

1. **Chia theo module nhưng review chéo:** A review pipeline B, B review metric C, C kiểm thử đầu ra A.
2. **Mỗi tuần có demo chạy được**, không chờ từng người “làm xong hết” mới ghép code.
3. **Giao diện dữ liệu thống nhất trong tuần 1**, có 10–20 dòng CSV mẫu để B/C phát triển ngay cả khi YOLO chưa xong.
4. **Mỗi pull request cần một người khác review**; branch riêng, không sửa thẳng vào `main`.
5. **Ai làm module nào chịu trách nhiệm tài liệu + unit test module đó**; C không phải người duy nhất viết báo cáo.
6. **Mỗi thành viên trình bày phần mình nghiên cứu** và đều nắm end-to-end pipeline.

**Phân chia đóng góp trong báo cáo:** A viết detection/tracking và benchmark runtime; B viết hình học/phương pháp vận tốc; C viết dataset/evaluation/results; cả ba thống nhất abstract, kết luận, hạn chế.

---

## 8. Tiến độ 8 tuần (có tiêu chí nghiệm thu)

| Tuần | A: Detection/Tracking | B: Calibration/Speed | C: Dataset/Evaluation | Mốc hoàn thành |
|---|---|---|---|---|
| 1 | Setup, đọc video, YOLO pretrained | Nghiên cứu homography, synthetic test | Lấy đoạn dataset, định nghĩa log + metric | **Chốt CSV/API**; video có bbox |
| 2 | ByteTrack và ổn định ID | Chọn điểm và tính H, plot bird's-eye | Parse ground truth, mẫu evaluation | Video có ID + transform test |
| 3 | Chuẩn hóa log, timestamps | Tính tốc độ baseline A và B | Script ghép GT và dự đoán | **Demo đầu tiên có km/h** |
| 4 | Sửa lỗi ID swap/occlusion | Smoothing C và kiểm tra outlier | Chạy E1/E2, báo cáo sơ bộ | Bảng MAE/RMSE ban đầu |
| 5 | Tối ưu inference FPS | Khảo sát vùng gần/xa | E3/E4, vẽ biểu đồ | So sánh tham số có số liệu |
| 6 | Cấu hình camera live | Kiểm chuẩn mốc thực | Tổ chức quay an toàn, E5 | Demo webcam thật + CSV |
| 7 | Fix bug, chuẩn bị tối ưu nhẹ | Phân tích failure cases | Viết báo cáo/đóng gói | Release candidate |
| 8 | Benchmark hoàn chỉnh | Chốt phương pháp và hạn chế | Slides, kiểm chứng tái lập | **Demo, báo cáo, video dự phòng** |

**Nếu chỉ có 4–5 tuần:** bỏ fine-tune, hồi quy XGBoost và Jetson; tập trung tuần 1–4 + một buổi quay thực tế.

**Tiêu chí nghiệm thu MVP:** có video đầu ra với ID + km/h, có calibration có đơn vị mét, có một bảng MAE/RMSE trên tập có tốc độ chuẩn, có ít nhất một so sánh baseline với phương án đề xuất, có README chạy lại.

---

## 9. Cấu trúc repository và quy trình Git

```text
vehicle-speed-estimation/
├── README.md
├── requirements.txt
├── configs/
│   ├── project.yaml
│   └── calibration.example.json
├── src/
│   ├── camera_io.py           # A
│   ├── detector_tracker.py    # A
│   ├── calibration.py         # B
│   ├── speed_estimator.py     # B
│   ├── evaluate.py            # C
│   └── app.py                 # A+C
├── tests/
│   ├── test_homography.py     # B
│   ├── test_speed.py          # B
│   └── test_io.py             # A+C
├── scripts/
│   ├── benchmark.py
│   └── extract_frames.py
├── docs/
│   ├── literature_review.md
│   └── experiment_protocol.md
├── data/                      # Git ignored
├── models/                    # Large weights Git ignored
└── outputs/                   # Git ignored
```

**Thỏa thuận kỹ thuật tuần 1:** Python version; tọa độ `xyxy`; đơn vị `timestamp_s` (giây); điểm đại diện `bottom_center`; tọa độ ground `(X_m,Y_m)`; phiên bản model; kích thước frame; định dạng calibration; quy tắc đặt ID; output CSV. Viết 2–3 test số học đơn giản chạy được trong CI/local.

**Họp nhóm:** 15–20 phút/tuần để ghép demo + 1 buổi review PR; mọi issue có người phụ trách, hạn hoàn thành và tiêu chí chấp nhận. Lưu video demo ngắn để phòng trường hợp camera/internet gặp lỗi lúc bảo vệ.

---

## 10. Phần cứng, thời gian train và triển khai

- **Laptop/webcam:** dùng để debug; camera laptop tích hợp thường khó đặt trên cao nên có thể đặt cố định trên giá/điểm nhìn phù hợp hoặc sử dụng camera Jetson/clip quay sẵn để thực nghiệm.
- **GTX 1650 Super 4GB:** phù hợp thử YOLO nano và fine-tune mức nhỏ; không cần fine-tune nếu pretrained đủ tốt. Tùy batch/imgsz và phiên bản CUDA, có thể phải giảm `batch` để tránh OOM. Dự báo giờ train phải dựa vào thử nghiệm 3–5 epochs trên máy thật.
- **Dataset trên HDD:** DataLoader đọc ảnh qua RAM, theo batch chuyển tensor qua VRAM; HDD có thể gây chờ I/O. Nếu máy đang có vấn đề lag HDD, ưu tiên backup và thử trên SSD/clip nhỏ trước khi train dài.
- **Jetson Nano 2GB:** mục tiêu mở rộng; phụ thuộc JetPack, RAM 2GB, camera CSI/USB, nguồn USB-C 5V/3A đúng chuẩn; kiểm tra tương thích PyTorch/TensorRT. Không cam kết FPS trước khi đo.
- **Chiến lược tối ưu:** giảm kích thước input, xử lý thưa frame hợp lý, inference engine phù hợp, giữ timestamps gốc, cache khi có đủ RAM.

**Khuyến nghị:** đào sâu **hình học và đánh giá sai số**, không dành 3–4 tuần train detector chỉ để có một mô hình mới.

---

## 11. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án dự phòng |
|---|---|---|
| Không lấy được đủ BrnoCompSpeed | Link tải bị hạn chế / dữ liệu lớn | Sử dụng clip đã tải hợp lệ, ForeSpeed và video tự quay; ghi rõ benchmark nào đã thực hiện |
| Không có thông tin mét thực tế | Không có mốc/khoảng cách | Không báo cáo km/h như phép đo có cơ sở; quay lại cảnh có mốc chuẩn hoặc dùng metadata calibration hợp lệ |
| Tracking nhảy ID | Vận tốc tăng/giảm bất thường | Yêu cầu track đủ dài, reset lịch sử khi ID thay đổi, lọc outlier |
| Camera/fps không ổn định | Video bị drop frame/giật | Lưu timestamp từng frame, phát hiện frame trùng, không dùng FPS inference để tính thời gian |
| Giới hạn GTX 1650 Super 4GB | CUDA OOM | Model nano, batch 2–4 hoặc nhỏ hơn, imgsz 512, không fine-tune nếu không cần |
| Jetson Nano triển khai khó | Thiếu RAM, thư viện/nguồn lỗi | Demo laptop đã hoàn thiện + video offline |
| Dữ liệu thực tế thiếu GT chính xác | Chỉ có tốc độ đồng hồ xe | Chỉ coi là kiểm tra tham khảo, không dùng để tuyên bố độ chính xác cao |

---

## 12. Tài liệu khoa học nên đọc (theo thứ tự)

### Bắt buộc đọc

**[P1] Sochor et al. — *Comprehensive Data Set for Automatic Single Camera Visual Speed Measurement* (IEEE T-ITS, 2018).**  
Đọc để hiểu bản chất benchmark, hiệu chuẩn camera, ground truth và cách đánh giá.  
DOI: https://doi.org/10.1109/TITS.2018.2825609  
Trang tác giả: https://www.fit.vut.cz/research/result/c147193/.en

**[P2] *Accurate Vehicle Speed Estimation from Monocular Camera Footage* (ISPRS Annals, 2020).**  
Gần với pipeline nhóm: YOLO + SORT + homography. Dễ dùng làm baseline khoa học.  
https://isprs-annals.copernicus.org/articles/V-2-2020/419/2020/

**[P3] *Robust Automatic Monocular Vehicle Speed Estimation for Traffic Surveillance* (ICCV, 2021).**  
Nghiên cứu camera calibration thực tế và tính robust của hệ thống.  
https://openaccess.thecvf.com/content/ICCV2021/html/Revaud_Robust_Automatic_Monocular_Vehicle_Speed_Estimation_for_Traffic_Surveillance_ICCV_2021_paper.html

**[P4] Zhang et al. — *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (ECCV, 2022).**  
Giải thích vì sao tracking theo ID có ích khi đối tượng bị che khuất/độ tin cậy detection thay đổi.  
https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf

**[P5] *Vehicle speed measurement method using monocular cameras* (Scientific Reports, 2025).**  
Ví dụ pipeline YOLOv7 + DeepSORT và ước lượng tốc độ nhiều mục tiêu.  
https://doi.org/10.1038/s41598-025-87077-6

### Đọc để mở rộng hoặc làm phần thảo luận

**[P6] *Traffic surveillance camera calibration by 3D model bounding box alignment for accurate vehicle speed measurement* (Computer Vision and Image Understanding, 2017).**  
Nêu rõ camera calibration là nguồn sai số lớn, nghiên cứu giải pháp hiệu chuẩn nâng cao.  
https://www.sciencedirect.com/science/article/pii/S1077314217301108

**[P7] *Vision-based vehicle speed estimation: A survey* (IET Intelligent Transport Systems, 2021).**  
Bài tổng quan: phương pháp, dataset, metric và giới hạn.  
https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12079

**[P8] *Roadside Monocular Camera Calibration Based on Adaptive Structure Tensor and Hierarchical Vanishing Point Estimation* (Electronics, 2026).**  
Bài mới về tự hiệu chuẩn vanishing points; hướng mở rộng khi không đo tay nhiều mốc.  
https://www.mdpi.com/2079-9292/15/14/3215

**[P9] *An end-to-end deep learning model for vehicle speed estimation using a single camera* (2026).**  
Hướng end-to-end để so sánh định tính, **không khuyến nghị triển khai đầu tiên** trên 4GB VRAM.  
https://www.sciencedirect.com/science/article/pii/S2542660526000338

**[P10] *ForeSpeed: A Real-World Video Dataset of CCTV Cameras with Different Settings for Vehicle Speed Estimation* (Journal of Imaging, 2026).**  
Đọc về camera thực tế, nén video, fps cố định/biến thiên và benchmark khác biệt.  
https://www.mdpi.com/2313-433X/12/10/488

**[P11] *Traffic flow velocity estimation by virtually calibrated monocular surveillance cameras* (2026).**  
Có thử nghiệm trên BrnoCompSpeed và dữ liệu thu thập tại TP.HCM; tham khảo bối cảnh Việt Nam.  
https://link.springer.com/article/10.1007/s44163-026-00989-8

**[P12] *Calibration-Free Vehicle Speed Estimation: A Monocular Keypoint-Template Approach* (arXiv preprint, 2026).**  
Hướng nghiên cứu không cần mốc hiệu chuẩn truyền thống, nhưng phụ thuộc giả định và mô hình; chỉ đọc mở rộng, không xem là cách giải quyết hoàn toàn vấn đề tỷ lệ từ một ảnh vô điều kiện.  
https://arxiv.org/abs/2608.16785

**Nguồn triển khai:**
- OpenCV perspective transform: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- Ultralytics speed estimation reference: https://docs.ultralytics.com/guides/speed-estimation/
- ByteTrack code: https://github.com/FoundationVision/ByteTrack

**Ghi chú khi trích dẫn:** Nên lấy BibTeX/DOI từ trang nhà xuất bản, xác nhận tên tác giả, năm và phiên bản xuất bản trước khi đưa vào báo cáo chính thức. Không trích README/GitHub thay cho bài báo nếu có nguồn học thuật gốc.

---

## 13. Câu chuyện khoa học để bảo vệ đồ án

**Đặt vấn đề:** Một camera đơn tiết kiệm chi phí nhưng không đo trực tiếp quãng đường thực do phối cảnh.

**Đóng góp đề xuất (chỉ tuyên bố nếu thực nghiệm xác nhận):**
1. Xây dựng pipeline detection + tracking + homography + tốc độ theo timestamp và hỗ trợ nhiều xe.
2. Đánh giá định lượng tác động của **hiệu chuẩn phối cảnh** và **làm mượt quỹ đạo**.
3. Phân tích trade-off **FPS suy luận / sai số vận tốc** trên phần cứng tầm thấp.
4. Tạo bộ video demo tự quay và phân tích các trường hợp thất bại thực tế.

**Tránh tuyên bố:** “Độ chính xác 99%”, “đạt chuẩn đo phạt nguội”, “camera nào cũng không cần hiệu chuẩn”, “Jetson chạy realtime 30 FPS” nếu chưa có số liệu và điều kiện kiểm chứng.

---

## 14. Checklist khởi động ngay tuần đầu

- [ ] Cả ba đọc [P1], [P2] (ít nhất abstract, phương pháp, bảng kết quả).
- [ ] Thống nhất loại xe và hướng camera (cố định, mặt đường gần phẳng).
- [ ] Chọn một clip mẫu có metadata phù hợp; kiểm tra có được phép sử dụng.
- [ ] Thiết kế `tracks.csv` + `calibration.json` + `predictions.csv`.
- [ ] Chạy pretrained YOLO lên MP4, xác nhận detection đúng class.
- [ ] ByteTrack tạo ID xuyên các frame; ghi timestamp đúng.
- [ ] Test homography bằng 4 điểm trên mặt phẳng mô phỏng, kiểm tra mét.
- [ ] Test số học: xe đi 10 m trong 0,5 s → 72 km/h.
- [ ] Lên Git repo, `.gitignore`, `README.md`, 3 branch cá nhân.
- [ ] Hẹn một buổi demo cuối tuần: video có ID + ít nhất tốc độ sơ bộ.

---

## Quyết định đề xuất cuối cùng

**MVP = YOLO pretrained + ByteTrack + homography đo tay bằng mốc chuẩn + tốc độ theo timestamp + smoothing + đánh giá trên một phần BrnoCompSpeed.**

Nếu hoàn thành sớm mới làm thêm: fine-tune YOLO, speed regression, tự hiệu chuẩn vanishing point, Jetson Nano 2GB. Điều giúp đồ án thuyết phục hơn không phải có nhiều model, mà là **phân tích được sai số và chứng minh được từng bước cải thiện bằng số liệu có thể tái lập**.
