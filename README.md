# Bridge traffic speed estimation

YOLO11n detects five vehicle classes, ByteTrack assigns IDs, and a calibrated road region converts vehicle motion to km/h. Training figures, configuration and CPU/GPU benchmarks are in report_visdrone/.

Run from the repository root in the lane-det environment with Python 3.10, Ultralytics, PyTorch, OpenCV, NumPy and lap. Supply the original fixed-camera video locally.

1. Export detections with the final model.

       python -m src.detector_tracker IMG_9471.MOV runs/traffic/weights/best.pt outputs/bridge_tracks.csv

   Create the outputs directory first. The CSV uses zero-based frames, recording timestamps and original-resolution pixel coordinates.

2. Calibrate using four identifiable points on the same flat road surface. Measure their coordinates in a common metric coordinate system. For a measured rectangle use (0,0), (length,0), (length,width), (0,width). Choose points around the measurement area. Use the full decoded frame in the same orientation. The GUI accepts eight numbers following --meters, in the same order as the four clicks.

       python -m src.calibrate outputs/bridge_calibration_frame.jpg outputs/calibration.json --meters X1 Y1 X2 Y2 X3 Y3 X4 Y4

3. Generate the speed overlay and per-observation CSV.

       python -m src.road_speed IMG_9471.MOV outputs/bridge_tracks.csv outputs/calibration.json outputs

   The box bottom center serves as the ground contact estimate. Estimates start after half a second inside the calibrated region. The estimator resets on gaps over 0.25 seconds and takes a five-estimate median for display. A moved camera or a different road plane needs its own calibration.

4. Evaluate with independently measured vehicle passages from this same clip. Prepare a CSV with track_id,start_s,end_s,reference_kmh. Leave the ID empty for missed passages. One reference option uses two independently measured road marks and manually timed crossings, with reference_kmh = 3.6 * distance_m / elapsed_s.

       python -m src.evaluate_speed outputs/speeds.csv reference_passages.csv

   Compare raw-window and smoothed speeds, record matched passage coverage, and inspect missed motorcycles and ID switches. VisDrone detection scores do not measure speed accuracy.
