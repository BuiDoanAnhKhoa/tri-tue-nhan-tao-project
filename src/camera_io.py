import cv2


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open {path}")
    previous = -1
    try:
        frame_id = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if seconds <= previous:
                raise ValueError(f"Invalid video timestamp at frame {frame_id}")
            yield frame_id, seconds, frame
            previous = seconds
            frame_id += 1
    finally:
        capture.release()
