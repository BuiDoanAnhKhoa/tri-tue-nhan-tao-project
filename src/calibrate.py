"""Click four measured road points in perimeter order on a full-resolution frame."""
import argparse
import json
from pathlib import Path

import cv2
from .road_speed import geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("output")
    parser.add_argument("--meters", required=True, type=float, nargs=8, metavar="M")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise OSError(args.image)
    points = []
    def click(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
    title = "Click measured corners in order | U undo | Enter save | Esc cancel"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)
    cv2.setMouseCallback(title, click)
    try:
        while True:
            display = image.copy()
            for i, point in enumerate(points, 1):
                cv2.circle(display, point, 5, (0,255,255), -1)
                cv2.putText(display,str(i),point,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.imshow(title, display)
            key = cv2.waitKey(30) & 255
            if key == 27:
                return
            if key == ord("u") and points:
                points.pop()
            if key in (10,13) and len(points) == 4:
                config = dict(frame_size=list(image.shape[1::-1]),image_xy=points,
                              road_xy_m=[args.meters[i:i+2] for i in range(0,8,2)])
                geometry(config)
                Path(args.output).write_text(json.dumps(config,indent=2))
                return
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
