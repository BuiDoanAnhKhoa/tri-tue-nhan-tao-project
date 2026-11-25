"""Compare median track speed over each manually identified reference passage."""
import argparse
import csv
from collections import defaultdict
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("speeds")
    parser.add_argument("references")
    args = parser.parse_args()
    by_track = defaultdict(list)
    with open(args.speeds) as f:
        for row in csv.DictReader(f):
            if row["speed_kmh"]:
                by_track[int(row["track_id"])].append(row)
    errors = []
    total = 0
    with open(args.references) as f:
        for ref in csv.DictReader(f):
            total += 1
            start,end = float(ref["start_s"]),float(ref["end_s"])
            if end <= start:
                raise ValueError("Reference end time must follow start")
            rows = [r for r in by_track.get(ref["track_id"] and int(ref["track_id"]), [])
                    if start <= float(r["timestamp_s"]) <= end]
            if rows:
                errors.append([np.median([float(r[key]) for r in rows])-float(ref["reference_kmh"])
                               for key in ("raw_kmh","speed_kmh")])
    print(f"Matched {len(errors)} of {total} reference passages")
    if errors:
        for name, e in zip(("Window", "Median-smoothed window"),np.asarray(errors).T):
            print(f"{name}: MAE {np.abs(e).mean():.2f}, RMSE {np.sqrt((e**2).mean()):.2f}, bias {e.mean():.2f} km/h")


if __name__ == "__main__":
    main()
