import os
import glob
import json
import argparse
from collections import Counter
import shutil

def to_int(v):
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v.strip())
        except Exception:
            return None
    return None

def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true","1","yes","y","t"):
            return True
        if s in ("false","0","no","n","f"):
            return False
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_dir", default="/mnt/data1/data_collection_OGameData/text_minicpm/")
    ap.add_argument("--pattern", default="**/*.json")
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--video_dir", default="/mnt/data1/data_collection_OGameData/processed_video2/")
    ap.add_argument("--dest_root", default="/mnt/data1/data_collection_OGameData/dynamics_count/")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.text_dir, args.pattern), recursive=True))
    counter = Counter()
    invalid = 0
    copied_by_count = Counter()
    copied_ui_or_borders = 0
    missing_video = []
    os.makedirs(args.dest_root, exist_ok=True)
    ui_dir = os.path.join(args.dest_root, "ui_or_borders")
    os.makedirs(ui_dir, exist_ok=True)

    for jf in files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            invalid += 1
            continue
        ans = data.get("answer", data)
        cnt = to_int(ans.get("dynamic_foreground_object_count"))
        ui_b = to_bool(ans.get("has_game_ui_element"))
        bb_b = to_bool(ans.get("has_black_borders"))
        if cnt is None:
            invalid += 1
            continue
        counter[cnt] += 1
        base = os.path.splitext(os.path.basename(jf))[0]
        src_video = os.path.join(args.video_dir, base + ".mp4")
        if ui_b is False and bb_b is False:
            dest_dir = os.path.join(args.dest_root, str(cnt))
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isfile(src_video):
                shutil.copy2(src_video, os.path.join(dest_dir, base + ".mp4"))
                copied_by_count[cnt] += 1
            else:
                missing_video.append(src_video)
        elif ui_b is True or bb_b is True:
            if os.path.isfile(src_video):
                shutil.copy2(src_video, os.path.join(ui_dir, base + ".mp4"))
                copied_ui_or_borders += 1
            else:
                missing_video.append(src_video)

    total = sum(counter.values())
    print("Total counted:", total)
    for k in sorted(counter.keys()):
        print(f"{k}: {counter[k]}")
    print("Invalid or missing:", invalid)
    print("Copied by count:")
    for k in sorted(copied_by_count.keys()):
        print(f"{k}: {copied_by_count[k]}")
    print("Copied to ui_or_borders:", copied_ui_or_borders)
    print("Missing video files:", len(missing_video))

    if args.out_path:
        out = {
            "total_counted": total,
            "counts": {str(k): counter[k] for k in sorted(counter.keys())},
            "invalid_or_missing": invalid,
            "source_dir": args.text_dir
        }
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()