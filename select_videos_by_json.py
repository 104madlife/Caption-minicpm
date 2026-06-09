import os
import json
import argparse
import shutil
import glob
from tqdm import tqdm

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

def matches_criteria(ans, min_cnt, max_cnt):
    cnt = to_int(ans.get("dynamic_foreground_object_count"))
    ui = to_bool(ans.get("has_game_ui_element"))
    bb = to_bool(ans.get("has_black_borders"))
    if cnt is None or ui is None or bb is None:
        return False
    return cnt >= min_cnt and cnt <= max_cnt and ui is False and bb is False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="/mnt/data1/data_collection_OGameData/processed_video2/")
    ap.add_argument("--text_dir", default="/mnt/data1/data_collection_OGameData/text_minicpm/")
    ap.add_argument("--out_video_dir", default="/mnt/data1/data_collection_OGameData/processed_video3/")
    ap.add_argument("--out_text_dir", default="/mnt/data1/data_collection_OGameData/text3_minicpm/")
    ap.add_argument("--json_pattern", default="**/*.json")
    ap.add_argument("--min_cnt", type=int, default=1)
    ap.add_argument("--max_cnt", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_video_dir, exist_ok=True)
    os.makedirs(args.out_text_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.text_dir, args.json_pattern), recursive=True))
    total = len(json_files)
    selected = []
    copied_video = 0
    copied_txt = 0
    missing_video = []
    missing_txt = []
    invalid_json = []
    not_match = []

    for jf in tqdm(json_files, total=total, desc="Processing", unit="json"):
        base = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            invalid_json.append({"json": jf, "error": str(e)})
            continue

        ans = data.get("answer", data)
        if not matches_criteria(ans, args.min_cnt, args.max_cnt):
            not_match.append(base)
            continue

        selected.append(base)
        src_video = os.path.join(args.video_dir, base + ".mp4")
        src_txt = os.path.join(args.text_dir, base + ".txt")
        dst_video = os.path.join(args.out_video_dir, base + ".mp4")
        dst_txt = os.path.join(args.out_text_dir, base + ".txt")

        if os.path.isfile(src_video):
            shutil.copy2(src_video, dst_video)
            copied_video += 1
        else:
            missing_video.append(src_video)

        if os.path.isfile(src_txt):
            try:
                shutil.copy2(src_txt, dst_txt)
                copied_txt += 1
            except Exception as e:
                missing_txt.append(f"{src_txt}: {e}")
        else:
            missing_txt.append(src_txt)
            print(f"[WARNING] Missing text file: {src_txt}")

    summary = {
        "total_json": total,
        "selected_count": len(selected),
        "selected_basenames": selected,
        "copied_video": copied_video,
        "copied_txt": copied_txt,
        "missing_video": missing_video,
        "missing_txt": missing_txt,
        "invalid_json": invalid_json,
        "not_match_count": len(not_match),
        "not_match_basenames": not_match,
        "criteria": {
            "dynamic_foreground_object_count": f">={args.min_cnt} and <={args.max_cnt}",
            "has_game_ui_element": False,
            "has_black_borders": False
        }
    }

    summary_path = os.path.join(args.out_text_dir, "selection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    print(f"[SUMMARY] {summary_path}")

if __name__ == "__main__":
    main()