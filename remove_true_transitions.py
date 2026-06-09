import os
import json
import argparse
import glob

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
    ap.add_argument("--root", default="/mnt/data1/data_collection_OGameData/dynamics_count/")
    ap.add_argument("--counts", default="0,1,2,3,4,5,6,7,8,9")
    args = ap.parse_args()

    counts = [c.strip() for c in args.counts.split(",") if c.strip() != ""]
    summary = {"deleted": 0, "missing": 0, "folders": {}}

    for c in counts:
        folder = os.path.join(args.root, c)
        json_path = os.path.join(folder, "has_transition.json")
        deleted = 0
        missing = 0
        processed = 0
        if not os.path.isdir(folder):
            summary["folders"][c] = {"processed": 0, "deleted": 0, "missing": 0, "status": "folder_not_found"}
            continue
        if not os.path.isfile(json_path):
            summary["folders"][c] = {"processed": 0, "deleted": 0, "missing": 0, "status": "json_not_found"}
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception as e:
            summary["folders"][c] = {"processed": 0, "deleted": 0, "missing": 0, "status": f"json_error:{e}"}
            continue
        if not isinstance(mapping, dict):
            summary["folders"][c] = {"processed": 0, "deleted": 0, "missing": 0, "status": "json_not_dict"}
            continue
        for name, flag in mapping.items():
            processed += 1
            b = to_bool(flag)
            if b is True:
                path = os.path.join(folder, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        deleted += 1
                    except Exception:
                        missing += 1
                else:
                    missing += 1
        summary["folders"][c] = {"processed": processed, "deleted": deleted, "missing": missing, "status": "ok"}
        summary["deleted"] += deleted
        summary["missing"] += missing

    out_path = os.path.join(args.root, "remove_true_transitions_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(out_path)

if __name__ == "__main__":
    main()
