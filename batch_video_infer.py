import os
import glob
import argparse
import json
import multiprocessing as mp
import sys
from PIL import Image
from decord import VideoReader, cpu
# import torch (Moved inside worker)
from transformers import AutoModel, AutoTokenizer

import re

def encode_video(video_path, target_num_frames):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    target = int(target_num_frames)
    if total_frames <= target:
        frame_idx = list(range(total_frames))
    else:
        frame_idx = uniform_sample(range(total_frames), target)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames), flush=True)
    return frames

def parse_json_answer(answer):
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        # Try to find the first '{' and the last '}'
        match = re.search(r'\{.*\}', answer, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None

# sanitize helpers
def sanitize_text(s):
    if not isinstance(s, str):
        return s
    try:
        return s.encode('utf-8', 'replace').decode('utf-8')
    except Exception:
        return s

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, str):
        return sanitize_text(obj)
    return obj

def gpu_worker(gpu_id, task_q, result_q, model_path, dtype, attn_impl, max_slice_nums, max_num_frames, question):
    print(f"[Worker {gpu_id}] Starting...", flush=True)
    try:
        import faulthandler, signal
        faulthandler.enable()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # os.environ.setdefault('DISABLE_FLASH_ATTENTION','1')
        # os.environ.setdefault('FLASH_ATTENTION_FORCE_DISABLE','1')
        # os.environ.setdefault('XFORMERS_DISABLE_FLASH_ATTN','1')
        import torch
        print(f"[Worker {gpu_id}] Loading model from {model_path}...", flush=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          attn_implementation=attn_impl,
                                          torch_dtype=getattr(torch, dtype))
        print(f"[Worker {gpu_id}] Model object created.", flush=True)
        print(f"[Worker {gpu_id}] Moving model to CUDA...", flush=True)
        model = model.to("cuda").eval()
        print(f"[Worker {gpu_id}] Model moved to CUDA.", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[Worker {gpu_id}] Tokenizer loaded.", flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside worker!")
        print(f"[Worker {gpu_id}] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        torch.zeros(1, device='cuda')
        print(f"[Worker {gpu_id}] CUDA allocation test passed.", flush=True)

        params = {"use_image_id": False, "max_slice_nums": max_slice_nums}

        while True:
            video_path = task_q.get()
            if isinstance(video_path, str):
                print(f"[Worker {gpu_id}] Processing: {os.path.basename(video_path)}", flush=True)
            if video_path is None:
                break

            frames = None
            try:
                frames = encode_video(video_path, 41)
            except Exception as e:
                 result_q.put({"video": video_path, "error": f"Video encode error: {str(e)}"})
                 continue

            success = False
            msgs = [{'role': 'user', 'content': [question] + frames}]

            # Retry logic for JSON generation
            for attempt in range(3):
                try:
                    with torch.inference_mode():
                        answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)

                    parsed_json = parse_json_answer(answer)
                    if parsed_json is not None:
                        result_q.put({"video": video_path, "answer": parsed_json})
                        success = True
                        break
                    else:
                        # If it's the last attempt, we will fall through to failure
                        pass
                except Exception as e:
                    if attempt == 2:
                        result_q.put({"video": video_path, "error": f"Inference error: {str(e)}"})
                        success = True # Handled as error
                        break
                finally:
                    torch.cuda.empty_cache()

            if not success:
                result_q.put({"video": video_path, "error": "Failed to generate valid JSON after 3 attempts", "raw_answer": answer})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Worker {gpu_id}] Critical error: {e}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-V-4_5")
    parser.add_argument("--input_dir", type=str, default="/mnt/data1/data_collection_OGameData/dynamics_count/0")
    parser.add_argument("--pattern", type=str, default="**/*.mp4")
    parser.add_argument("--question", type=str, default="Describe the video")
    parser.add_argument("--gpu_ids", type=str, default="2", help="e.g. 0,1,2")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa","flash_attention_2"])
    parser.add_argument("--max_num_frames", type=int, default=81)
    parser.add_argument("--max_slice_nums", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/mnt/data1/data_collection_OGameData/dynamics_count/new_caption", help="Directory to save per-video JSON results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    question = """
    You are a professional video content analyzer focused on game scenes. Analyze the provided game video clip and output results strictly in the following JSON format (no extra text, no explanations, ensure key-value pairs match requirements):

    {
        "video_content_description": "Concise and accurate description of the game video content (e.g., 'A player controls a warrior character fighting three monsters in a forest scene, with sword-swinging animations and monster attack effects').",
        "dynamic_foreground_object_categories": ["List ONLY the categories of dynamic foreground objects (single word/phrase per category; merge same categories, e.g., 3 moving people → ['people'], a running dog + flying bird → ['dog', 'bird'])", "If no dynamic foreground objects, use empty array []"],
        "dynamic_foreground_object_count": "Integer value (0, 1, 2, ...) – total number of individual dynamic foreground objects (count each moving entity separately, regardless of category; e.g., 3 moving people → 3, 1 dog + 1 bird → 2, no moving objects → 0)",
        "has_game_ui_element": "Only answer 'true' or 'false' (UI elements include game menus, health bars, score displays, minimaps, buttons, text overlays, etc.)",
        "has_black_borders": "Only answer 'true' or 'false' (refers to black borders at the top, bottom, left, or right of the video frame)"
    }

    Requirements:
        video_content_description: Avoid redundancy, focus on core scene, characters, and key actions (within 100 words).
        dynamic_foreground_object_categories: Do not list specific quantities or detailed descriptions, only object categories; use lowercase unless it's a proper noun (e.g., game-specific character names).
        dynamic_foreground_object_count: Count each individual moving entity as one (ignore category; e.g., 2 players + 3 monsters = 5); if no dynamic objects, output 0; do not use text, only numeric value.
        Strictly adhere to JSON syntax (correct commas, quotes, brackets) – invalid JSON will be considered useless.
        Answer based solely on the video content; do not make assumptions if information is unclear (e.g., if UI is ambiguous, answer 'false').

    """

    videos = sorted(glob.glob(os.path.join(args.input_dir, args.pattern), recursive=True))
    if not videos:
        print("No videos found"); return

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else [0] # list(range(torch.cuda.device_count())) -> avoid torch init in main
    q_capacity = max(2 * len(gpu_ids), len(videos) + len(gpu_ids))
    task_q = mp.Queue(maxsize=q_capacity)
    result_q = mp.Queue()
    print(f"Prepared task queue with capacity {q_capacity}")

    mp.set_start_method("spawn", force=True)
    mp.set_executable(sys.executable)
    print(sys.executable)
    workers = []
    for gid in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gid, task_q, result_q, args.model_path, args.dtype, args.attn_impl,
                                                args.max_slice_nums, args.max_num_frames, question))
        p.daemon = False
        p.start()
        print(f"Started worker for GPU {gid} with PID {p.pid}", flush=True)
        workers.append(p)

    print(f"Enqueuing {len(videos)} tasks...", flush=True)
    for v in videos:
        task_q.put(v)
    for _ in workers:
        task_q.put(None)
    print("All tasks enqueued.", flush=True)

    done = 0
    total = len(videos)
    while done < total:
        try:
            item = result_q.get(timeout=60)
        except Exception:
            alive = sum(1 for p in workers if p.is_alive())
            print(f"Waiting for results... Progress {done}/{total}. Alive workers: {alive}", flush=True)
            if alive == 0:
                exit_info = ", ".join([f"PID {p.pid} exitcode {p.exitcode}" for p in workers])
                print(f"All workers exited; aborting. Exit info: {exit_info}", flush=True)
                break
            continue
        base = os.path.splitext(os.path.basename(item['video']))[0]
        out_path = os.path.join(args.output_dir, base + ".json")
        if "error" in item:
            print(f"[ERROR] {item['video']}: {item['error']}")
            data = {"video": item['video'], "error": item['error']}
        else:
            print(f"[OK] {item['video']}: {item['answer']}")
            data = {"video": item['video'], "answer": item['answer']}
            desc = item['answer'].get("video_content_description")
            if isinstance(desc, str):
                txt_path = os.path.join(args.output_dir, base + ".txt")
                with open(txt_path, "w", encoding="utf-8", errors="replace") as tf:
                    tf.write(sanitize_text(desc).strip())
        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(sanitize_json(data), f, ensure_ascii=False, indent=2)
        done += 1

    for p in workers:
        p.join()

if __name__ == "__main__":
    main()