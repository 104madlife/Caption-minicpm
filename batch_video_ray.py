import os
import glob
import argparse
import json
from PIL import Image
from decord import VideoReader, cpu
import re
import ray

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
        match = re.search(r'\{.*\}', answer, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None

@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self, model_path, dtype, attn_impl, max_slice_nums, max_num_frames, question):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.max_num_frames = max_num_frames
        self.question = question
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation=attn_impl, torch_dtype=getattr(torch, dtype))
        self.model = self.model.to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.params = {"use_image_id": False, "max_slice_nums": max_slice_nums}
        print("Worker init done", flush=True)

    def process(self, video_path):
        try:
            frames = encode_video(video_path, self.max_num_frames)
        except Exception as e:
            return {"video": video_path, "error": f"Video encode error: {str(e)}"}
        msgs = [{'role': 'user', 'content': [self.question] + frames}]
        answer = None
        for attempt in range(3):
            try:
                with self.torch.inference_mode():
                    answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, **self.params)
                parsed_json = parse_json_answer(answer)
                if parsed_json is not None:
                    return {"video": video_path, "answer": parsed_json}
            except Exception as e:
                if attempt == 2:
                    return {"video": video_path, "error": f"Inference error: {str(e)}"}
            finally:
                self.torch.cuda.empty_cache()
        return {"video": video_path, "error": "Failed to generate valid JSON after 3 attempts", "raw_answer": answer}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-V-4_5")
    parser.add_argument("--input_dir", type=str, default="/mnt/data1/data_collection_OGameData/dynamics_count/2")
    parser.add_argument("--pattern", type=str, default="**/*.mp4")
    parser.add_argument("--question", type=str, default="Describe the video")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa","flash_attention_2"])
    parser.add_argument("--max_num_frames", type=int, default=70)
    parser.add_argument("--max_slice_nums", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/mnt/data1/data_collection_OGameData/dynamics_count/new_caption")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    question = """
    You are an expert video content analyzer focused on game scenes. Analyze the provided game video clip and output results strictly in the following JSON format (no extra text, no explanations, ensure key-value pairs match requirements):
    {
        "video_content_description": "A concise and fluent descriptive caption (see Requirements below).",
        }
    }
    Requirements:
        video_content_description: Please provide a single, continuous, and fluent textual description of the video content, with around 150 to 200 words. Avoid using any formatting, bullet points, or lists. Use natural language to describe the video in a cohesive paragraph, covering the main scene, actions, environment, characters, and any notable features. Keep the description concise yet comprehensive, ensuring it reflects the overall content of the video.
    """

    videos = sorted(glob.glob(os.path.join(args.input_dir, args.pattern), recursive=True))

    # # 测试用，随机抽取100个视频
    # import random
    # random.shuffle(videos)
    # videos = videos[:50]   # 调试用，只处理随机抽取的100个视频
    # videos.append("/mnt/data1/data_collection_OGameData/games_count/Cyberpunk_2077/0J55aRPrgOM_scene-439_1.mp4")

    if not videos:
        print("No videos found"); return

    # checkpoint-resume: skip already processed videos
    to_process = []
    for v in videos:
        base = os.path.splitext(os.path.basename(v))[0]
        json_path = os.path.join(args.output_dir, base + ".json")
        txt_path = os.path.join(args.output_dir, base + ".txt")
        def non_empty(p):
            try:
                return os.path.isfile(p) and os.path.getsize(p) > 0
            except Exception:
                return False
        if non_empty(json_path) and non_empty(txt_path):
            print(f"[SKIP] {v} already processed.")
            continue
        to_process.append(v)

    if not to_process:
        print("All videos already processed; nothing to do.")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    ray.init(ignore_reinit_error=True, include_dashboard=False, runtime_env={"env_vars": {
        "CUDA_VISIBLE_DEVICES": args.gpu_ids
    }})

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else [0]
    workers = [GPUWorker.remote(args.model_path, args.dtype, args.attn_impl, args.max_slice_nums, args.max_num_frames, question) for _ in gpu_ids]

    futures = []
    for i, v in enumerate(to_process):
        actor = workers[i % len(workers)]
        futures.append(actor.process.remote(v))

    pending = list(futures)
    total = len(pending)
    done = 0
    stats = {"total": total, "success": 0, "failure": 0, "errors": []}
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=60)
        if not ready:
            print(f"Waiting... Progress {done}/{total}", flush=True)
            continue
        item = ray.get(ready[0])
        base = os.path.splitext(os.path.basename(item['video']))[0]
        out_path = os.path.join(args.output_dir, base + ".json")
        if "error" in item:
            print(f"[ERROR] {item['video']}: {item['error']}")
            data = {"video": item['video'], "error": item['error']}
            stats["failure"] += 1
            stats["errors"].append({"video": item['video'], "error": item['error']})
        else:
            print(f"[OK] {item['video']}: {item['answer']}")
            data = {"video": item['video'], "answer": item['answer']}
            stats["success"] += 1
            desc = item['answer'].get("video_content_description")
            if isinstance(desc, str):
                txt_path = os.path.join(args.output_dir, base + ".txt")
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(desc.strip())
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        done += 1

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)
    print(f"Summary: {stats}")

    ray.shutdown()

if __name__ == "__main__":
    main()