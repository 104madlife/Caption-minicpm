import argparse
import base64
import math
from io import BytesIO
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import torch

MAX_NUM_FRAMES = 82
# 尝试导入 llama_cpp，如果未安装则提示
# try:
#     from llama_cpp import Llama
#     from llama_cpp.llama_chat_format import Llava15ChatHandler
# except ImportError:
#     print("Error: 请先安装 llama-cpp-python 库: pip install llama-cpp-python")
#     exit(1)

# ==========================================
# 1. 视频处理函数 (复用官方 PyTorch 逻辑)
# ==========================================
def encode_video(video_path, target_num_frames=MAX_NUM_FRAMES):
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
    print('num frames:', len(frames))
    return frames

def image_to_base64_data_uri(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"

# ==========================================
# 2. 主推理逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.5 GGUF Video Inference")
    parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-V-4_5", help="Path to main GGUF model file")
    # parser.add_argument("--mmproj_path", type=str, required=True, help="Path to multimodal projector GGUF file")
    parser.add_argument("--video_path", type=str, default="/mnt/data1/OGameData/slice30/NawS7M0MwzI_scene-169_1.mp4", help="Path to input video")
    # parser.add_argument("--prompt", type=str, default="Describe this video in detail.", help="User prompt")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample (lower is safer for GGUF context)")
    parser.add_argument("--num_frames", type=int, default=75, help="Number of frames to sample from the video")

    args = parser.parse_args()

    model_path = args.model_path
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',  # sdpa or flash_attention_2
        torch_dtype=torch.float16,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    video_path = args.video_path
    video_path = "/mnt/data1/data_collection/OGameData50/slice/1A6cT-BOkO4_scene-13_1.mp4"
    frames = encode_video(video_path, args.num_frames)

    # 提问prompt
    question = """
    You are a professional video content analyzer focused on game scenes. Analyze the provided game video clip and output results strictly in the following JSON format (no extra text, no explanations, ensure key-value pairs match requirements):

    {
        "video_content_description": "Detailed and vivid description of the game video content (150-250 words), covering: 1) Core scene setting (e.g., jungle, castle, battlefield) and key environmental elements (types of vegetation, structures, terrain); 2) Spatial layout (foreground/background/left-right/center positions of key objects); 3) Visual details (colors, textures, lighting effects, atmosphere); 4) Character/entity details (appearances, equipment, states if any); 5) Dynamic actions (attack, movement, interaction) or lack thereof.",
        "dynamic_foreground_object_categories": ["List ONLY the categories of dynamic foreground objects (single word/phrase per category; merge same categories, e.g., 3 moving people → ['people'], a running dog + flying bird → ['dog', 'bird'])", "If no dynamic foreground objects, use empty array []"],
        "dynamic_foreground_object_count": "Integer value (0, 1, 2, ...) – total number of individual dynamic foreground objects (count each moving entity separately, regardless of category; e.g., 3 moving people → 3, 1 dog + 1 bird → 2, no moving objects → 0)",
        "has_game_ui_element": "Only answer 'true' or 'false' (UI elements include game menus, health bars, score displays, minimaps, buttons, text overlays, etc.)",
        "has_black_borders": "Only answer 'true' or 'false' (refers to black borders at the top, bottom, left, or right of the video frame)"
    }

    Requirements:
        video_content_description: video_content_description: Cover all 5 required dimensions without redundancy; use specific vocabulary (avoid vague terms like 'plants' → specify 'ferns, broad-leafed plants'); maintain a coherent narrative flow; keep within 150-250 words.
        dynamic_foreground_object_categories: Do not list specific quantities or detailed descriptions, only object categories; use lowercase unless it's a proper noun (e.g., game-specific character names).
        dynamic_foreground_object_count: Count each individual moving entity as one (ignore category; e.g., 2 players + 3 monsters = 5); if no dynamic objects, output 0; do not use text, only numeric value.
        Strictly adhere to JSON syntax (correct commas, quotes, brackets) – invalid JSON will be considered useless.
        Answer based solely on the video content; do not make assumptions if information is unclear (e.g., if UI is ambiguous, answer 'false').

    """

    msg = [
        {'role': 'user', 'content': [question] + frames},
    ]

    # Set decode params for video
    params = {}
    params["use_image_id"] = False
    # use 1 if cuda OOM and video resolution > 448*448
    params["max_slice_nums"] = 2

    res = model.chat(
        image=None,
        msgs=msg,
        tokenizer=tokenizer,
        **params
    )

    print("Response:")
    print(res)

if __name__ == "__main__":
    main()