import argparse
import base64
import math
from io import BytesIO
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

# 尝试导入 llama_cpp，如果未安装则提示
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    print("Error: 请先安装 llama-cpp-python 库: pip install llama-cpp-python")
    exit(1)

# ==========================================
# 1. 视频处理函数 (复用官方 PyTorch 逻辑)
# ==========================================
def encode_video(video_path, choose_fps=3, max_num_frames=60):
    """
    从视频中提取帧。
    注意：GGUF 模式下我们主要关注帧的提取，暂时忽略 temporal_ids 的复杂分组，
    而是将视频作为图像序列输入。
    """
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps

    # 计算需要提取的帧数
    # 这里简化逻辑，确保不超过 max_num_frames 以避免上下文溢出
    total_frames = min(int(video_duration * choose_fps), max_num_frames)
    if total_frames < 1: total_frames = 1

    frame_idx = [i for i in range(0, len(vr))]
    frame_idx = uniform_sample(frame_idx, total_frames)

    print(f"Processing video: {video_path}")
    print(f"Duration: {video_duration:.2f}s, Extracting {len(frame_idx)} frames")

    frames = vr.get_batch(frame_idx).asnumpy()
    pil_frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]

    return pil_frames

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
    parser.add_argument("--model_path", type=str, required=True, help="Path to main GGUF model file")
    parser.add_argument("--mmproj_path", type=str, required=True, help="Path to multimodal projector GGUF file")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.", help="User prompt")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample (lower is safer for GGUF context)")

    args = parser.parse_args()

    # 1. 初始化 Llama 模型
    # 注意：n_ctx 需要设置得足够大以容纳视频帧。
    # MiniCPM-V 4.5 的投影器通常能很好地压缩图像，但在 GGUF 中可能取决于实现。
    # 建议至少 4096 或 8192。
    chat_handler = Llava15ChatHandler(clip_model_path=args.mmproj_path)

    llm = Llama(
        model_path=args.model_path,
        chat_handler=chat_handler,
        n_ctx=8192,
        n_gpu_layers=-1, # 使用所有 GPU 层，如果显存不足请调小
        verbose=True
    )

    # 2. 处理视频
    frames = encode_video(args.video_path, choose_fps=args.fps)

    # 3. 构建输入消息
    # 将所有帧作为图像内容加入到 content 列表中
    content = []
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_base64_data_uri(frame)
            }
        })

    # 添加文本提示
    content.append({
        "type": "text",
        "text": args.prompt
    })

    messages = [
        {"role": "system", "content": "You are a helpful assistant capable of understanding videos."},
        {"role": "user", "content": content}
    ]

    # 4. 执行推理
    print("\nStarting inference...")
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    print("\nRefined Result:")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()