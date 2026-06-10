# MiniCPM pipeline adapters

This directory contains adapters used by the BOS GBuffer data pipeline. The main
pipeline should call these adapters through the subprocess runtime protocol. It
should not import MiniCPM internals directly.

## Adapter: `run_caption_from_index.py`

`run_caption_from_index.py` consumes the output of the pipeline indexer stage:

```text
build_caption_index_v1 -> minicpm_caption_v1
```

The indexer writes a prepared JSONL item list. Each line should contain at least:

```json
{
  "item_id": "2026-04-21 00-11-18__GT__slice_000000",
  "video_key": "test/output/channel_slice_v1/GT/2026-04-21 00-11-18/slice_000000.mp4",
  "local_video_path": "/data/bc/pipeline_cache/caption_inputs/smoke/GT/2026-04-21 00-11-18/2026-04-21_00-11-18__GT__slice_000000.mp4",
  "prepared": true
}
```

The adapter reads the prepared list from the previous stage artifacts:

```text
artifacts.build_caption_index_v1.prepared_items_local_path
artifacts.build_caption_index_v1.prepared_items_key
```

For the current smoke test, the local path is preferred and must exist on the
same machine. This is intentional: the pipeline is optimized for stable
single-machine batch execution rather than cross-machine task migration.

## Runtime parameters

Recommended smoke-test parameters:

```json
{
  "model_path": "/data/models/openbmb/MiniCPM-V-4_5",
  "gpu_ids": "0,1",
  "cuda_visible_devices": "0,1",
  "dtype": "bfloat16",
  "attn_impl": "sdpa",
  "max_num_frames": 38,
  "max_slice_nums": 1,
  "limit_items": 2,
  "hf_home": "/data/cache/huggingface",
  "pytorch_cuda_alloc_conf": "expandable_segments:True",
  "output_prefix": "test/output/minicpm_caption_v1/smoke/"
}
```

The adapter calls:

```bash
python batch_video_ray.py \
  --model_path /data/models/openbmb/MiniCPM-V-4_5 \
  --input_list <local_jsonl> \
  --output_dir <local_output_dir> \
  --gpu_ids 0,1 \
  --dtype bfloat16 \
  --attn_impl sdpa \
  --max_num_frames 38 \
  --max_slice_nums 1
```

## GPU selection

The current smoke test uses explicit GPU IDs. A future adapter version can add an
`auto` mode that calls `nvidia-smi` before launching MiniCPM, chooses cards with
enough free memory, then sets `CUDA_VISIBLE_DEVICES`.

Be careful with GPU numbering: after setting `CUDA_VISIBLE_DEVICES=2,5`, many
CUDA programs see those devices as local `0,1`. Keep the semantics consistent
with `batch_video_ray.py --gpu_ids`.

## Outputs

The adapter writes local files and returns them through the runtime protocol so
the main Worker uploads them to BOS before marking the task done:

```text
captions.jsonl
_manifest.json
_SUCCESS
batch_video_ray.log
```

`captions.jsonl` contains one row per item. Successful rows include:

```json
{
  "item_id": "...",
  "video_key": "...",
  "local_video_path": "...",
  "status": "done",
  "caption": "...",
  "answer": {
    "video_content_description": "..."
  }
}
```

