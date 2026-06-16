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

The adapter supports `gpu_ids=auto`. In that mode the main repository's GPU
worker launcher probes `nvidia-smi`, chooses cards with enough free memory,
acquires local GPU leases, and passes the allocation through
`BOS_PIPELINE_ALLOCATED_GPUS`. The adapter then sets `CUDA_VISIBLE_DEVICES`
before launching `batch_video_ray.py`.

Be careful with GPU numbering: after setting `CUDA_VISIBLE_DEVICES=2,5`, many
CUDA programs see those devices as local `0,1`. Keep the semantics consistent
with `batch_video_ray.py --gpu_ids`.

## Outputs

The adapter writes local files and returns them through the runtime protocol so
the main Worker uploads them to BOS before marking the task done:

```text
captions.jsonl
failed_items.jsonl        # only when at least one item fails
remaining_items.jsonl     # only when retryable items remain after a GPU/runtime failure
_manifest.json
_SUCCESS                 # batch finished successfully, possibly with tolerated item failures
_PARTIAL                 # retryable batch-level failure with diagnostic outputs
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

Failed rows stay in the same `captions.jsonl` file with `status=failed`,
`error`, `error_type`, and `retryable` fields. Ordinary item-level failures such
as one bad video no longer force the whole framework task to fail by default.
The adapter marks the task `done` when the subprocess finishes normally and the
failure count is within the configured tolerance.

Runtime parameters controlling that tolerance:

```json
{
  "fail_on_item_failure": false,
  "max_item_failure_count": null,
  "max_item_failure_ratio": null,
  "min_success_count": 1
}
```

GPU/OOM/runtime failures are different. If `batch_video_ray.py` exits with a
non-zero code, or if an item-level error text looks like CUDA OOM / Ray worker
failure, the adapter writes partial diagnostics and marks the framework task
`failed`:

```text
captions.jsonl
failed_items.jsonl
remaining_items.jsonl
_manifest.json
_PARTIAL
batch_video_ray.log
```

`remaining_items.jsonl` is designed for the main repository's
`scripts/recover_caption_batch.py` tool. It contains only items that are safe to
retry, so recovery can create a smaller caption task instead of rerunning the
entire original batch.
