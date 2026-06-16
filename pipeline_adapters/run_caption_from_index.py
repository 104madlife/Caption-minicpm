"""Pipeline adapter for running MiniCPM caption from a prepared JSONL index."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


GPU_OOM_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "failed to allocate",
    "out of memory",
)
GPU_RUNTIME_PATTERNS = (
    "rayactorerror",
    "actor died",
    "worker died",
    "node died",
    "illegal memory access",
    "device-side assert",
    "cuda error",
    "nccl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MiniCPM caption from pipeline prepared items.")
    parser.add_argument("--input", required=True, help="Path to runtime_input.json.")
    parser.add_argument("--output", required=True, help="Path to runtime_output.json.")
    return parser


def ensure_trailing_slash(prefix: str) -> str:
    value = prefix.strip().lstrip("/")
    if not value:
        return ""
    return value if value.endswith("/") else f"{value}/"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_runtime_output(
    path: Path,
    *,
    status: str,
    artifacts: dict[str, Any] | None = None,
    output_files: list[dict[str, str]] | None = None,
    reason: str | None = None,
) -> None:
    write_json(
        path,
        {
            "status": status,
            "artifacts": artifacts or {},
            "output_files": output_files or [],
            "next_stage_override": None,
            "stop_pipeline": False,
            "reason": reason,
        },
    )


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def parse_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def safe_item_stem(item_id: Any) -> str:
    value = str(item_id or "")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._") or "item"


def jsonl_line(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(jsonl_line(row) for row in rows), encoding="utf-8")


def classify_error_text(text: str, *, default: str = "algorithm_error") -> tuple[str, bool]:
    lowered = text.lower()
    if any(pattern in lowered for pattern in GPU_OOM_PATTERNS):
        return "gpu_oom", True
    if any(pattern in lowered for pattern in GPU_RUNTIME_PATTERNS):
        return "gpu_runtime", True
    if "no space left on device" in lowered:
        return "local_disk_full", False
    if "file not found" in lowered or "no such file" in lowered:
        return "input_missing", False
    return default, False


def classify_subprocess_failure(returncode: int, stdout: str, stderr: str) -> tuple[str, bool]:
    if returncode < 0:
        return "process_signal", True
    failure_type, retryable = classify_error_text(
        "\n".join([stdout, stderr]),
        default="subprocess_failed",
    )
    return failure_type, retryable


def nested_get(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing runtime input path: {dotted_path}")
        current = current[part]
    return current


def read_prepared_items(runtime_input: dict[str, Any], work_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    task = runtime_input.get("task") or {}
    artifacts = task.get("artifacts") or {}
    index_artifacts = artifacts.get("build_caption_index_v1") or {}
    params = runtime_input.get("parameters") or {}
    local_path_value = (
        params.get("prepared_items_local_path")
        or index_artifacts.get("prepared_items_local_path")
    )
    key_value = params.get("prepared_items_key") or index_artifacts.get("prepared_items_key")

    if local_path_value and Path(str(local_path_value)).is_file():
        prepared_path = Path(str(local_path_value))
    elif key_value:
        prepared_path = work_dir / "prepared_items.jsonl"
        from bos_gbuffer_pipeline import storage  # type: ignore

        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_path.write_text(storage.read_text(str(key_value)), encoding="utf-8")
    else:
        raise RuntimeError("Missing prepared_items_local_path or prepared_items_key.")

    items = [
        json.loads(line)
        for line in prepared_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise RuntimeError(f"Prepared item list is empty: {prepared_path}")

    for item in items:
        local_video_path = Path(str(item.get("local_video_path") or ""))
        if not local_video_path.is_file():
            raise FileNotFoundError(f"Prepared video is missing: {local_video_path}")

    return items, prepared_path


def index_run_id(runtime_input: dict[str, Any]) -> str | None:
    task = runtime_input.get("task") or {}
    artifacts = task.get("artifacts") or {}
    index_artifacts = artifacts.get("build_caption_index_v1") or {}
    value = index_artifacts.get("run_id") or (task.get("batch") or {}).get("run_id") or task.get("task_id")
    return str(value).strip() if value else None


def resolve_gpu_ids(params: dict[str, Any]) -> tuple[str, str]:
    requested_gpu_ids = str(params.get("gpu_ids") or "auto").strip()
    requested_cuda_visible = str(params.get("cuda_visible_devices") or "auto").strip()
    allocated = os.environ.get("BOS_PIPELINE_ALLOCATED_GPUS", "").strip()

    if requested_gpu_ids.lower() == "auto":
        if not allocated:
            raise RuntimeError(
                "gpu_ids=auto requires BOS_PIPELINE_ALLOCATED_GPUS from the GPU worker launcher."
            )
        gpu_ids = allocated
        gpu_mode = "auto"
    else:
        gpu_ids = requested_gpu_ids
        gpu_mode = "explicit"

    if requested_cuda_visible.lower() == "auto":
        cuda_visible_devices = gpu_ids
    else:
        cuda_visible_devices = requested_cuda_visible

    os.environ["BOS_PIPELINE_ALLOCATED_GPUS"] = gpu_ids
    os.environ["BOS_PIPELINE_ALLOCATED_GPU_COUNT"] = str(len([item for item in gpu_ids.split(",") if item.strip()]))
    return gpu_ids, gpu_mode if cuda_visible_devices == gpu_ids else f"{gpu_mode}:custom_cuda_visible"


def write_input_list(path: Path, items: list[dict[str, Any]], *, limit_items: int | None) -> list[dict[str, Any]]:
    selected = items[:limit_items] if limit_items is not None else items
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in selected),
        encoding="utf-8",
    )
    return selected


def caption_from_answer(answer: Any) -> str | None:
    if isinstance(answer, dict):
        value = answer.get("video_content_description")
        return value if isinstance(value, str) else None
    return None


def aggregate_outputs(
    *,
    selected_items: list[dict[str, Any]],
    batch_output_dir: Path,
    caption_jsonl_path: Path,
    failed_items_path: Path,
    remaining_items_path: Path,
    subprocess_failed: bool,
    subprocess_failure_type: str | None,
    subprocess_retryable: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    remaining_rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    missing_output_count = 0
    retryable_failure_count = 0

    for item in selected_items:
        item_id = str(item.get("item_id") or "")
        safe_item_id = safe_item_stem(item_id)
        output_path = batch_output_dir / f"{safe_item_id}.json"
        if not output_path.is_file():
            error = f"Missing batch output: {output_path}"
            retryable = bool((subprocess_failed and subprocess_retryable) or not subprocess_failed)
            row = {
                **item,
                "status": "failed",
                "error": error,
                "error_type": subprocess_failure_type or "missing_output",
                "retryable": retryable,
            }
            rows.append(row)
            failed_rows.append(row)
            if retryable:
                remaining_rows.append(item)
                retryable_failure_count += 1
            failure_count += 1
            missing_output_count += 1
            continue

        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            row = {
                **item,
                "status": "failed",
                "error": f"Invalid batch output JSON: {type(exc).__name__}: {exc}",
                "error_type": "output_json_invalid",
                "retryable": True,
            }
            rows.append(row)
            failed_rows.append(row)
            remaining_rows.append(item)
            failure_count += 1
            retryable_failure_count += 1
            continue
        if data.get("error"):
            error = str(data["error"])
            error_type, retryable = classify_error_text(error)
            row = {
                **item,
                "status": "failed",
                "error": error,
                "error_type": error_type,
                "retryable": retryable,
            }
            rows.append(row)
            failed_rows.append(row)
            if retryable:
                remaining_rows.append(item)
                retryable_failure_count += 1
            failure_count += 1
            continue

        caption = caption_from_answer(data.get("answer"))
        rows.append(
            {
                **item,
                "status": "done",
                "caption": caption,
                "answer": data.get("answer"),
            }
        )
        success_count += 1

    caption_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(caption_jsonl_path, rows)
    write_jsonl(failed_items_path, failed_rows)
    write_jsonl(remaining_items_path, remaining_rows)
    return {
        "success_count": success_count,
        "failure_count": failure_count,
        "missing_output_count": missing_output_count,
        "retryable_failure_count": retryable_failure_count,
        "rows": rows,
        "failed_rows": failed_rows,
        "remaining_rows": remaining_rows,
    }


def should_fail_framework_task(
    *,
    params: dict[str, Any],
    total_item_count: int,
    success_count: int,
    failure_count: int,
    retryable_failure_count: int,
    subprocess_returncode: int,
) -> tuple[bool, str | None]:
    if subprocess_returncode != 0:
        return True, f"MiniCPM subprocess exited with code {subprocess_returncode}."
    if retryable_failure_count:
        return True, f"MiniCPM produced {retryable_failure_count} retryable item failure(s)."

    min_success_count = int(params.get("min_success_count") or 1)
    if success_count < min_success_count:
        return True, f"MiniCPM success_count={success_count} is below min_success_count={min_success_count}."

    if failure_count <= 0:
        return False, None

    if parse_bool(params.get("fail_on_item_failure"), default=False):
        return True, f"MiniCPM caption failed for {failure_count} item(s)."

    max_failure_count = parse_optional_int(params.get("max_item_failure_count"))
    if max_failure_count is not None and failure_count > max_failure_count:
        return True, (
            f"MiniCPM item failure_count={failure_count} exceeds "
            f"max_item_failure_count={max_failure_count}."
        )

    max_failure_ratio = parse_optional_float(params.get("max_item_failure_ratio"))
    if max_failure_ratio is not None and total_item_count > 0:
        failure_ratio = failure_count / total_item_count
        if failure_ratio > max_failure_ratio:
            return True, (
                f"MiniCPM item failure_ratio={failure_ratio:.6f} exceeds "
                f"max_item_failure_ratio={max_failure_ratio:.6f}."
            )

    return False, None


def main() -> int:
    args = build_parser().parse_args()
    main_repo_root = Path(__file__).resolve().parents[3]
    if str(main_repo_root) not in sys.path:
        sys.path.insert(0, str(main_repo_root))

    runtime_input_path = Path(args.input)
    runtime_output_path = Path(args.output)
    runtime_input = json.loads(runtime_input_path.read_text(encoding="utf-8"))
    params = runtime_input.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}

    work_dir = Path(str(runtime_input["work_dir"]))
    output_dir = Path(str(runtime_input["output_dir"]))
    batch_output_dir = output_dir / "batch_outputs"
    input_list_path = output_dir / "caption_input_items.jsonl"
    caption_jsonl_path = output_dir / "captions.jsonl"
    failed_items_path = output_dir / "failed_items.jsonl"
    remaining_items_path = output_dir / "remaining_items.jsonl"
    manifest_path = output_dir / "_manifest.json"
    success_path = output_dir / "_SUCCESS"
    partial_path = output_dir / "_PARTIAL"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        items, prepared_path = read_prepared_items(runtime_input, work_dir)
        limit_items = int(params["limit_items"]) if params.get("limit_items") is not None else None
        selected_items = write_input_list(input_list_path, items, limit_items=limit_items)

        repo_root = Path(__file__).resolve().parents[1]
        batch_script = repo_root / "batch_video_ray.py"
        gpu_ids, gpu_mode = resolve_gpu_ids(params)
        cmd = [
            sys.executable,
            str(batch_script),
            "--model_path",
            str(params.get("model_path") or "/data/models/openbmb/MiniCPM-V-4_5"),
            "--input_list",
            str(input_list_path),
            "--output_dir",
            str(batch_output_dir),
            "--gpu_ids",
            gpu_ids,
            "--dtype",
            str(params.get("dtype") or "bfloat16"),
            "--attn_impl",
            str(params.get("attn_impl") or "sdpa"),
            "--max_num_frames",
            str(int(params.get("max_num_frames") or 38)),
            "--max_slice_nums",
            str(int(params.get("max_slice_nums") or 1)),
        ]
        if limit_items is not None:
            cmd.extend(["--limit_items", str(limit_items)])

        env = os.environ.copy()
        requested_cuda_visible = str(params.get("cuda_visible_devices") or "auto").strip()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids if requested_cuda_visible.lower() == "auto" else requested_cuda_visible
        env["BOS_PIPELINE_ALLOCATED_GPUS"] = gpu_ids
        env["BOS_PIPELINE_ALLOCATED_GPU_COUNT"] = str(len([item for item in gpu_ids.split(",") if item.strip()]))
        env["HF_HOME"] = str(params.get("hf_home") or env.get("HF_HOME") or "/data/cache/huggingface")
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(
            params.get("pytorch_cuda_alloc_conf")
            or env.get("PYTORCH_CUDA_ALLOC_CONF")
            or "expandable_segments:True"
        )

        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        log_path = output_dir / "batch_video_ray.log"
        log_path.write_text(
            "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
            encoding="utf-8",
        )
        subprocess_failed = completed.returncode != 0
        subprocess_failure_type: str | None = None
        subprocess_retryable = False
        if subprocess_failed:
            subprocess_failure_type, subprocess_retryable = classify_subprocess_failure(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )

        aggregate = aggregate_outputs(
            selected_items=selected_items,
            batch_output_dir=batch_output_dir,
            caption_jsonl_path=caption_jsonl_path,
            failed_items_path=failed_items_path,
            remaining_items_path=remaining_items_path,
            subprocess_failed=subprocess_failed,
            subprocess_failure_type=subprocess_failure_type,
            subprocess_retryable=subprocess_retryable,
        )
        success_count = int(aggregate["success_count"])
        failure_count = int(aggregate["failure_count"])
        missing_output_count = int(aggregate["missing_output_count"])
        retryable_failure_count = int(aggregate["retryable_failure_count"])
        rows = list(aggregate["rows"])
        failed_rows = list(aggregate["failed_rows"])
        remaining_rows = list(aggregate["remaining_rows"])

        should_fail, failure_reason = should_fail_framework_task(
            params=params,
            total_item_count=len(selected_items),
            success_count=success_count,
            failure_count=failure_count,
            retryable_failure_count=retryable_failure_count,
            subprocess_returncode=completed.returncode,
        )
        run_id = index_run_id(runtime_input)
        if should_fail:
            if subprocess_failure_type:
                failure_type = subprocess_failure_type
            elif retryable_failure_count:
                failure_type = "retryable_item_failure"
            elif success_count <= 0:
                failure_type = "no_successful_items"
            else:
                failure_type = "item_failure_threshold_exceeded"
            batch_status = "failed_retryable" if subprocess_retryable or retryable_failure_count else "failed"
        elif failure_count:
            failure_type = "item_failures"
            batch_status = "completed_with_item_failures"
        else:
            failure_type = None
            batch_status = "completed"

        manifest = {
            "schema_version": "caption_manifest.v1",
            "task_id": runtime_input.get("task_id"),
            "run_id": run_id,
            "prepared_items_path": str(prepared_path),
            "input_list_path": str(input_list_path),
            "total_item_count": len(selected_items),
            "success_count": success_count,
            "failure_count": failure_count,
            "missing_output_count": missing_output_count,
            "retryable_failure_count": retryable_failure_count,
            "remaining_item_count": len(remaining_rows),
            "subprocess_returncode": completed.returncode,
            "subprocess_failure_type": subprocess_failure_type,
            "subprocess_retryable": subprocess_retryable,
            "batch_status": batch_status,
            "failure_type": failure_type,
            "model_path": str(params.get("model_path") or "/data/models/openbmb/MiniCPM-V-4_5"),
            "gpu_ids": gpu_ids,
            "gpu_mode": gpu_mode,
            "allocated_gpu_count": int(env["BOS_PIPELINE_ALLOCATED_GPU_COUNT"]),
            "max_num_frames": int(params.get("max_num_frames") or 38),
            "max_slice_nums": int(params.get("max_slice_nums") or 1),
        }
        write_json(manifest_path, manifest)
        if should_fail:
            partial_path.write_text("", encoding="utf-8")
        else:
            success_path.write_text("", encoding="utf-8")

        raw_output_prefix = str(params.get("output_prefix") or "test/output/minicpm_caption_v1/smoke/")
        if run_id:
            raw_output_prefix = raw_output_prefix.format(run_id=run_id)
        output_prefix = ensure_trailing_slash(raw_output_prefix)
        output_files = [
            {"local_path": str(caption_jsonl_path), "output_key": f"{output_prefix}captions.jsonl"},
            {"local_path": str(manifest_path), "output_key": f"{output_prefix}_manifest.json"},
            {"local_path": str(log_path), "output_key": f"{output_prefix}batch_video_ray.log"},
        ]
        if failed_rows:
            output_files.append({"local_path": str(failed_items_path), "output_key": f"{output_prefix}failed_items.jsonl"})
        if remaining_rows:
            output_files.append(
                {"local_path": str(remaining_items_path), "output_key": f"{output_prefix}remaining_items.jsonl"}
            )
        if should_fail:
            output_files.append({"local_path": str(partial_path), "output_key": f"{output_prefix}_PARTIAL"})
        else:
            output_files.append({"local_path": str(success_path), "output_key": f"{output_prefix}_SUCCESS"})

        artifacts = {
            "caption_jsonl_key": f"{output_prefix}captions.jsonl",
            "manifest_key": f"{output_prefix}_manifest.json",
            "log_key": f"{output_prefix}batch_video_ray.log",
            "total_item_count": len(selected_items),
            "success_count": success_count,
            "failure_count": failure_count,
            "missing_output_count": missing_output_count,
            "retryable_failure_count": retryable_failure_count,
            "remaining_item_count": len(remaining_rows),
            "batch_status": batch_status,
            "failure_type": failure_type,
            "retryable": bool(subprocess_retryable or retryable_failure_count),
            "subprocess_returncode": completed.returncode,
            "subprocess_failure_type": subprocess_failure_type,
            "gpu_ids": gpu_ids,
            "gpu_mode": gpu_mode,
            "allocated_gpu_count": int(env["BOS_PIPELINE_ALLOCATED_GPU_COUNT"]),
            "sample_item_ids": [str(row.get("item_id")) for row in rows[:5]],
        }
        if failed_rows:
            artifacts["failed_items_key"] = f"{output_prefix}failed_items.jsonl"
        if remaining_rows:
            artifacts["remaining_items_key"] = f"{output_prefix}remaining_items.jsonl"
        if should_fail:
            artifacts["partial_key"] = f"{output_prefix}_PARTIAL"
            write_runtime_output(
                runtime_output_path,
                status="failed",
                artifacts=artifacts,
                output_files=output_files,
                reason=failure_reason or "MiniCPM caption batch failed.",
            )
            return 0

        artifacts["success_key"] = f"{output_prefix}_SUCCESS"
        write_runtime_output(
            runtime_output_path,
            status="done",
            artifacts=artifacts,
            output_files=output_files,
        )
        return 0
    except Exception as exc:
        write_runtime_output(runtime_output_path, status="failed", reason=f"{type(exc).__name__}: {exc}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
