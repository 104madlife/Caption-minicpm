"""Pipeline adapter for running MiniCPM caption from a prepared JSONL index."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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
) -> tuple[int, int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    for item in selected_items:
        item_id = str(item.get("item_id") or "")
        safe_item_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in item_id).strip("._") or "item"
        output_path = batch_output_dir / f"{safe_item_id}.json"
        if not output_path.is_file():
            rows.append({**item, "status": "failed", "error": f"Missing batch output: {output_path}"})
            failure_count += 1
            continue

        data = json.loads(output_path.read_text(encoding="utf-8"))
        if data.get("error"):
            rows.append({**item, "status": "failed", "error": data["error"]})
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
    caption_jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return success_count, failure_count, rows


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
    manifest_path = output_dir / "_manifest.json"
    success_path = output_dir / "_SUCCESS"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        items, prepared_path = read_prepared_items(runtime_input, work_dir)
        limit_items = int(params["limit_items"]) if params.get("limit_items") is not None else None
        selected_items = write_input_list(input_list_path, items, limit_items=limit_items)

        repo_root = Path(__file__).resolve().parents[1]
        batch_script = repo_root / "batch_video_ray.py"
        gpu_ids = str(params.get("gpu_ids") or "0,1")
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
        env["CUDA_VISIBLE_DEVICES"] = str(params.get("cuda_visible_devices") or gpu_ids)
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
        if completed.returncode != 0:
            raise RuntimeError(f"batch_video_ray.py failed with exit code {completed.returncode}; log={log_path}")

        success_count, failure_count, rows = aggregate_outputs(
            selected_items=selected_items,
            batch_output_dir=batch_output_dir,
            caption_jsonl_path=caption_jsonl_path,
        )
        run_id = index_run_id(runtime_input)
        manifest = {
            "schema_version": "caption_manifest.v1",
            "task_id": runtime_input.get("task_id"),
            "run_id": run_id,
            "prepared_items_path": str(prepared_path),
            "input_list_path": str(input_list_path),
            "total_item_count": len(selected_items),
            "success_count": success_count,
            "failure_count": failure_count,
            "model_path": str(params.get("model_path") or "/data/models/openbmb/MiniCPM-V-4_5"),
            "gpu_ids": gpu_ids,
            "max_num_frames": int(params.get("max_num_frames") or 38),
            "max_slice_nums": int(params.get("max_slice_nums") or 1),
        }
        write_json(manifest_path, manifest)
        success_path.write_text("", encoding="utf-8")

        raw_output_prefix = str(params.get("output_prefix") or "test/output/minicpm_caption_v1/smoke/")
        if run_id:
            raw_output_prefix = raw_output_prefix.format(run_id=run_id)
        output_prefix = ensure_trailing_slash(raw_output_prefix)
        output_files = [
            {"local_path": str(caption_jsonl_path), "output_key": f"{output_prefix}captions.jsonl"},
            {"local_path": str(manifest_path), "output_key": f"{output_prefix}_manifest.json"},
            {"local_path": str(success_path), "output_key": f"{output_prefix}_SUCCESS"},
            {"local_path": str(log_path), "output_key": f"{output_prefix}batch_video_ray.log"},
        ]
        artifacts = {
            "caption_jsonl_key": f"{output_prefix}captions.jsonl",
            "manifest_key": f"{output_prefix}_manifest.json",
            "success_key": f"{output_prefix}_SUCCESS",
            "log_key": f"{output_prefix}batch_video_ray.log",
            "total_item_count": len(selected_items),
            "success_count": success_count,
            "failure_count": failure_count,
            "sample_item_ids": [str(row.get("item_id")) for row in rows[:5]],
        }
        if failure_count:
            write_runtime_output(
                runtime_output_path,
                status="failed",
                artifacts=artifacts,
                output_files=output_files,
                reason=f"MiniCPM caption failed for {failure_count} item(s).",
            )
            return 0

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
