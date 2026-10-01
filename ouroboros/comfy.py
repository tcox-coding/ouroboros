"""Minimal ComfyUI HTTP client: upload, queue, wait, download."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import requests


class ComfyError(RuntimeError):
    pass


class Cancelled(Exception):
    """A queued task was removed while it was running."""


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188", timeout: float = 600):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def choices(self, node_type: str, input_name: str) -> list[str]:
        """Allowed values of a combo input, e.g. KSampler sampler_name."""
        info = requests.get(f"{self.url}/object_info/{node_type}", timeout=30).json()
        return list(info[node_type]["input"]["required"][input_name][0])

    def upload_image(self, path: Path, subfolder: str = "ouroboros") -> str:
        """Upload into ComfyUI's input folder; returns the name LoadImage expects. Each
        run uploads into its own subfolder: jobs running side by side would otherwise
        overwrite each other's reference_input.png."""
        with open(path, "rb") as f:
            r = requests.post(
                f"{self.url}/upload/image",
                files={"image": (path.name, f, "image/png")},
                data={"overwrite": "true", "subfolder": subfolder},
                timeout=60,
            )
        r.raise_for_status()
        d = r.json()
        return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]

    def queue(self, graph: dict) -> str:
        r = requests.post(f"{self.url}/prompt", json={"prompt": graph, "client_id": self.client_id}, timeout=30)
        if r.status_code != 200:
            raise ComfyError(f"ComfyUI rejected the workflow: {r.text[:2000]}")
        return r.json()["prompt_id"]

    def wait(self, prompt_ids: list[str], poll: float = 1.0, should_stop=None) -> dict[str, dict]:
        """Block until every prompt finishes; returns prompt_id -> history entry.
        should_stop() returning True cancels the unfinished prompts and raises Cancelled."""
        done: dict[str, dict] = {}
        deadline = time.monotonic() + self.timeout * max(1, len(prompt_ids))
        while len(done) < len(prompt_ids):
            if should_stop and should_stop():
                for pid in prompt_ids:
                    if pid not in done:
                        self.cancel(pid)
                raise Cancelled("cancelled")
            if time.monotonic() > deadline:
                raise ComfyError("Timed out waiting for ComfyUI")
            for pid in prompt_ids:
                if pid in done:
                    continue
                h = requests.get(f"{self.url}/history/{pid}", timeout=30).json().get(pid)
                if not h:
                    continue
                status = h.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError(f"Prompt {pid} failed: {status.get('messages')}")
                if status.get("completed"):
                    done[pid] = h
            time.sleep(poll)
        return done

    def cancel(self, prompt_id: str) -> None:
        """Take a prompt out of ComfyUI's queue, or stop it if it's the one rendering."""
        try:
            q = requests.get(f"{self.url}/queue", timeout=10).json()
            if any(len(item) > 1 and item[1] == prompt_id for item in q.get("queue_running", [])):
                requests.post(f"{self.url}/interrupt", json={"prompt_id": prompt_id}, timeout=10)
            else:
                requests.post(f"{self.url}/queue", json={"delete": [prompt_id]}, timeout=10)
        except requests.RequestException:
            pass

    def fetch_images(self, history: dict, output_node: str) -> list[bytes]:
        images = history["outputs"].get(output_node, {}).get("images", [])
        out = []
        for img in images:
            r = requests.get(f"{self.url}/view", params=img, timeout=60)
            r.raise_for_status()
            out.append(r.content)
        return out
