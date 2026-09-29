"""Patch an API-format ComfyUI workflow by role, so the LLM never edits graph JSON.

workflows/nodes.json names the workflow file and maps roles to [node_id, input_name]
(without one, nodes.example.json and the example workflow it names are used):

    {
      "file": "My Workflow.json",
      "output": "21",               node whose images are the result
      "remove_nodes": ["17"],       e.g. a Save node, so candidates don't land in your output folder
      "roles": {"positive": ["8", "text"], "seed": ["10", "seed"], "masked": ["51", "value"], ...}
    }

Inputs without a role (width, height, ...) keep the values saved in the workflow.
"lora_loader" names a Power Lora Loader (rgthree) node; its lora_N entries are rewritten
from each candidate's LoRA set (the LoRAs switched on in the saved workflow are the
default set).
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from .params import MODES, GenParams

# Model names inside a saved workflow or the LoRA catalog use the separator of the OS they
# were made on ("Pony\\styles\\x.safetensors"); ComfyUI only knows its own OS's form.
_MODEL_FILE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf)$", re.I)
_EMBEDDING = re.compile(r"embedding:[^\s,()]+")


def _local_sep(value):
    if os.sep == "\\" or not isinstance(value, str) or "\\" not in value:
        return value
    if _MODEL_FILE.search(value) and "\n" not in value:
        return value.replace("\\", "/")
    return _EMBEDDING.sub(lambda m: m.group(0).replace("\\", "/"), value)


ROLES = {"positive", "negative", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
         "image", "use_reference", "masked", "batch_size", "checkpoint", "width", "height"}


def spec_path(folder: Path) -> Path | None:
    """Your nodes.json, or the shipped example so a fresh clone runs as it is."""
    return next((f for f in (folder / "nodes.json", folder / "nodes.example.json") if f.exists()), None)


class Workflows:
    def __init__(self, folder: Path):
        spec_file = spec_path(folder)
        if spec_file is None:
            raise FileNotFoundError("workflows/nodes.json is missing. Export your ComfyUI workflow with "
                                    "'Export (API)' and map its nodes (see workflows/nodes.example.json).")
        self.spec = json.loads(spec_file.read_text(encoding="utf-8"))
        self.graph = json.loads((folder / self.spec["file"]).read_text(encoding="utf-8"))
        if "nodes" in self.graph:
            raise ValueError(f"{self.spec['file']} is in UI format; re-export it with 'Export (API)'.")
        for role, (node_id, input_name) in self.spec["roles"].items():
            if role not in ROLES:
                raise KeyError(f"Unknown role '{role}' in nodes.json")
            if node_id not in self.graph or input_name not in self.graph[node_id]["inputs"]:
                raise KeyError(f"nodes.json role '{role}': {node_id}.{input_name} not in {self.spec['file']}")
        for node_id in self.spec.get("remove_nodes", []):
            self.graph.pop(node_id, None)

    @property
    def output_node(self) -> str:
        return self.spec["output"]

    @property
    def lora_node(self) -> str | None:
        node = self.spec.get("lora_loader")
        return node if node in self.graph else None

    def default_loras(self) -> tuple:
        """((name, strength), ...) switched on in the saved workflow."""
        if not self.lora_node:
            return ()
        inputs = self.graph[self.lora_node]["inputs"]
        return tuple((v["lora"], round(float(v.get("strength", 1.0)), 2))
                     for k, v in sorted(inputs.items(), key=lambda kv: _slot(kv[0]))
                     if k.startswith("lora_") and isinstance(v, dict) and v.get("on") and v.get("lora"))

    def fixed_inputs(self) -> dict:
        """Values that stay as saved in the workflow (for reproducing a result)."""
        out = {}
        for node_id, node in self.graph.items():
            title = node.get("_meta", {}).get("title", "")
            if node.get("class_type", "").startswith("Primitive") and "value" in node["inputs"]:
                out[title or node_id] = node["inputs"]["value"]
        return out

    def default(self, role: str):
        """The value saved in the workflow for a role, e.g. its negative prompt."""
        node_id, input_name = self.spec["roles"][role]
        return self.graph[node_id]["inputs"][input_name]

    def workflow_size(self) -> tuple[int, int] | None:
        """(width, height) saved in the workflow, if it has width/height roles."""
        roles = self.spec["roles"]
        if "width" in roles and "height" in roles:
            return int(self.default("width")), int(self.default("height"))
        return None

    def build(self, p: GenParams, image_name: str, batch_size: int = 1, checkpoint: str | None = None,
              positive: str | None = None, size: tuple[int, int] | None = None,
              control: dict | None = None, ipadapter: dict | None = None) -> dict:
        use_reference, masked, _source = MODES[p.mode]
        graph = copy.deepcopy(self.graph)
        values = {
            "positive": p.positive if positive is None else positive,
            "negative": p.negative,
            "seed": p.seed,
            "steps": p.steps,
            "cfg": p.cfg,
            "sampler_name": p.sampler_name,
            "scheduler": p.scheduler,
            "denoise": p.denoise,
            "image": image_name,
            "use_reference": use_reference,
            "masked": masked,
            "batch_size": batch_size,
            "checkpoint": checkpoint,
            "width": size[0] if size else None,
            "height": size[1] if size else None,
        }
        for role, (node_id, input_name) in self.spec["roles"].items():
            if values[role] is not None:
                graph[node_id]["inputs"][input_name] = values[role]
        if self.lora_node:
            inputs = graph[self.lora_node]["inputs"]
            for k in [k for k in inputs if k.startswith("lora_")]:
                del inputs[k]
            for i, (name, strength) in enumerate(p.loras, 1):
                inputs[f"lora_{i}"] = {"on": True, "lora": name, "strength": strength}
        for node in graph.values():
            for k, v in node.get("inputs", {}).items():
                if isinstance(v, dict) and "lora" in v:
                    v["lora"] = _local_sep(v["lora"])
                else:
                    node["inputs"][k] = _local_sep(v)
        if control:
            self._add_controlnet(graph, control, masked)
        if ipadapter:
            self._add_ipadapter(graph, ipadapter)
        return graph

    def _add_controlnet(self, graph: dict, c: dict, masked: bool) -> None:
        """Insert a ControlNet between the prompts and the sampler:
        LoadImage(control image) -> [Inpaint Crop, same mask] -> ControlNetApplyAdvanced
        (Union ControlNet, type set by SetUnionControlNetType) -> KSampler positive/negative.

        When the workflow repaints a masked region, its Inpaint Crop node cuts that region
        out and samples it at 1024x1024, so the control image goes through a copy of
        the same crop (same mask, same settings): the skeleton lands on the same pixels.
        c: {"model", "image" (ComfyUI name), "type", "strength", "start", "end"}."""
        sampler = self.spec["roles"]["seed"][0]
        ks = graph[sampler]["inputs"]
        ids = iter(f"cn{i}" for i in range(1, 100))
        load, loader, kind, apply = next(ids), next(ids), next(ids), next(ids)
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": c["image"]}}
        graph[loader] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": c["model"]}}
        net = [loader, 0]
        if c.get("type"):
            graph[kind] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": net, "type": c["type"]}}
            net = [kind, 0]
        image = [load, 0]
        crop = next((k for k, n in graph.items() if n.get("class_type") == "InpaintCropImproved"), None)
        if masked and crop:
            copy_id = next(ids)
            graph[copy_id] = copy.deepcopy(graph[crop])
            graph[copy_id]["inputs"]["image"] = image
            image = [copy_id, 1]  # cropped_image
        graph[apply] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ks["positive"], "negative": ks["negative"], "control_net": net, "image": image,
            "strength": float(c.get("strength", 0.7)), "start_percent": float(c.get("start", 0.0)),
            "end_percent": float(c.get("end", 0.8))}}
        ks["positive"], ks["negative"] = [apply, 0], [apply, 1]


    def _add_ipadapter(self, graph: dict, ip: dict) -> None:
        """Insert an IP-Adapter between the model and the sampler:
        LoadImage(style/character reference) -> IPAdapterUnifiedLoader -> IPAdapterAdvanced
        -> KSampler.model.

        A ControlNet steers the *shape* of an image; this steers what the subject looks
        like, by conditioning the model on a reference image rather than on words. That is
        the part a prompt cannot carry: "the same character" survives a pose change here,
        where a text description of a face does not.

        ip: {"image" (ComfyUI name), "preset", "weight", "weight_type", "start", "end"}.
        Needs ComfyUI_IPAdapter_plus and its models installed.
        """
        sampler = self.spec["roles"]["seed"][0]
        ks = graph[sampler]["inputs"]
        ids = iter(f"ip{i}" for i in range(1, 100))
        load, loader, apply = next(ids), next(ids), next(ids)
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": ip["image"]}}
        graph[loader] = {"class_type": "IPAdapterUnifiedLoader",
                         "inputs": {"model": ks["model"], "preset": ip.get("preset", "PLUS (high strength)")}}
        graph[apply] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": [loader, 0], "ipadapter": [loader, 1], "image": [load, 0],
            "weight": float(ip.get("weight", 0.8)), "weight_type": ip.get("weight_type", "linear"),
            "combine_embeds": "concat", "start_at": float(ip.get("start", 0.0)),
            "end_at": float(ip.get("end", 1.0)), "embeds_scaling": "V only"}}
        ks["model"] = [apply, 0]


def _slot(key: str) -> int:
    try:
        return int(key.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0
