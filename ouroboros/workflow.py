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


# Added to every render's negative prompt. Many LoRAs were trained on booru-tagged data
# that includes images of minors (the LoRA classifier finds such tags in a large share of
# a typical library), and LoRAs are now chosen by a model; this keeps every render adult.
# The same list the LoRA classifier uses for its own test renders.
SAFETY_NEGATIVE = ["child", "loli", "shota", "toddler", "kid"]


def with_safety_negative(negative: str) -> str:
    from .params import norm_tag, split_tags
    have = {norm_tag(t) for t in split_tags(negative or "")}
    missing = [t for t in SAFETY_NEGATIVE if t not in have]
    if not missing:
        return negative
    return (negative.rstrip().rstrip(",") + ", " if (negative or "").strip() else "") + ", ".join(missing)


def _lora_root() -> Path | None:
    """The classified library's loras folder (loras.catalog_dir/loras), if there is one."""
    try:
        from .runner import load_config
        d = (load_config().get("loras") or {}).get("catalog_dir")
        root = Path(d).expanduser() / "loras" if d else None
        return root if root and root.is_dir() else None
    except Exception:
        return None


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
              control: dict | list | None = None, ipadapter: dict | list | None = None,
              pick: int | None = None) -> dict:
        """pick: render only image `pick` (0-based) of a batch of batch_size, exactly as the
        batch would have drawn it: ComfyUI gives each image of a batch its own slice of the
        seed's noise, so image 8 of a batch of 8 is seed + position 8, not the seed alone
        (LatentFromBatch keeps its position, and the sampler draws that slice)."""
        use_reference, masked, _source = MODES[p.mode]
        if pick is not None:
            batch_size = max(int(batch_size), int(pick) + 1)
        graph = copy.deepcopy(self.graph)
        values = {
            "positive": p.positive if positive is None else positive,
            "negative": with_safety_negative(p.negative),
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
        root = _lora_root()
        for node in graph.values():
            for k, v in node.get("inputs", {}).items():
                if isinstance(v, dict) and "lora" in v:
                    if root is not None and v["lora"]:  # where the file is now (see lora_catalog.current_name)
                        from .lora_catalog import current_name
                        v["lora"] = current_name(v["lora"], root)
                    v["lora"] = _local_sep(v["lora"])
                else:
                    node["inputs"][k] = _local_sep(v)
        # One ControlNet (a dict) or several (a list: e.g. a pose skeleton and the edges that
        # keep a character's shapes), chained in order.
        if pick is not None:
            ks = graph[self.spec["roles"]["seed"][0]]["inputs"]
            graph["pick"] = {"class_type": "LatentFromBatch",
                             "inputs": {"samples": ks["latent_image"], "batch_index": int(pick), "length": 1}}
            ks["latent_image"] = ["pick", 0]
        for n, c in enumerate(control if isinstance(control, list) else [control] if control else []):
            self._add_controlnet(graph, c, masked, n)
        # One IP-Adapter (a dict) or several (a list: e.g. the subject's, then the style's).
        for n, ip in enumerate(ipadapter if isinstance(ipadapter, list) else [ipadapter] if ipadapter else []):
            self._add_ipadapter(graph, ip, n)
        return graph

    def _add_controlnet(self, graph: dict, c: dict, masked: bool, n: int = 0) -> None:
        """Insert a ControlNet between the prompts and the sampler:
        LoadImage(control image) -> [Inpaint Crop, same mask] -> ControlNetApplyAdvanced
        (Union ControlNet, type set by SetUnionControlNetType) -> KSampler positive/negative.

        When the workflow repaints a masked region, its Inpaint Crop node cuts that region
        out and samples it at 1024x1024, so the control image goes through a copy of
        the same crop (same mask, same settings): the skeleton lands on the same pixels.
        c: {"model", "image" (ComfyUI name), "type", "strength", "start", "end", and optionally
        "preprocess": "canny" to draw the image's edges first (ComfyUI's own Canny node), with
        "low" / "high" thresholds}. Several are chained (n = 0, 1, ...); a later one with the
        same model reuses the first one's loaded ControlNet."""
        sampler = self.spec["roles"]["seed"][0]
        ks = graph[sampler]["inputs"]
        ids = iter(f"cn{n * 10 + i}" for i in range(1, 10))
        load, loader, kind, apply, pre = next(ids), next(ids), next(ids), next(ids), next(ids)
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": c["image"]}}
        first = "cn2"
        if n and graph.get(first, {}).get("inputs", {}).get("control_net_name") == c["model"]:
            loader = first
        else:
            graph[loader] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": c["model"]}}
        net = [loader, 0]
        if c.get("type"):
            graph[kind] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": net, "type": c["type"]}}
            net = [kind, 0]
        image = [load, 0]
        if c.get("preprocess") == "canny":
            graph[pre] = {"class_type": "Canny", "inputs": {"image": image, "low_threshold": float(c.get("low", 0.2)),
                                                           "high_threshold": float(c.get("high", 0.5))}}
            image = [pre, 0]
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


    def _add_ipadapter(self, graph: dict, ip: dict, n: int = 0) -> None:
        """Insert an IP-Adapter between the model and the sampler:
        LoadImage(style/character reference) -> IPAdapterUnifiedLoader -> IPAdapterAdvanced
        -> KSampler.model.

        A ControlNet steers the *shape* of an image; this steers what the subject looks
        like, by conditioning the model on a reference image rather than on words. That is
        the part a prompt cannot carry: "the same character" survives a pose change here,
        where a text description of a face does not.

        ip: {"image" (ComfyUI name), "preset", "weight", "weight_type", "start", "end"}.
        Needs ComfyUI_IPAdapter_plus and its models installed.

        Several are chained (n = 0, 1, ...): each applies to the model the previous one
        produced, and a later one with the same preset reuses the first one's loaded
        IP-Adapter and CLIP vision models instead of loading them again.
        """
        sampler = self.spec["roles"]["seed"][0]
        ks = graph[sampler]["inputs"]
        load, loader, apply = f"ip{3 * n + 1}", f"ip{3 * n + 2}", f"ip{3 * n + 3}"
        preset = ip.get("preset", "PLUS (high strength)")
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": ip["image"]}}
        graph[loader] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ks["model"], "preset": preset}}
        first = "ip2"
        if n and graph.get(first, {}).get("inputs", {}).get("preset") == preset:
            graph[loader]["inputs"]["ipadapter"] = [first, 1]
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
