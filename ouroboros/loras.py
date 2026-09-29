"""LoRA library: what each LoRA in the managed folders does, how it's triggered, and
how it has scored here.

Sources, per LoRA file:
  - the file itself: safetensors training metadata (kohya dataset folders like
    "8_Jabstyle" name the instance token; tag frequencies);
  - rgthree's "<file>.rgthree-info.json", if ComfyUI already fetched Civitai info;
  - Civitai, looked up by the file's SHA256: title, base model, trigger words, tags,
    description, and example images with the prompts and LoRA weights used;
  - local results (lora_stats.json): how candidates rendered with it scored.

Everything is cached under cache/ so jobs never wait on the network; refresh() updates
it (Settings -> LoRAs -> Refresh).
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import statistics
import struct
import threading
import time
from pathlib import Path

import requests

CIVITAI = "https://civitai.com/api/v1"
INDEX_VERSION = 2  # bump when what's stored per LoRA changes; older records are rebuilt
GENERIC_FOLDERS = {"img", "image", "images", "train", "training", "data", "dataset", "style", "best", "base", "good"}
LORA_TAG = re.compile(r"<lora:([^:>]+):([-\d.]+)[^>]*>", re.I)


def checkpoint_base(name: str, overrides: dict | None = None) -> str:
    """Base model family of a checkpoint, from config overrides or its name."""
    for pattern, base in (overrides or {}).items():
        if pattern.lower() in name.lower():
            return base
    n = name.lower()
    if "flux" in n:
        return "Flux"
    if "illustrious" in n or "noob" in n:
        return "Illustrious"
    if "pony" in n or "autismmix" in n:
        return "Pony"
    return "SDXL"


def compatible(lora_base: str | None, ckpt_base: str) -> bool | None:
    """True/False, or None when the LoRA's base is unknown."""
    if not lora_base:
        return None
    lb = lora_base.lower()
    cb = ckpt_base.lower()
    if cb == "flux":
        return "flux" in lb
    if "flux" in lb or "sd 1" in lb or lb.startswith("sd1") or "anima" in lb:
        return False
    if cb in lb:
        return True
    # Pony and Illustrious are both SDXL-derived; generic SDXL LoRAs mostly work on both.
    return lb.startswith("sdxl")


def _safetensors_metadata(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 50_000_000:
                return {}
            return json.loads(f.read(n)).get("__metadata__", {}) or {}
    except (OSError, ValueError, struct.error):
        return {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_html(text: str, limit: int) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _thumb_url(url: str, width: int = 450) -> str:
    return re.sub(r"/(original=true|width=\d+)/", f"/width={width}/", url) if url else url


def clean_prompt(prompt: str) -> str:
    """Example prompt without <lora:...> tags (ComfyUI's text encoder doesn't read them)."""
    return re.sub(r"\s*,\s*,", ",", LORA_TAG.sub("", prompt or "")).strip(" ,")


class LoraLibrary:
    def __init__(self, cfg: dict, cache_dir: Path):
        self.cfg = cfg
        self.dirs = [Path(d) for d in cfg.get("dirs", [])]
        self.comfy_root = Path(cfg.get("comfy_root", ""))
        self.cache_dir = cache_dir
        self.index_file = cache_dir / "lora_index.json"
        self.stats_file = cache_dir / "lora_stats.json"
        self.thumbs = cache_dir / "lora_images"
        self.status = {"running": False, "done": 0, "total": 0, "message": ""}
        self._lock = threading.Lock()

    # --- index -----------------------------------------------------------------
    def index(self) -> dict[str, dict]:
        """name (as ComfyUI lists it, e.g. "Pony\\styles\\x.safetensors") -> record.

        When loras.catalog_dir points at a lora-classifier output, that is the library:
        it covers every installed LoRA and already knows what each one does, its trigger
        words and the weight range its own test renders supported. The folder scan below
        is the fallback for setups without a catalog.
        """
        catalog_dir = (self.cfg.get("catalog_dir") or "").strip()
        if catalog_dir and (Path(catalog_dir) / "state").is_dir():
            from .lora_catalog import records
            try:
                return records(Path(catalog_dir), self.cache_dir)
            except (OSError, ValueError):
                pass
        try:
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def files(self) -> list[Path]:
        """Every .safetensors under the configured dirs, however deeply nested: the
        library is sorted into folders (styles/artists, styles/anime/western, ...) and
        each of those may gain sub-folders. ComfyUI names a LoRA by its path below the
        loras root, so nesting changes the name, not whether it can be used."""
        out = []
        for d in self.dirs:
            if d.is_dir():
                out += sorted(p for p in d.rglob("*.safetensors") if p.is_file())
        return out

    def comfy_name(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.comfy_root)).replace("/", "\\")
        except ValueError:
            return path.name

    def managed(self, name: str) -> bool:
        return name in self.index()

    def refresh(self, online: bool = True, log=print) -> dict[str, dict]:
        """Rebuild the index: new or changed files are hashed and looked up."""
        with self._lock:
            old = self.index()
            files = self.files()
            self.status.update(running=True, done=0, total=len(files), message="")
            new: dict[str, dict] = {}
            try:
                for path in files:
                    name = self.comfy_name(path)
                    st = path.stat()
                    rec = old.get(name)
                    fresh = (rec and rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime
                             and rec.get("version") == INDEX_VERSION)
                    if not fresh or (online and rec.get("source") == "local" and not rec.get("civitai_checked")):
                        self.status["message"] = f"reading {path.name}"
                        rec = self._build(path, name, st, online, log)
                    new[name] = rec
                    self.status["done"] += 1
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.index_file.write_text(json.dumps(new, indent=1), encoding="utf-8")
                self.status["message"] = f"{len(new)} LoRAs indexed"
            finally:
                self.status["running"] = False
            return new

    def _build(self, path: Path, name: str, st, online: bool, log) -> dict:
        meta = _safetensors_metadata(path)
        rec = {"name": name, "version": INDEX_VERSION, "file": str(path), "size": st.st_size, "mtime": st.st_mtime,
               "title": meta.get("modelspec.title") or meta.get("ss_output_name") or path.stem,
               "base_model": None, "trigger_words": [], "training_tags": [], "tags": [],
               "description": "", "examples": [], "typical_weight": None, "civitai_url": None,
               "source": "local", "error": None}

        # Kohya metadata: dataset folders "<repeats>_<token>" name the instance token.
        try:
            dirs = json.loads(meta.get("ss_dataset_dirs") or "{}")
            freq = json.loads(meta.get("ss_tag_frequency") or "{}")
        except ValueError:
            dirs, freq = {}, {}
        tokens = [re.sub(r"^\d+_", "", d).strip() for d in dirs]
        total_imgs = sum(int(v.get("img_count", 0)) for v in dirs.values()) or None
        counts: dict[str, int] = {}
        for tags in freq.values():
            for t, c in tags.items():
                counts[t.strip()] = counts.get(t.strip(), 0) + int(c)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
        rec["training_tags"] = [f"{t} ({c})" for t, c in top]
        # A dataset folder name is a trigger only if the training captions use it too
        # ("8_Jabstyle" + tag "jabstyle"); names like "10_img" or "5_best" aren't.
        tagged = {t.lower() for t in counts}
        rec["trigger_words"] = list(dict.fromkeys(t for t in tokens if t and t.lower() in tagged
                                                  and t.lower() not in GENERIC_FOLDERS))

        info_file = path.with_name(path.name + ".rgthree-info.json")
        sha = None
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
                sha = info.get("sha256")
                civ = (info.get("raw") or {}).get("civitai") or {}
                self._apply_civitai(rec, civ, None)
                rec["source"] = "rgthree"
            except (OSError, ValueError):
                pass
        if online and rec["source"] == "local":
            try:
                sha = sha or _sha256(path)
                rec["sha256"] = sha
                r = requests.get(f"{CIVITAI}/model-versions/by-hash/{sha}", timeout=30,
                                 headers=self._auth())
                rec["civitai_checked"] = True
                if r.status_code == 404:
                    rec["error"] = "not on Civitai"
                else:
                    r.raise_for_status()
                    version = r.json()
                    model = None
                    try:
                        model = requests.get(f"{CIVITAI}/models/{version['modelId']}", timeout=30,
                                             headers=self._auth()).json()
                    except (requests.RequestException, ValueError, KeyError):
                        pass
                    self._apply_civitai(rec, version, model)
                    rec["source"] = "civitai"
            except requests.RequestException as e:
                rec["error"] = f"Civitai lookup failed: {e}"
                log(f"LoRA {path.name}: {rec['error']}")
        if online:
            self._download_thumbs(rec)
        return rec

    def _auth(self) -> dict:
        from . import keys
        key = keys.get("civitai")
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _apply_civitai(self, rec: dict, version: dict, model: dict | None) -> None:
        if not version:
            return
        rec["base_model"] = version.get("baseModel") or rec["base_model"]
        mname = (version.get("model") or {}).get("name") or (model or {}).get("name")
        vname = version.get("name") or ""
        rec["title"] = (mname if not vname or vname.lower() in mname.lower() else f"{mname} ({vname})") if mname else rec["title"]
        if version.get("modelId"):
            rec["civitai_url"] = f"https://civitai.com/models/{version['modelId']}?modelVersionId={version.get('id')}"
        words = [x.strip() for w in version.get("trainedWords") or [] for x in w.split(",") if x.strip()]
        rec["trigger_words"] = list(dict.fromkeys(words + rec["trigger_words"]))
        rec["tags"] = (model or {}).get("tags") or rec["tags"]
        rec["description"] = _strip_html((version.get("description") or "") + " "
                                         + ((model or {}).get("description") or ""), 400)
        allowed = self.cfg.get("civitai_max_nsfw_level", 32)
        stems = {rec["title"].lower(), Path(rec["name"]).stem.lower()}
        examples, weights = [], []
        for img in version.get("images") or []:
            if img.get("type", "image") != "image":
                continue
            level = img.get("nsfwLevel") or 1
            if isinstance(level, int) and level > allowed:
                continue
            meta = img.get("meta") or {}
            prompt = meta.get("prompt") or ""
            w = self._weight_in(meta, prompt, stems)
            if w is not None:
                weights.append(w)
            examples.append({"url": img.get("url"), "prompt": clean_prompt(prompt)[:600],
                             "negative": (meta.get("negativePrompt") or "")[:300], "weight": w,
                             "nsfw_level": level, "thumb": None})
        rec["examples"] = examples[:6]
        if weights:
            rec["typical_weight"] = round(statistics.median(weights), 2)

    @staticmethod
    def _weight_in(meta: dict, prompt: str, stems: set[str]) -> float | None:
        """The weight this LoRA was used at in an example image, if it can be told."""
        for res in meta.get("resources") or []:
            if (res.get("type") or "").lower() in ("lora", "locon", "lycoris") and res.get("weight") is not None:
                if len(meta.get("resources")) == 1 or any(s[:8] in (res.get("name") or "").lower() for s in stems):
                    return float(res["weight"])
        tags = LORA_TAG.findall(prompt)
        if len(tags) == 1:
            return float(tags[0][1])
        for tname, w in tags:
            if any(s[:8] in tname.lower() or tname.lower()[:8] in s for s in stems):
                return float(w)
        return None

    def _download_thumbs(self, rec: dict) -> None:
        folder = self.thumbs / re.sub(r"[^\w.-]+", "_", Path(rec["name"]).stem)
        for i, ex in enumerate(rec["examples"][:4]):
            out = folder / f"{i}.jpg"
            if out.exists():
                ex["thumb"] = str(out)
                continue
            try:
                r = requests.get(_thumb_url(ex["url"]), timeout=30)
                r.raise_for_status()
                folder.mkdir(parents=True, exist_ok=True)
                out.write_bytes(r.content)
                ex["thumb"] = str(out)
            except requests.RequestException:
                pass

    # --- local results -----------------------------------------------------------
    def stats(self) -> dict[str, dict]:
        try:
            return json.loads(self.stats_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def record_results(self, results: list[tuple[tuple, dict, float]]) -> None:
        """results: (lora set as ((name, strength), ...), criteria scores, total score)
        for each judged candidate."""
        if not results:
            return
        with self._lock:
            stats = self.stats()
            for loras, criteria, total in results:
                for name, strength in loras:
                    s = stats.setdefault(name, {"n": 0, "total": 0.0, "style": 0.0, "strengths": []})
                    s["n"] += 1
                    s["total"] += total
                    s["style"] += criteria.get("style", 0)
                    s["strengths"] = (s["strengths"] + [round(strength, 2)])[-30:]
                    s["updated"] = time.strftime("%Y-%m-%d")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.stats_file.write_text(json.dumps(stats, indent=1), encoding="utf-8")

    def stats_line(self, name: str) -> str:
        s = self.stats().get(name)
        if not s or not s["n"]:
            return "not tested here yet"
        st = sorted(s["strengths"])
        rng = f"{st[0]:g}-{st[-1]:g}" if st and st[0] != st[-1] else (f"{st[0]:g}" if st else "?")
        return (f"tested here in {s['n']} candidates: avg score {s['total'] / s['n']:.0f}, "
                f"avg style {s['style'] / s['n']:.1f}/10, strengths {rng}")

    # --- helpers for prompts -------------------------------------------------------
    def card(self, rec: dict, detail: bool = True) -> str:
        parts = [f"{Path(rec['name']).stem}: {rec['title']}"]
        if rec.get("base_model"):
            parts.append(f"base {rec['base_model']}")
        parts.append("triggers: " + (", ".join(rec["trigger_words"][:6]) or "none"))
        if rec.get("typical_weight") is not None:
            parts.append(f"usual weight {rec['typical_weight']:g}")
        if detail and rec.get("tags"):
            parts.append("tags: " + ", ".join(rec["tags"][:8]))
        if detail and rec.get("description"):
            parts.append("about: " + rec["description"][:160])
        parts.append(self.stats_line(rec["name"]))
        return " | ".join(parts)

    def triggers_of(self, name: str) -> list[str]:
        rec = self.index().get(name)
        return list(rec["trigger_words"]) if rec else []


def with_triggers(positive: str, active: tuple, library: LoraLibrary | None, split_tags, norm_tag,
                  protected: set[str] | None = None) -> str:
    """The positive prompt as rendered:
    - the main trigger word (the first listed) of each active managed LoRA is added
      (only the main one: some LoRAs list many, e.g. every character they know);
    - trigger words of managed LoRAs that are off are removed, since they mean nothing
      without their LoRA, unless they're in `protected` (the user's own prompt) or
      shared with an active LoRA;
    - LoRA file names and titles are removed: they aren't tags (a prompt writer copied
      "OldAnimeStyle_XL_v3" into a prompt once)."""
    if not library:
        return positive
    index = library.index()
    active_names = [n for n, _ in active]
    want = [index[n]["trigger_words"][0] for n in active_names if index.get(n, {}).get("trigger_words")]
    keep = {norm_tag(w) for w in want} | {norm_tag(w) for n in active_names
                                          for w in index.get(n, {}).get("trigger_words", [])}
    drop = {norm_tag(w) for n, rec in index.items() if n not in active_names
            for w in rec.get("trigger_words", [])} - keep - (protected or set())
    names = {norm_tag(Path(n).stem) for n in index} | {norm_tag(rec.get("title", "")) for rec in index.values()}
    drop |= {x for x in names if x} - (protected or set())
    # An active trigger written loosely ("1990s (style)") is put back exactly as listed
    # ("1990s \(style\)"): unescaped brackets are weight syntax to ComfyUI.
    exact = {norm_tag(w): w for n in active_names for w in index.get(n, {}).get("trigger_words", [])}
    lines = []
    for line in positive.splitlines():
        tags = [t.strip() for t in line.split(",") if t.strip()]
        kept = [exact.get(norm_tag(t), t) if t.lower() == norm_tag(t) else t
                for t in tags if norm_tag(t) not in drop]
        if kept or not tags:
            lines.append(", ".join(kept))
    out = "\n".join(lines).strip()
    have = {norm_tag(t) for t in split_tags(out)}
    missing = [w for w in dict.fromkeys(want) if norm_tag(w) not in have]
    return (", ".join(missing) + ", " + out) if missing else out
