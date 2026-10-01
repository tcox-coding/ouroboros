# Ouroboros: how it works

Implementation notes and design decisions: how a round runs, how the judge is kept cheap and
consistent, and how LoRAs, poses, the IP-Adapter and the hand refiner fit in. To install and
run Ouroboros, see the [README](README.md).

> Status: runs end to end against your ComfyUI workflow. Tuned with
> `qwen3-vl:8b-instruct` in Ollama as the judge (its abliterated version,
> `huihui_ai/qwen3-vl-abliterated:8b-instruct`), which is still one save away in
> Settings if you'd rather not send images to a hosted model.

---

## 1. The core idea

The manual loop this replaces has three roles:

| Role | By hand | Automated |
|---|---|---|
| **Generator** | ComfyUI + SDXL | same, driven through ComfyUI's HTTP API |
| **Judge** ("how close is it?") | you + ChatGPT | vision LLM (DeepInfra, Ollama or OpenAI) scoring against a fixed rubric; local similarity model as a pre-filter |
| **Planner** ("what should change?") | ChatGPT | the same LLM call, returning structured JSON edits |

The loop:

```
job ─► initial params ─► ComfyUI batch (N candidates, local, cheap)
                               │
                   local similarity pre-filter (free)
                               │ top-k
                               ▼
            ONE judge call: score top-k vs reference + propose next edit
                               │
          best ≥ threshold? ─yes─► save, next job
                               │ no
             plateau / budget hit? ─yes─► save best, flag, next job
                               │ no
                  apply edit ─► next round
```

## 2. How rounds are kept low (time and cost)

The expensive, slow part is the OpenAI round trip, not local GPU time. So the design
**spends GPU to save API calls**:

1. **Batch per round, judge once.** Each round renders several candidates (e.g. 4 seeds,
   or seed × cfg). They all go to one OpenAI call as a contact sheet, or as the top-k
   images. One call scores every candidate and plans the next move. That's roughly
   4× more exploration per dollar than 1 image per call.
2. **Free local pre-filter.** CLIP or DINOv2 similarity to the reference ranks the batch
   locally. Only the top-k go to OpenAI, and obviously bad rounds can skip the API
   entirely and re-seed.
3. **Structured output with a small action space.** The model returns JSON that must
   match a schema: rubric scores per candidate, plus one edit (`prompt_add`,
   `prompt_remove`, `cfg`, `denoise`, `mode`, `mask_target`, …) with bounded ranges.
   No free-form chat, no parsing failures, no retries. Values are clamped in code.
4. **Rubric scoring, not a single "vibe" number.** The model scores fixed criteria
   (identity/character, pose/composition, style, colour, details/artifacts), each
   with written anchors. The weighted total is the likeness score. Per-criterion scores
   tell the planner *what* to fix, which cuts wasted rounds. Anchored rubrics are also
   far less noisy between calls, so the threshold means something.
5. **Coarse-to-fine phases.** Fix the big things first, then freeze them:
   - *Explore:* txt2img or img2img, prompt edits, seed sweep, fewer steps.
   - *Refine:* lock the best seed, tune cfg / denoise / sampler, full steps.
   - *Repair:* inpaint a named region ("hands", "collar"). The mask is made **locally**
     by CLIPSeg/SAM3 from a text target, because GPT can't draw pixel masks reliably
     but can name what's wrong.

   Parameters frozen in an earlier phase leave the action space, so the search narrows.
6. **Stop early and stop cheaply.** Stop when any of these is true:
   - the score is ≥ threshold;
   - no improvement greater than ε over the last K rounds (plateau);
   - the max-rounds or max-$ budget is spent.

   In the last two cases the job keeps its best image and gets flagged for you.
7. **Cheap, cache-friendly requests.** The system prompt, rubric, and reference image
   form a fixed prefix, so OpenAI's prompt caching discounts them on every round after
   the first. History is sent as a compact text log (params → scores), not old
   images. Candidate images are downscaled (512 px, `detail: low` where it's enough).
8. **Learn across jobs.** Every round's (params, scores) goes to `runs/*.jsonl`.
   Winning settings seed the next job's starting point (e.g. your usual sampler,
   scheduler, and cfg band), so later jobs start closer to done.

A rough cost model: a job that needs ~6 rounds × 1 call is 6 API calls. The naive
version (one image per call, separate judge and planner calls) is ~20–40 calls.

## 3. Components

```
ouroboros/
  ouroboros/
    comfy.py        ComfyUI HTTP client: upload image, queue prompt, wait, fetch outputs
    workflow.py     Loads API-format workflow templates, patches params by role
    scoring.py      Optional local CLIP/DINOv2 similarity pre-filter
    judge.py        One LLM call per round: rubric scores + next edit (JSON schema);
                    fresh-eyes confirmation; summaries of older rounds
    prompter.py     Writes the starting prompt from a description (or merges it into yours)
    backends.py     Judge backends: DeepInfraBackend and OpenAIBackend (hosted), OllamaBackend (LAN)
    generate.py     One-off generation for the Home tab: the loop's renderer, driven by hand;
                    the generation queue, its time estimate and History reruns
    autofix.py      Auto-fix: inspect an image for flaws, repaint them, keep what a review approves
    keys.py         API keys for DeepInfra, OpenAI and Civitai (Settings -> API keys)
    thumbs.py       Cached JPEG thumbnails for History and the LoRA example renders
    lora_catalog.py The classified LoRA library (lora-classifier output) as cards for the picker
    params.py       GenParams dataclass, bounds, clamping, applying edits, tag helpers
    loop.py         One job's round loop, phases, confirmation, stop rules, judge memory
    jobs.py         Job queue (folders in jobs/pending → done/failed)
    runner.py       Runs the queue on a background thread; live state for the UI; config load/save
    server.py       Local web server (standard library only): JSON API + images
    __main__.py     `python -m ouroboros` (web UI) or `... run` (headless)
    masks.py        mask_target text -> CLIPSeg mask -> image with that region transparent
    comfy_launcher.py  Starts ComfyUI hidden in the background (Comfy Desktop's install)
    pose.py         DWPose skeletons (rtmlib), drawn for the openpose ControlNet; the pose library
    handfix.py      Hand refiner: repaint each hand along a fitted five-finger skeleton
    llm_queue.py    One first-come-first-served queue for every LLM call (several jobs at once)
    judge_eval.py   Compare judge models on quality and speed (python -m ouroboros eval ...)
    sizes.py        Output size from the reference's shape (SDXL ~1 MP sizes)
    loras.py        LoRA library: file metadata + Civitai (trigger words, examples, weights) + local results
    lora_picker.py  Picks a job's style LoRAs (text shortlist, then example images vs the reference)
    model_presets.json  Recommended judge settings per vision model (filled in when you pick one)
  static/index.html The browser UI
  workflows/        Your ComfyUI workflow exported via "Export (API)" + nodes.json role map
  jobs/pending/     One folder per job: reference.png + job.json
  runs/             Per-job output: every round's images, graphs/, log.jsonl, run.json, summary.json
  runs/manual/      Home tab generations (one folder per generation, run.json + images)
  cache/            LoRA index and card images, thumbnails, History reruns, lora_stats.json
  api_keys.json     Your API keys (not committed)
  config.example.json
```

### ComfyUI side
The ComfyUI API is: `POST /upload/image`, `POST /prompt` with the API-format graph,
wait on `/ws` (or poll `GET /history/{id}`), then `GET /view` for the outputs. The code
never lets the LLM edit graph JSON. There's one workflow with switch nodes. The planner
picks a **mode**, and `workflow.py` sets the switches and inputs through the role →
node-id map in `workflows/nodes.json`:

| mode | Use Reference Image? | Masked? | image loaded |
|---|---|---|---|
| `txt2img` | off | off | reference (unused) |
| `img2img_reference` | on | off | reference |
| `img2img_best` | on | off | best image so far |
| `inpaint_reference` / `inpaint_best` | on | on | that image, with the `mask_target` region made transparent |

The workflow reads its inpaint mask from the LoadImage alpha channel, just like a mask
painted in ComfyUI's editor. So `masks.py` runs WAS **CLIPSeg Masking** on the image
for the planner's text ("left hand"), thresholds and grows the mask, and uploads a copy
with that region transparent. If CLIPSeg finds nothing, the round falls back to img2img.

**New image, not a copy.** The goal is a new image of the reference's character and
style, so two settings (Settings tab, or `loop` in config) stop the loop from winning by
copying the reference:

- `min_reference_denoise` (default 0.7): img2img from the reference never runs below
  this denoise.
- `allow_inpaint_reference` (default off): inpainting the reference keeps everything
  outside the mask identical to it, so the mode is removed from the planner's choices.
  `inpaint_best` still repairs the loop's own results.

The job's reference image (uploaded in the Queue tab, or the one in a prompt-generator
folder) always replaces the image saved in the workflow's Load Image node.

Width, height, LoRAs and everything else without a role keep the values saved in the
workflow. Batch size is forced to 1 and candidates are queued as separate prompts, so
each candidate can be reproduced from its seed. The Save node is removed
(`remove_nodes`), so candidates don't fill your output folder; results come from the
Preview node and are saved under `runs/`.

### ComfyUI without the desktop app
Ouroboros starts ComfyUI itself, hidden: no window, no browser tab. It starts when
the UI opens, so it's warm by the time you press Start, and again whenever the queue
starts if it isn't running. It uses the ComfyUI that Comfy Desktop installed:
- the install's own Python environment (`ComfyUI/.venv`), custom nodes and ComfyUI Manager;
- the model folders (`instance-model-paths\<id>.yaml`) and the shared input/output folders;
- its launch arguments, all read from Comfy Desktop's `installations.json` and
  `settings.json`.

It adds `--disable-auto-launch` and listens on the URL in Settings. Its output goes to
`logs/comfyui.log` (Settings -> ComfyUI -> *Launch command and log*).

- If something already answers on that URL (e.g. you opened Comfy Desktop), that one is
  used and nothing is started.
- It stops when Ouroboros exits (`comfyui.stop_on_exit`), including when the console
  window is closed: on Windows it's in a job object tied to Ouroboros's process, on
  Linux it gets a parent-death signal (`PR_SET_PDEATHSIG`).
- Comfy Desktop's files are read from `%APPDATA%\Comfy Desktop` on Windows, and from
  `~/.config/comfyui-desktop-2` and `~/.local/share/comfyui-desktop-2` on Linux.
- Settings -> ComfyUI has Start/Stop buttons and the on/off switch (`comfyui.autostart`).
- To use a different install, set `comfyui.python`, `comfyui.main` and `comfyui.args`
  in `config.json`.
- Avoid running Comfy Desktop's ComfyUI and this one at the same time (e.g. on another
  port): two ComfyUIs share the GPU and a render took 179 s instead of ~30 s in a test.

### The Home tab
The first tab is an ordinary image generator: prompt, settings, Generate. Everything below the
decision-making is shared with the loop - the same workflow patching, LoRA handling and reference
plumbing - so an image made by hand and one made in round 3 of a job come out of the same pipeline.

**Prompts.** Either write the positive and negative prompts yourself, or describe the image in
words and let the LLM write them. The two are mutually exclusive and the UI enforces it: text in
the description locks the prompt boxes, text in either prompt box locks the description, and
"Clear prompts" unlocks. Whatever the LLM writes lands in the same boxes you would have typed
into, so you can see and edit it. With a reference image and no description at all, the prompt is
written from the image.

**References.** One reference supplies the character, style *and* pose. Tick **Separate pose
reference** and it splits in two: one image for the character and drawing style, another (or a
saved pose from the library) for where the limbs go. As in a job, img2img starts from the
reference centre-cropped to the output size (transparency flattened onto white); **auto** size
follows the reference's shape, or with no reference, the pose's.

**LoRAs.** The picker lists every classified LoRA with its Civitai showcase image, name, tags and
a one-line description of what it does. Each pick gets a strength slider bounded by the range that
LoRA's own test renders supported, and a chip that inserts its trigger words - most of them do
nothing without those. **Let the LLM choose** sends your prompt and reference with a menu of all
of them and asks which suit this render; it answers with menu numbers, picks nothing when nothing
fits, and costs about half a cent (~28K tokens, mostly a cached prefix).

**Run automatically** hands the same settings to the refinement loop, with the loop-only knobs
(threshold, rounds, candidates, hand refine, LoRA mode) under *Automatic run options*.

Each tab owns one thing: Home makes images, Queue lists jobs, History holds finished runs, LoRAs
browses the library, Poses manages saved poses, Settings holds the defaults.

### One LoRA library
`loras.catalog_dir` points at a lora-classifier output, and that is the library: the Home tab's
picker, the loop's automatic picker, the judge's LoRA menu and the prompt writer's trigger words
all read the same records (`lora_catalog.records()` returns them in the shape `LoraLibrary.index()`
always returned, so nothing downstream had to change). The folder scan is the fallback for setups
without a catalog, and **Rebuild the fallback index** on the LoRAs tab is only for that case.

It matters because the scan only saw the folders listed in `loras.dirs` - 83 LoRAs of 574 installed
- and the picker then truncated that to `loras.max_candidates` (14) before the model ever looked.
Picking for a 1990s anime reference from 14 arbitrary entries found a generic flat-colour LoRA;
picking from 523 finds `90s_pony` and `90s4n1m3XLP`, which is what the reference actually needed.

The two stages are unchanged: a text shortlist over everything (one compact line each, ~30 tokens,
so the whole catalog fits), then a visual comparison of the shortlist's example renders against the
reference. `loras.max_candidates` now only caps the detailed path used when no catalog is present.

### LoRA briefs
A title ("Jab Style Illustrious & Flux & Pony & 1.5") tells a model nothing about what a LoRA
does. The classifier writes a long description of each LoRA it finished
(`catalog/descriptions/<category>/<name>.json`: summary, style metrics, compatible subjects and
framings, triggers, the weight curve, strengths, weaknesses, usage tips). `lora_briefs.py` has the
configured judge model condense each one into a brief of fixed fields, shown as one line of
about 60-70 words wherever a model chooses LoRAs (Home's *Also choose LoRAs*, the loop's
shortlist and visual pick, the judge's LoRA menu):

```
[style] Semi-realistic painterly look; sculpted athletic-curvy adult female anatomy, glossy skin,
cinematic light | look: glossy skin, specular highlights, soft diffuse light, desaturated warm
neutrals | best for: solo adult women, full body to close-up | weak: same face every subject,
poor multi-character | trigger: Jabstyle | weight 0.5-1.2, usually 0.9 (recognizable at 0.6,
full look 0.9-1.0, overbaked past 1.2) | suggestive
```

Run it once, and again when the classifier adds LoRAs; it only writes missing or changed briefs
(about $0.0003 each with DeepSeek V4.1 Flash):

```
.venv/bin/python -m ouroboros briefs            # missing or changed ones
.venv/bin/python -m ouroboros briefs --force    # all of them
```

`--only TEXT` limits it to categories or names containing TEXT, `--limit N` to N briefs. They're
cached in `cache/lora_briefs.json`. LoRAs without a brief keep the old one-line card.

Only LoRAs the classifier finished are described or offered. Those it flagged (the
`caution loras` folder: training tags, example prompts or its own age screening pointing to
minors) are never used. A brief also records whether the description shows the LoRA is made to
depict minors; one that is is left out of every menu. Separately, every render's negative prompt
includes `child, loli, shota, toddler, kid` (`workflow.SAFETY_NEGATIVE`), the same list the
classifier uses for its own test renders, since much of a typical library was trained on
booru-tagged data that includes such images.

### Carrying a character across a pose (IP-Adapter)
A ControlNet steers the *shape* of an image. An IP-Adapter steers what the subject *looks like*, by
conditioning the model on a reference image instead of on words - which is the part a prompt cannot
carry, since "the same character" survives a pose change here where a description of a face does not.
It goes in between the model and the sampler (`workflow._add_ipadapter`), so it stacks with the pose
ControlNet: character from one image, pose from another.

Install: `ComfyUI_IPAdapter_plus` in `custom_nodes`, `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
in `models/clip_vision`, and the SDXL adapters (`ip-adapter-plus_sdxl_vit-h`,
`ip-adapter-plus-face_sdxl_vit-h`, `ip-adapter_sdxl_vit-h`) in the install's `models/ipadapter`
(the Unified Loader matches those filenames exactly). The VIT-G preset needs a bigG encoder that
isn't installed, so it isn't offered.

**Weight matters more than the preset.** On this Pony checkpoint, PLUS at weight 0.8 `linear` burns
the image out - saturated colour, melted anatomy. 0.4-0.6 `linear` is clean, as are the
`style transfer` modes at higher weights, so the default is **0.6**. Expect it to carry palette,
hair and the character of a costume rather than reproduce an outfit exactly; for exact costume
fidelity the refinement loop and its judge are still the better tool.

### Judge backends
Every backend gets the same ordered text and image parts and the same JSON schema.

**DeepInfra** (the default) calls their OpenAI-compatible `/chat/completions` with
`response_format = json_schema` (strict), images as `data:` URLs. Any model their
catalogue tags **vision** can judge; **Load models** in Settings lists exactly those.
The default is `deepseek-ai/DeepSeek-V4.1-Flash` (a 1M-token window, prompt caching,
$0.20/$0.60 per million input/output tokens, and $0.006 per million cached input
tokens). It reasons before answering, so `max_tokens` is 8192: a truncated answer is
retried, and after the last attempt the error says to raise it. The key is set in
**Settings → API keys** (see *API keys* under Setup). Cost comes from the API's own `estimated_cost` when it sends
one, otherwise from `judge.deepinfra.price_per_mtok`, and it counts against
`loop.max_cost_usd`. If a model refuses the schema, the backend falls back to plain
JSON mode; the schema is in the system message either way, and answers wrapped in a
code fence or preceded by a `<think>` block are unwrapped before parsing.

**Ollama** (`/api/chat`) passes the schema as `format`, so output is constrained to
valid JSON, with exactly one entry per candidate. Without that constraint, small models
sometimes score the reference as an extra candidate. `keep_alive` keeps the model
loaded between rounds. Sampling uses the model's own defaults. Don't set temperature 0
on thinking models (e.g. `qwen3-vl`): greedy decoding makes them loop until they hit
`num_predict`, which is capped and retried once. You need a vision model (e.g.
`qwen3-vl`, `gemma3`, `llama3.2-vision`). The default is
`huihui_ai/qwen3-vl-abliterated:8b-instruct`: use an **instruct** tag, not a thinking one.
It takes several images per message, so each is sent at `image_max_side`.
llama3.2-vision models take only one image per message: turn on **Send one grid image**,
and the reference and candidates are tiled into one labelled grid of at most
`judge.contact_sheet_size` pixels (1120, the largest llama3.2-vision reads without
shrinking). Note that `Drews54/llama3.2-vision-abliterated` uses the old llama3.2-vision
format, which Ollama 0.34 refuses to load. Cost is reported as $0, so the round and plateau limits do the
stopping.

**OpenAI** makes one Responses API call per round with `text.format = json_schema` (strict). The input
is the fixed prefix (instructions, rubric, reference image) followed by this round's
candidates, the current params, and the compact history. The output looks like:

```json
{
  "candidates": [{"index": 0, "scores": {"identity": 7, "pose": 8, ...}, "notes": "..."}],
  "best_index": 2,
  "diagnosis": "hair colour too saturated, left hand malformed",
  "edit": {"phase": "repair", "mode": "inpaint", "mask_target": "left hand",
           "prompt_add": [], "prompt_remove": [], "cfg": null, "denoise": 0.45, ...}
}
```

The **score is computed in code** from the rubric (weighted mean → 0–100). The model
doesn't decide when to stop; the threshold does.

### One candidate per judge call
The judge scores each candidate in its own call: the reference plus that one image,
differences listed first, with the same notes, rules and LoRA list (`judge.scoring:
"per_candidate"`). The best is the highest score; ties go to the image with fewer listed
differences. The next edit comes from the best image's own call. Each call is told
that candidate's own settings (seed, cfg, denoise, LoRA set), since its edit is applied
to them. In a LoRA sweep round (round 1) every candidate goes to the judge; the local
pre-filter would otherwise drop some LoRA sets untested.

This is from a position test with qwen3-vl 8B: the same 3 images in all 6 orders, twice.
With several candidates in one call, the model copied scores between neighbouring
images (10 of 12 calls had ties), couldn't separate a strong image from a middling one
(64.9 vs 64.4 average), docked the last image ~4 points and favoured the first on ties.
Three near-identical images tied in 6 of 6 calls, the first always "won", and the shared
score swung 67.8-83.9 with order alone. A round now costs one call per candidate
(~2.5-3K tokens each) instead of one ~7K-token call. `"scoring": "batch"` restores the
old behaviour.

Single-image scores run lower than scores given alongside other images (about 10 points
in the tests), so the default threshold is 78 rather than the 85 used with batch
scoring. In two calibration runs on one reference, images with the wrong hair or several
added items scored 70-72, the good ones 81-82 (confirmed), and one with added shoulder
tabs and an emblem 78.3: at 78 the fresh-look confirmation is what keeps such borderline
images out.

### Stopping only on a confirmed pass
One judge call can be wrong: it can say an issue is fixed when it isn't. So when a
candidate reaches the threshold, the loop asks for a **second, fresh-context look** at
that one image before stopping (`loop.confirm_pass`). The confirmation call gets none of
the notes from earlier rounds, so a claim like "the emblem is gone" can't carry over. It
costs one judge call (~10–15 s), with no extra renders. If the fresh look scores it
below the threshold, the job continues from that image using the fresh look's fix, and
the fresh look's score replaces the first (in testing it was the more careful of the two).

### Building on the best image
Each round builds on the best image so far. If a round doesn't beat it, the judge's new
edit is applied to the best image's settings rather than to that round's (worse)
winner. The log says so ("round 4 didn't beat r03_c1.png (73); the next edit starts
from it"). img2img_best and inpaint_best already start from the best image, so its
settings now go with it, and the refine gate checks the best image's scores.

### Local fixes go to masked repainting
Low-denoise img2img can't add or remove a detail such as an emblem, strap or hand. The
judge is told to name the region (`mask_target`, e.g. "left sleeve") for local
problems. Any edit with a `mask_target` becomes `inpaint_best`, so only that region is
repainted. When the loop reworks an existing image (img2img_best or inpaint), each
candidate gets its own seed. With the same seed and a nearby denoise, the candidates
came out nearly identical.

### Big problems first (refine gate)
Reworking or repainting the best image can't fix a wrong outfit, hairstyle or pose. In a
test, the judge kept choosing hair repaints on an image with the wrong outfit and pose, and
the score went down every round. So `loop.refine_gate` (default: identity and composition
both at least 7/10) must be met before `img2img_best` or inpaint runs. Until then, such
an edit becomes an explore step with the judge's prompt edits and new seeds. It uses
img2img from the reference when composition is off, since that borrows the reference's
layout at or above the minimum denoise. Otherwise it uses txt2img. The log shows when this
happens ("big differences remain (identity 4/10, composition 3/10) …").

### Extra items (optional criterion)
`judge.rubric_optional.extras` scores items the reference doesn't have (accessories,
emblems, straps, props). Whether it counts is a toggle, because additions are fine for
some characters and wrong for others:

- **Settings → Judging → Penalize extra items**: the default (`loop.penalize_extras`).
- **Add a job → Extra items**: default / penalize / allow, for that job only.

When it's off, the judge is told additions are acceptable.

### Keeping the judge's context short
Every judge call is a fresh request, not a growing chat, and the parts that matter
most go first: rubric, goal, rules and the reference image. Models attend best to the
start of their context. What the judge is told about earlier rounds is kept short:

- the last `loop.history_keep_rounds` rounds (default 3) word for word;
- everything older folded into a summary (≤150 words) that the LLM writes. It covers the
  best result so far, which edits helped, which didn't, and what still differs.

The summary is refreshed whenever there are more verbatim notes than that, or when the
last judge prompt used more than `loop.context_fill_limit` (default 0.6) of the model's
context window (`num_ctx` for Ollama). The Run tab shows each round's prompt size, e.g.
"judge prompt 4,512 / 16,384 tokens", and a note when the notes were summarized. If
the images alone fill more than 85% of the window, the log says so. Lower the image size
or the candidates per round in that case.

### Output size
Uploads can be any size or shape. The output size is the SDXL resolution closest to the
reference's shape: 1024×1024, 1152×896, 1216×832, 1344×768 or 1536×640, or the tall
version of each. SDXL works best at about one megapixel, so a 512×512 drawing renders
at 1024×1024 and a 1920×1080 screenshot at 1344×768. The reference is centre-cropped to
that exact shape and resized (`reference_input.png` in the run). txt2img, img2img (which
encodes the reference itself) and masked repaints then all come out the same size.
Transparent areas are flattened onto white, both for rendering and for the judge. A
drawing on a transparent background would otherwise start img2img from black. The
judge still compares against the original upload.

Change it in Settings (default) or per job: **match the reference's shape**, **as saved in
the workflow**, or a specific size. Wide sizes tend to fill the canvas with extra people
(a one-character prompt at 1344×768 produced three in a test). For a single character
on a wide canvas, a tall size is safer. The width/height roles in `nodes.json` point at
the workflow's Width and Height nodes.

### Checkpoints and LoRAs
**Checkpoint:** Settings (default) or a job's form. The list comes from ComfyUI (your
`models/checkpoints` folder). Each checkpoint's family (Pony, Illustrious, Flux) is
guessed from its name; override that in `checkpoint_bases` in config if a name misleads.
Flux checkpoints are greyed out, since this workflow is SDXL.

**LoRA choice** (`loras.mode`, per job overridable):

- **auto**: the judge picks style LoRAs from the folders in `loras.dirs` (default
  `Pony\styles`). LoRAs outside those folders, like a hand-fix LoRA, stay as saved in the
  workflow.
- **workflow**: the LoRAs switched on in the saved workflow; the judge may tune their
  strengths.
- **off**: no LoRAs.

**What the picker knows about each LoRA** (Settings -> LoRAs -> *Refresh from files and
Civitai*; cached in `cache/`, so jobs never wait on the network):

- From the file: kohya training metadata. A dataset folder like `8_Jabstyle` whose token
  also appears in the captions is a trigger word; generic folders like `10_img` aren't.
- rgthree's `.rgthree-info.json`, when ComfyUI already fetched it.
- Civitai, looked up by the file's SHA256: title, base model, trigger words, tags,
  description, up to 6 example images with the prompts and the LoRA weight they used (the
  median becomes the "usual weight"). Example images up to the chosen NSFW level are
  saved as thumbnails. A Civitai API key is optional (Settings → API keys).
- Local results (`cache/lora_stats.json`): every judged candidate adds its score and
  style score to each LoRA it used, at that strength. Later picks see e.g. "tested here in
  14 candidates: avg score 72, avg style 7.9/10, strengths 0.4-0.8".

**How a job picks** (two calls, to fit the prompt budget; see below):

1. A text shortlist: the reference image plus a card per compatible LoRA (title, base,
   triggers, usual weight, tags, description, one example prompt, local results). About
   3.6K tokens.
2. The reference next to one Civitai example image per shortlisted LoRA (5 by default,
   `pick_visual_candidates`). It picks up to `max_loras` (3) with strengths, plus
   alternatives. About 7.7K tokens.

Picking none is a valid answer, and a render without LoRAs is always part of round 1.

Round 1 then renders the picked set, your workflow's own LoRA set (the baseline the picks must
beat), the same render with no LoRAs, and sets with one alternative or other shortlisted LoRA
swapped in, all on the **same seed** (`loras.sweep`), so the judge compares LoRAs rather than
seeds. Round 1 has the largest batch (see *Batch size*), so that's where the most LoRA options
are tried.

After that, each edit changes **one axis** (`loop.enforce_focus`), so the next round shows what
that change did. The judge says which with `focus`:
- `prompt`: prompt and negative tags (with a mode/denoise change or masked repaint if needed);
  the LoRAs and strengths stay.
- `lora_weights`: only the strengths of the LoRAs in use; the prompt stays. The next round
  renders the proposed strengths plus a sweep around them (±0.15, ±0.3, one LoRA at a time,
  within each LoRA's working range from its brief) on one seed.
- `loras`: a different set: swap one, add one, or none at all; the prompt stays. The next round
  compares the new set, the one it replaces, no LoRAs, and further swaps, on one seed.
- `settings`: mode, cfg, denoise, sampler, mask only.

An edit that touches both is cut down to its focus. The judge is told LoRAs fix the look and the
prompt fixes the content, and to try the other axis when one didn't help. Switching LoRAs
(`loras`) is only allowed in the first `loop.lora_switch_rounds` rounds (3) while exploring; the
judge's menu says whether switching is open, and a switch proposed later becomes a strength
change. The judge's menu holds the picks, alternatives, the rest of the shortlist and the
workflow's LoRAs, each with its brief.

### Batch size
Round 1 renders `loop.batch_start` candidates (12, at most 16); later rounds fewer, down to
`loop.batch_end` (3). The batch follows whichever is furthest along: rounds done (each 30%
closer), the phase (refine is 60% of the way, repair all of it), or the score's progress from
round 1 toward the threshold. With the defaults and no other progress: 12, 9, 7, 6, 5, 5, 4...
A job with its own *Fixed batch size* (`candidates_per_round`) keeps that.

Every candidate is judged in its own call, and a round's calls now run side by side up to
`queue.llm_parallel`, so a 12-image round takes about three call-lengths with 4 at once, not
twelve. With the local pre-filter installed (torch, torchvision, transformers), only the top
`send_top_k` go to the judge, except in rounds that compare LoRA sets or strengths.

**Trigger words:** each active LoRA's main trigger word (the first listed) is added to
the prompt at render time. A switched-off LoRA's trigger words are removed only when they
are made-up tokens no other LoRA lists (`p0seA`, `RSV1.2`, `melkor_style`, `PuffyNips`):
many LoRAs list ordinary tags as triggers (`1girl`, `Kiss`, `69`, `full body`), and those
stay, as do your own words. Some LoRAs list many (Melkor lists character names,
princess_xl every princess), so only the main one is automatic. The prompt writer sees them all, plus each active LoRA's example
prompt, and uses the relevant ones. LoRA picks, reasons and alternatives are on the Run
tab and in History.

The LoRAs tab shows the library: example image, base model and whether it suits the
current checkpoint, trigger words, usual weight, local results, and the Civitai link.

### Model presets and the prompt budget
`ouroboros/model_presets.json` holds recommended judge settings per vision model.
Choosing a model in Settings fills them in: context size, max answer tokens, sampling,
grid image on or off, image size, prompt budget. For Qwen3-VL instruct:

| setting | value | why |
|---|---|---|
| `prompt_token_budget` | 8000 | Absolute cap on the judge prompt (~3% of the 256K native window). Accuracy drops as prompts grow, from around 10-20% of the window, well before it's full. Older rounds are summarized to stay under it. |
| `history_keep_rounds` | 2 | Fewer verbatim notes: the middle of a long prompt is recalled worst. |
| `num_ctx` | 16384 | The prompt (up to ~8K) plus the answer (up to 4K) with headroom. |
| `temperature` / `top_p` / `top_k` | 0.7 / 0.8 / 20 | Qwen's recommendation for VL instruct models; steadier scores than 1.0. |
| `image_max_side` | 512 | In Ollama each image costs ~1,100 tokens however small, so fewer images matter more than smaller ones. A hosted model charges by the pixel, so it is also what a judged round costs. |

Placement: the rubric, rules, LoRA menu and reference go first (models attend best to the
start), then the notes and candidates, then a short task reminder at the very end (the
other place they attend well). The JSON schema is sent compactly.

### Reproducing a result
History -> **View rounds** shows a **Reproduce the best image** panel with everything used
to render it:

- the prompts as rendered, with LoRA trigger words;
- the checkpoint, LoRAs and strengths, workflow file and size;
- the mode, the source image (and the masked copy, for inpainting), seed, steps, CFG,
  sampler, scheduler and denoise.

**Download ComfyUI workflow** gives the exact API-format graph that was queued (every
candidate's graph is in `runs/<run>/graphs/`); ComfyUI can open it.
**Copy all as JSON** copies everything. Runs from before this was recorded show "not
recorded" for the checkpoint and LoRAs.

**Delete** on a History card takes two clicks: the first turns it red, the second moves
the run to the recycle bin, `runs/_removed/` (it resets by itself after a few seconds).
**Empty recycle bin** at the top of History deletes everything in there for good, also
with two clicks. Poses delete the same way, into `poses/_removed/`.

### Prompts from a description
A job can have a **description** (plain language) instead of, or on top of, a prompt:

- **Description only:** the LLM writes the positive and negative prompts from the
  description and the reference image (`loop.prompt_sees_reference`). It follows the
  tag style of your workflow's saved prompts: score tags, style/LoRA trigger words and
  rendering tags, but not their subject.
- **Description + your prompt/negative:** the description is merged into your
  prompts. Your tags are kept unless the description contradicts them. The model
  must list what it removed, and anything it drops without listing is put back.

Either way, negative tags that would ban something the positive asks for (e.g.
"belt" when the positive says "thin dark belt") are removed and listed in the notes. The
same check runs when the judge adds a tag mid-run. The prompt is written when the job
starts, saved to `runs/<run>/prompt.json`, and shown on the Run tab with Copy buttons.
**Preview AI prompt** in the Add a job form shows it beforehand, and **Use these as the
prompt** copies it into the prompt fields.

On the Home tab, **Write prompts** only writes prompts. Any LoRAs already selected (picked by
hand in the browser, or earlier by the LLM) are passed to the writer with what each does,
its trigger words and an example prompt, so the prompt includes the triggers and suits the
LoRA's style, character, pose or concept. Ticking **Also choose LoRAs** first has the LLM pick
LoRAs from the whole catalog: the selected ones stay, count toward `loras.max_loras`, and
are shown to it so it adds only what they lack and nothing that fights them (a second style
or pose). The prompts are then written for the full set. A generation whose prompt the LLM
writes at render time (description only) is written for its selected LoRAs the same way.

### The generation queue
Home generations, and auto-fixes started from History, go into one queue (`generate.py`):
one worker, oldest first, so the Home tab is free again as soon as Generate is pressed and
more can be added while one renders. ComfyUI is started, if needed, when a task's turn comes.
The Result panel follows the queue: the running task's stage and progress (against the time
estimate it was queued with), how many wait behind it, and the images of whatever finished
last. The time estimate on Home adds what's still queued ahead.

The Queue tab lists the generations (running, waiting with their place in line, and the
last dozen finished) above the automatic-run jobs. **Remove** (two clicks) drops a waiting
task; on the running one it's **Cancel**, which also takes its prompt out of ComfyUI's queue
or interrupts the render, and a cancelled generation that produced no image leaves no
History entry. The queue lives in memory: tasks still waiting when the server stops are not
kept.

### Auto-fix
The likeness loop compares an image with a reference. Auto-fix (`autofix.py`) looks at an
image on its own, as a person would before keeping it:

1. **Inspect.** The judge model sees the image (`autofix.image_max_side`, 1024 px by default,
   large enough to count fingers) and the prompt it was made from, and lists up to six flaws:
   what's wrong, the region it's in ("left hand", "waist"), its kind, a severity (1 barely
   noticeable, 2 noticeable, 3 glaring), a repair method, and prompt tags for how the region
   should look and what to avoid. It's told not to report anything the prompt asks for (an
   intentionally unusual pose, a stylised build).
2. **Repair** the worst `autofix.fixes_per_round` flaws from `autofix.min_severity` up, one
   after another, each starting from the previous result: `inpaint` finds the region with
   CLIPSeg and repaints only it through the workflow's masked path (`autofix.denoise`, +0.1
   for glaring flaws); `hands` uses the hand refiner (one pass per round repaints every
   hand, at the same strength rule) and falls back to a masked repaint of the hand when it
   finds no skeleton or failed before; `img2img` is a light whole-image pass for problems
   spread everywhere.
3. **Review.** The model sees BEFORE and AFTER, says which targeted flaws are fixed, what
   the repair broke, whether AFTER is better, and lists what's still wrong. The round is kept
   only if the model says it's better *and* the remaining flaws weigh less (summed severity)
   than before: asked "better?", a model says yes too easily, even while describing new
   damage. Flaws the round didn't touch are carried forward when the new list forgets them.
   A discarded repair is reported to the next round so it isn't tried again the same way.

Up to `autofix.max_rounds` rounds; each after the first costs one LLM call plus the renders.
In testing, a waving character with six fingers on both hands came out with five on each
after one kept round; rounds that made a hand worse were thrown away.

Where it runs:
- **Home:** *Auto-fix each image afterwards* (under Generation) fixes every image of the batch
  as part of the queued task. The Result panel shows the fixed image with a link to the
  original.
- **Automatic runs:** `loop.auto_fix` (Settings, or per job under *Run automatically*) fixes
  the final image after the hand refiner. The fix must keep the likeness: it replaces
  `best.png` (the original is kept as `best_before_autofix.png`) only if a fresh look scores
  it at most `autofix.keep_margin` points below the original.
- **History:** an **Auto-fix** button on every image (and *Auto-fix all*) queues it. The result
  sits next to the original as `<name>_fixed.png`; nothing is replaced. The View dialog shows
  both side by side with what was found, each round, and what's left.

Every step's files are in the run folder's `autofix/`, and the record in `autofix.json`.

### Pose ControlNet and the pose library
Uses the xinsir ControlNet Union SDXL "promax" model (`models/controlnet/xinsir_union_sdxl_promax.safetensors`)
in openpose mode. Poses are extracted here, not in ComfyUI: DWPose (RTMW whole-body keypoints via
`rtmlib` and onnxruntime; its person detector is trained on HumanArt, so it reads drawings)
finds the body, hand and face points, and the skeleton is drawn the way openpose ControlNets
were trained (controlnet_aux's DWPose renderer). No ComfyUI custom node is needed.

A job's **Pose** (job form, or Settings -> default pose):
- **none**;
- **from the reference**: every render follows the reference's pose, so txt2img gets the
  composition right without borrowing the reference's colours the way img2img does;
- **a library pose** (Poses tab): the character comes from the reference, the pose from the
  pose image. Nothing may start from the reference then (img2img would bring its pose back),
  the output size follows the pose image's shape, the prompt writer is given the pose's
  description, and the judge gets the skeleton and scores composition against it.

Add poses in the **Poses** tab from any image of a person. The skeleton is saved as
normalized keypoints (`poses/<name>/pose.json`, plus `preview.png` and `source.png`), so it is
drawn fresh at each job's output size. The LLM writes a short tag description of the pose.

Knee and ankle points pinned to the frame edge are dropped: in a cowboy shot the model still
"finds" the legs there, and drawn, they would tell the ControlNet the legs end at the edge.
When the workflow repaints a masked region (Inpaint Crop -> 1024 px -> Inpaint Stitch), the
skeleton goes through a copy of the same crop, so it lines up with the pixels being repainted.

Tested with the soldier prompt, one seed, the "hands on hips" pose: strength 0.6 and 0.9 both
put the pose in place without changing the drawing style. A pose image carries its framing
with it: a pose from a bust shot makes a bust shot. Settings: `controlnet.pose_strength`
(0.6), `controlnet.pose_end` (0.8 of the steps), `controlnet.pose_hands`.

### Hand refiner
HandRefiner's own ControlNet (Civitai 262788) is SD 1.5 only: a ControlNet works only with
the model family it was trained for, so it can't attach to Pony/SDXL. The same idea runs
natively in SDXL instead (`handfix.py`):
1. DWPose finds each hand and fits a 21-point hand skeleton, which always has five fingers.
   Hands it isn't sure of (mean keypoint score under 3.5, e.g. a hand behind the body) are
   skipped.
2. A tight ellipse around the hand is made transparent, and the workflow's masked path repaints
   just that region at 1024x1024 (the crop takes 3x the mask as context).
3. The Union ControlNet (openpose) follows that hand's skeleton; denoise 0.6, strength 0.8.

**Automatic** (Settings -> "Refine hands automatically", or per job): after the loop, the final
image's hands are repainted, both versions get a fresh judge look, and the refined one
becomes `best.png` only if it scores at least as high (`best_before_hands.png` keeps the
original). **Button**: a finished run's **Refine hands** repaints the hands of its `best.png`
in the background, with no further input; each attempt is shown before/after and kept as
`manualN_hands_*.png`, and `best.png` isn't replaced.

In a test, a hand with overlong, spindly fingers came back well proportioned. A looser mask
(0.6 of the hand size as padding) also recoloured the sleeve cuff next to it, hence the tight
default (`hands.pad` 0.3).

### Several jobs at once, one LLM queue
With a slow LLM, most of a round is waiting for its answers while the GPU idles. So up to
`queue.parallel_jobs` jobs (Settings -> Jobs at once, default 2) run side by side. Their
renders queue in ComfyUI, and every LLM call from any job (judging, prompt writing, LoRA
picking, summaries) goes through one first-come-first-served queue (`llm_queue.py`).
`queue.llm_parallel` (Settings -> LLM calls at once) sets how many of them may be in
flight together: **1** for a local GPU, which answers one request at a time, and more
for a hosted API, where queueing just adds latency (4 with DeepInfra). Admission stays
in order whatever the number. While one job waits for the LLM, the others render. The Run tab shows every active job and the LLM queue.
Each run uploads into its own ComfyUI input subfolder, so jobs can't overwrite each other's
images. In a test with a fake 1 s LLM and fake renders, 3 jobs took 22 s side by side and
27 s one after another; the slower the LLM, the bigger the gain.

### Comparing judge models
`python -m ouroboros eval MODEL [MODEL ...]` makes each model judge the same labelled images
(`eval/judge_set.json`, images in `eval/images/`) the way the loop does, each twice:
- quality (0-100) = 50% pair accuracy (known better/worse pairs ranked right) + 30% flaw
  recall (the judge's differences name the known defect: a crown, shoulder tabs, wrong hair)
  + 20% consistency (the two scores of one image agree);
- speed: seconds per judge call (as the loop sees it, including model loading) and the
  model's own output tokens per second;
- efficiency: quality points per minute of LLM time; also failed or retried calls.
Results are saved in `eval/results/`. `--repeats=1` for a slow model, `--temp=X` to try
the same model at another temperature, `--backend=deepinfra|ollama|openai` to say where
the model runs (a name with a `/` and no `:` is taken as a hosted one). Hosted and local
models are measured the same way, so they can be compared directly - but a hosted model's
seconds per call include the network round trip, and its cost does not show up here.

Measured on 12 labelled images (24 calls each); the qwen3-vl abliterated models ran on a
1080 Ti, DeepSeek on DeepInfra:

| judge | quality | pairs | flaws | score spread | s/call | tok/s |
|---|---|---|---|---|---|---|
| **DeepSeek V4.1 Flash** (hosted), temp 0 | **95.1** | **100%** | **100%** | 2.5 | 13 | 55 |
| 8B instruct, temp 0 | 89.1 / 89.5 | 85% | 93% | **0.3-0.5** | 14 | 43 |
| 8B instruct, temp 0.4 | 82.7 | 85% | 100% | 4.8 | 15 | 43 |
| 8B instruct, temp 0.7 | 76.5 | 92% | 100% | 9.8 | 16 | 43 |
| 30B-A3B instruct, temp 0.7 | 75.6 | 85% | 100% | 8.3 | 71 | 17 |
| 30B-A3B **thinking**, temp 0.7 | 84.9 | **100%** | 93% | 6.5 | 110 | 16 |

Two things came out of this:
- **Temperature 0 for instruct judges** (now the preset). At 0.7 the same image scored
  8-10 points apart between two looks, which is most of the gap between "pass" and
  "keep going"; at 0 it is 0.3-0.5, at the same speed. (Thinking models still need a
  temperature: greedy decoding makes them loop.)
- **The 30B MoE instruct judge isn't worth it here**: 4.5x slower (71 s vs 16 s per call,
  17 vs 43 tokens/s with most of it offloaded to system RAM), no better at ranking, and
  one call in 24 looped until it ran out of tokens.
- **The 30B thinking judge is the most accurate**: it was the only one to rank all 13
  labelled pairs right, but at 110 s a call it would turn a ~1 minute round of judging
  into ~7.5 minutes.
- **The hosted DeepSeek judge ranks best of all**: every pair right, every flaw named,
  13 s a call, and it names more real differences per image than the 8B does. It costs
  about $0.001 a call: $0.004 per judged round, ~$0.05 for a 10-round job. Its one
  weakness was repeatability, since hosted MoE inference isn't deterministic even at
  temperature 0 - capping the difference list fixed most of that (below), and the
  fresh-look confirmation covers what's left.

#### Tightening the judge prompt
Repeatability is a prompt property, not just a model property. Judging the same 12 fixed
images **five times each** (so every difference between looks is judge noise, not the
image changing), five wordings of `judge.INSTRUCTIONS` gave:

| variant | spread | worst | pairs | flaws | good-vs-flawed gap |
|---|---|---|---|---|---|
| as written before | 9.5 | 16.5 | 100% | 100% | 12.4 |
| **at most 8 differences** (shipped) | **5.0** | **11.8** | 100% | 100% | 13.1 |
| score by deduction rule | 8.2 | 16.5 | 100% | 100% | **22.4** |
| both | 7.2 | 13.5 | 100% | 100% | 16.4 |

The length of the difference list is what moves a score: a judge that enumerates
everything decides afresh each time whether the background shade deserves a line. Capping
it at 8, most visible first, halved the spread and made calls faster for having less to
write, with pair accuracy and flaw recall untouched.

Scoring by an explicit deduction rule (tag each difference major or minor; start at 10,
subtract 1 a major and 1 per two minors) nearly doubled the gap between the good and
flawed bands - real threshold headroom - but left the wobble at 8.2, because the rule
consumes an uncapped list, and combining the two gave most of that gap back. Worth
revisiting as a `severity` field in the schema rather than an instruction.

`seed` is passed through to the API when set, but it is not a fix: three calls at the
same seed still produced two different answers.

**Scores are only comparable within one judge.** Each model uses the 0-100 range
differently, so a threshold calibrated for one is meaningless for another. On the
labelled set:

| band | DeepSeek V4.1 Flash | qwen3-vl 8B, temp 0 |
|---|---|---|
| good images | 55-69 (mean 63) | 53-91 (mean 70) |
| flawed images | 43-61 (mean 52) | 42-70 (mean 55) |
| wrong character | 37 | 37 |

DeepSeek separates the bands cleanly but scores everything lower: its best images sit
near 69, so a threshold of 78 (calibrated for the 8B) can never be reached and every job
would run its full round budget. `loop.threshold` is **62** for this judge: the four
clearly-good images score 62-69, so a job can stop as soon as it reaches that cluster.
One flawed image (shoulder tabs the reference lacks) averages 61 and has touched 66 on a
single look, which is what the fresh-look confirmation is there to catch. Recalibrate
after changing judges - `python -m ouroboros eval` prints the bands.

### A second opinion for the pass check (judge.confirm_model)
Set `judge.confirm_model` (Settings -> Judge -> *Second opinion for the pass check*) to
use a different, more careful model **only** for the fresh look that decides whether a
job may stop. The fast model still ranks every candidate each round; the careful one
runs once per passing candidate, which is where a wrong call costs the most. Its sampling
settings come from its own preset (so a thinking model keeps its temperature and token
budget), and its calls queue with everyone else's. It has to run on the backend in use,
so the selector offers that backend's models.

On a single GPU the two local models don't fit at once, so each check also costs a model
load in each direction (about 30 s for the 8B, 80 s for a 30B). Worth it for the last
call of a job, not for ranking. On DeepInfra there is nothing to load, so a heavier model
(a large Qwen or a Claude) costs only its own latency and tokens for that one call.

### Jobs with only an image; two-stage goals
A job with neither a prompt nor a description gets its prompt written from the reference
image alone. A job can set `"stretch": {"from": 75, "extra_rounds": 10}`: once the best score
reaches 75, it gets 10 more rounds (from that round) to reach its threshold, and the plateau
rule no longer ends it early. `summary.json` records the round each milestone was reached.

### Controlled changes (from a manual tuning session)
- Each round's first candidate keeps the seed of the image the edit builds on, so its score
  shows what the edit itself did; the others explore new seeds (`loop.controlled_seed`).
- New prompt tags go into the line they belong with ("red shirt" joins the clothing line)
  instead of the end, where tags carry least weight; removing tags keeps the section layout.
- Added weights are capped at 1.5, and the judge is told not to raise a weight that already
  failed but to change the approach (a LoRA strength, the mode, a masked repaint).
- The judge lists head direction and eye gaze separately (a turned head can still look the
  other way), and face/chin shape and proportions.
- Prompts are written one section per line with a blank line between sections: quality,
  style, character, hair, clothing, pose and framing, background.

## 4. Build order

1. **Export workflows** (txt2img, img2img, inpaint) in API format and fill in
   `workflows/nodes.json`. Test `comfy.py` + `workflow.py` alone: one job, fixed
   params, image saved.
2. **Judge only:** score existing images against a reference. Calibrate the rubric
   and threshold on images you've already rated by hand. This is the most important
   step: a noisy judge causes most wasted rounds.
3. **Loop with planner:** single phase, batch of 4, max 8 rounds.
4. **Phases + local masks** (CLIPSeg/SAM3 inpaint for the repair phase).
5. **Local pre-filter** (CLIP/DINOv2) and plateau stopping.
6. ~~**Queue + button**~~: done as the web UI.
7. **Priors from history:** start new jobs from the best-performing settings.

## Setup in detail

Python comes from [pyenv](https://github.com/pyenv/pyenv); `.python-version` pins the
version (3.13.15). Dependencies go in a standard `venv` at `.venv`:

```
pyenv install --skip-existing
pyenv exec python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(On Windows: `.venv\Scripts\pip`.) Both launchers use `.venv` when it exists.

**ComfyUI requirements.** An SDXL checkpoint (Pony/Illustrious family; the LoRA logic is
built around them), and these custom nodes, installed from ComfyUI Manager or cloned
into `custom_nodes/`:

- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) (Power Lora Loader)
- [WAS Node Suite](https://github.com/ltdrdata/was-node-suite-comfyui) (image/latent switches, CLIPSeg masking)
- [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus), with
  `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` in `models/clip_vision` and the SDXL
  adapters (`ip-adapter-plus_sdxl_vit-h`, `ip-adapter-plus-face_sdxl_vit-h`,
  `ip-adapter_sdxl_vit-h`, from [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter))
  in `models/ipadapter`
- for poses and the hand refiner: `xinsir_union_sdxl_promax.safetensors` (from
  [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0))
  in `models/controlnet`

**API keys.** Settings → API keys has a row for each hosted service: DeepInfra and
OpenAI (judge backends) and Civitai (optional, for LoRA lookups). Paste a key and press
Save; Clear takes two clicks. Keys are stored in `api_keys.json` next to `config.json`,
readable only by your user, and are never sent back to the page (it only shows whether a
key is set, where it came from, and its last four characters). An environment variable
(`DEEPINFRA_API_KEY`, `OPENAI_API_KEY`, `CIVITAI_API_KEY`) takes precedence when set.

1. **Workflow.** In ComfyUI, choose *Workflow → Export (API)* and save the file into
   `workflows/`. `workflows/nodes.json` names that file and maps its node ids to roles
   (see `nodes.example.json`). Without a `nodes.json`, the included example
   (`nodes.example.json` + `example_workflow.json`) is used. Re-export after changing the workflow; node ids usually
   stay the same.
2. **The judge.** With the default DeepInfra backend, add your key under
   Settings → API keys once the UI is running (step 3). The judge pill in the header
   says when it's found.
   *Using Ollama on the LAN instead:* Ollama listens only on localhost by default, so on
   that computer set the environment variable `OLLAMA_HOST=0.0.0.0`, restart Ollama,
   allow port 11434 through its firewall, and pull a vision model
   (`ollama pull huihui_ai/qwen3-vl-abliterated:8b-instruct`).
3. **Start the UI.** Run `./run-ouroboros.sh` on Linux, double-click `Run Ouroboros.bat` on Windows (or `.venv/bin/python -m ouroboros`). It opens
   http://127.0.0.1:8765 and starts ComfyUI hidden in the background (about 1.5 minutes
   until it's ready; Comfy Desktop doesn't need to be open). In **Settings**, press
   **Load models** under the judge, pick the model, and save (for Ollama, enter its URL
   `http://<that-pc-ip>:11434` first). The pills in the header show whether ComfyUI, the judge, and the workflows are
   ready.

Settings are saved to `config.json`, which overrides `config.example.json`.

## Using it in detail

- **Queue tab:** add a job: a reference image, then a prompt or a description (or
  both), plus optional starting settings, stop rules and the extra-items toggle. The
  **prompt library** shows your saved prompts from `prompt-generator/output`.
  Clicking one opens its full positive and negative prompts with Copy buttons, plus its
  settings and LoRAs. **Use in a new job** fills in the form with them and the library
  image as the reference; **Queue as-is** queues the saved folder unchanged.
  Jobs are also plain folders in `jobs/pending/`, so you can drop a prompt-generator
  folder there by hand, or create `reference.png` + `job.json`
  (`{"prompt": "...", "negative": "...", "description": "...", "threshold": 78}`).
- **Start queue** processes jobs in order. The **Run** tab shows each round live: every
  candidate with its score (✓ outline = round best, "filtered" = dropped by the local
  pre-filter), the judge's diagnosis, the edit it chose next, whether a pass was
  confirmed, and the judge's prompt size. **Stop after round** finishes the current
  round and leaves the job in the queue.
- **Home:** one-off generations with your own settings, added to the generation queue (see
  *The generation queue*). The time estimate next to Generate comes from how fast your last
  renders ran. **Write prompts** turns a description into prompts written for the selected
  LoRAs; tick **Also choose LoRAs** to have the LLM pick some first. *Auto-fix each image
  afterwards* repairs flaws in the results.
- **History:** every finished automatic run and every Home generation (a batch is one
  entry), newest first, as cached thumbnails. For a run, **View rounds** replays it and
  **Run again** re-queues the job. For a Home generation, **View** shows its images,
  settings and prompts; **Rerun on Home** renders that exact entry again (same prompts,
  seed, LoRAs and references; the rerun isn't added to History), and **Use prompts**
  copies its prompts to Home.

Each job ends in `jobs/done/` (threshold reached), `jobs/review/` (plateau or budget
reached; best image kept), or `jobs/failed/`. Removing a job from the queue moves it
to `jobs/removed/`. Every round's images, scores, and the model's reasoning are in
`runs/<timestamp>_<job>/log.jsonl`.

Headless, without the UI: `python -m ouroboros run [--once]`. To run a second copy
of the UI next to one that's already open, set `OUROBOROS_PORT` (e.g. 8766). Both
copies share the same queue, so press Start in only one of them.

## Switching judge backends

Pick the backend in Settings (or set `judge.backend`). Each keeps its own section in
the config, so switching back and forth doesn't lose anything, and picking a model
fills in the settings recommended for it (`model_presets.json`, matched per backend).

- **DeepInfra** (`"deepinfra"`) — hosted, the default. Needs a DeepInfra key
  (Settings → API keys).
- **Ollama** (`"ollama"`) — a vision model on your LAN, no cost and nothing leaves the
  house. Set the URL and pick the model in Settings.
- **OpenAI** (`"openai"`) — needs an OpenAI key (Settings → API keys). `openai.model` and `openai.price_per_mtok` are placeholders: set them to the
  model you use and its current pricing.

Prices only feed the `max_cost_usd` budget check.

The server listens on 127.0.0.1 only. To reach it from another device, set
`web.host` to `0.0.0.0` in `config.json`. There's no login, so only do that on a
trusted network.
