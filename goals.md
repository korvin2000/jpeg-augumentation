You are OpenAI Codex. Write a production-quality Python 3.11 CLI tool that batch-encodes PNG images into 8-bit JPEG with maximum *realistic* variability in JPEG encoding parameters for training a JPEG artifact reduction model (e.g., JDEC).

Core CLI (exactly 3 required positional args):
  1) quality (int 1..100)
  2) src_dir (directory containing .png)
  3) dst_dir (output directory for .jpg)
You MAY add optional flags (seed, jobs, recursive, config, metadata output), but the tool must work with only those 3 required args.

High-level behavior:
- Iterate all PNG files in src_dir (optionally recursive if flag present).
- For each input image, choose ONE encoder implementation at random via a factory, then choose a set of encoder-specific options using *weighted* sampling that approximates “real-world JPEG diversity”:
  - common options (baseline sequential, standard quant tables, 4:2:0 subsampling, integer DCT) should happen most of the time
  - rarer options (custom/non-standard quant tables, progressive with unusual scan behavior, arithmetic coding, exotic DCT variants) should happen less often
  - do NOT sample uniformly; use probability weights and correlations

Initial encoders (plugin architecture; each encoder in its own file):
1) "cjpeg" (MozJPEG / libjpeg-turbo compatible cjpeg CLI)
2) "jpeg" (Fraunhofer/Thomas Richter JPEG reference encoder CLI; executable named "jpeg")

You are given two option inventory files:
- /mnt/data/cjpeg.txt  (help output / switch list for cjpeg)
- /mnt/data/jpeg.txt   (help output / switch list for jpeg)
Use ONLY the options that materially increase variability of the resulting JPEG bitstream and artifacts.
Ignore purely informational, debug, and non-variability switches.

Engineering requirements:
- Clean, readable, structured code; easy to extend with new encoders.
- Encoder plugins registered via a registry decorator or explicit registration.
- Encoder selection via factory with weights (e.g., default: cjpeg 0.75, jpeg 0.25; configurable).
- Deterministic reproducibility:
  - global RNG seed optional (default random)
  - per-file deterministic derivation (e.g., seed + stable hash of relative path) so re-runs reproduce the same options per image if seed is fixed.
- Robust execution:
  - detect missing executables early (shutil.which)
  - good error messages
  - optionally continue-on-error with a flag
- Output naming:
  - mirror file stem: input/foo.png -> dst/foo.jpg
  - if recursive mode, optionally mirror subfolders under dst_dir
- Metadata logging:
  - write a CSV or JSONL (one line per image) with: input_path, output_path, encoder_name, chosen_options (normalized dict), full command line (list), seed used.
  - store it in dst_dir by default (e.g., dst_dir/encoding_manifest.jsonl)

JPEG realism guidance (implement as weights + correlations):
1) Progressive vs baseline:
  - baseline/sequential is more common than progressive overall; progressive is still substantial on the web.
  - default weights: baseline 0.70, progressive 0.30
2) Subsampling:
  - default weights: 4:2:0 0.80, 4:4:4 0.15, 4:2:2 0.05
3) Quantization tables:
  - prefer standard-ish tables most of the time, but include other known tables and rare custom tables:
  - default weights (example): standard/Annex K 0.50, ImageMagick-like 0.25, MS-SSIM/PSNR-HVS tuned 0.20, custom random/perturbed 0.05
4) Arithmetic coding:
  - supported by some encoders but uncommon; keep very rare (e.g., 0.2% to 1%)
5) Correlations:
  - higher quality -> more likely 4:4:4 and progressive
  - lower quality -> more likely 4:2:0 and baseline
  - if custom quant tables are used, keep them rare and mostly baseline-compatible unless explicitly chosen as “nonstandard”
  - restart markers should be occasional and biased toward “off” (common case)

Encoder-specific variability knobs to implement (curated from the inventory files):

A) cjpeg encoder (cjpeg):
Use the quality positional arg as base:
- -quality N
Variability options to sample (weighted):
- mode: -baseline vs -progressive
- quantization:
  - -quant-table N (0..8 per inventory)
  - occasionally use -qtables FILE + -qslots ... (generate temp qtable file on the fly)
  - optionally use -quant-baseline when output should be baseline-compatible
- sampling factors:
  - -sample HxV,... (choose canonical patterns matching 444/422/420)
- DCT method:
  - -dct int (most), -dct fast (rare), -dct float (rare)
- restart interval:
  - -restart N (rare; choose small values, sometimes “rows” vs “blocks with B” if supported by encoder)
- progressive scan / R-D tuning knobs (rare-ish, not always):
  - -dc-scan-opt {0,1,2}
  - trellis / tuning:
    - default keep trellis on; rarely disable: -notrellis or -notrellis-dc
    - -tune-hvs-psnr (common among tune modes), -tune-ssim/-tune-ms-ssim (rare)
  - -fastcrush occasionally to reduce scan optimization variability
Do NOT bother toggling things that are default-enabled with no clear off-switch unless it changes output meaningfully.

Implementation detail:
- cjpeg typically takes PPM/PGM/BMP/TGA, not PNG. Convert PNG->PPM in-memory using Pillow.
- Prefer streaming PPM via stdin if possible; otherwise use a NamedTemporaryFile.
- Always output 8-bit JPEG.
- Ensure alpha is handled: convert RGBA->RGB (choose a deterministic or configurable background, default white).

B) jpeg encoder (Fraunhofer/Thomas Richter “jpeg” tool; executable "jpeg"):
Use the quality positional arg as base:
- enforce encoding with -q N (per inventory)
Variability options to sample (weighted):
- baseline vs extended vs progressive:
  - -bl (baseline) vs default extended sequential
  - -v (progressive) with optional -qv (simplified scan pattern) sometimes
- entropy coding:
  - -h (optimize Huffman) sometimes
  - -a (arithmetic coding) very rare
- subsampling:
  - -s WxH,... using the tool’s convention from the inventory
    - default 444 is 1x1,1x1,1x1
    - 420 often used is 1x1,2x2,2x2 (per inventory text)
- quantization:
  - -qt n (0..8)
  - rarely -qtf file (generate temp file with 64*2 integers: luma+chroma)
- restart markers:
  - -z mcus (rare)
- optional artifact-shaping knobs (rare; only if you judge them variability-relevant):
  - -dz (deadzone quantizer)
  - -oz (optimize quantizer)
  - -dr (de-ringing filter)
Ignore HDR/residual/alpha/JPEG-LS/etc; this tool should produce standard 8-bit JPEG artifacts.

Implementation detail:
- This encoder likely expects PPM/PGM input. Convert PNG->PPM (temp file), then call:
  jpeg [options] source.ppm target.jpg

Architecture / code organization (must implement):
Project layout:
  jpeg_variety/
    __init__.py
    cli.py                 (argparse entrypoint)
    pipeline.py            (iteration, RNG, metadata writing)
    config.py              (weights, defaults, optional JSON/YAML config)
    utils/
      rng.py               (seed + per-file deterministic RNG)
      files.py             (file discovery, path mirroring)
      subprocess.py        (safe run wrapper)
    encoders/
      __init__.py          (registry + factory)
      base.py              (JPEGEncoder interface)
      cjpeg_encoder.py
      fraunhofer_jpeg_encoder.py
  pyproject.toml           (console_scripts entrypoint, deps)
  README.md                (usage, install, how to add encoders)

Core abstractions:
- class JPEGEncoder(Protocol or ABC):
    name: str
    weight: float
    def is_available(self) -> bool
    def sample_options(self, base_quality: int, rng: random.Random, context: EncodeContext) -> EncoderOptions
    def encode(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> CompletedProcess
- registry decorator @register_encoder
- EncoderFactory that:
    - auto-imports encoder modules
    - filters available encoders
    - weighted-random selects an encoder per image (weights configurable)

Option sampling implementation:
- Implement a small weighted-choice helper supporting:
    - weighted categorical sampling
    - Bernoulli(p) toggles
    - correlated choices (e.g., based on quality bucket: low/med/high)
- Define “quality buckets” derived from the base quality (e.g., <=40, 41..75, >75).
- Permit small optional “quality jitter” (e.g., +-3) but default OFF to respect the positional quality arg strictly.

Metadata:
- Write JSONL manifest with normalized option keys:
  {
    "input": "...",
    "output": "...",
    "encoder": "cjpeg",
    "quality": 75,
    "subsampling": "420",
    "progressive": true,
    "quant_table": {"kind":"predefined","id":3},
    "dct": "int",
    "entropy": {"huffman_opt": true, "arithmetic": false},
    "restart": null,
    "cmd": ["cjpeg", "-quality", "75", "..."],
    "seed": 12345
  }
Make sure the schema is stable and encoder-agnostic where possible.

Deliverables:
- All files above, ready to run as:
    python -m jpeg_variety.cli 75 ./pngs ./out_jpgs
  and as installed console script (e.g., jpeg-variety).
- Include docstrings and comments explaining the rationale behind weights and correlations.
- Include a small “--dry-run” option to print chosen encoder/options without encoding (optional but recommended).

IMPORTANT:
- Only include encoder CLI switches that materially affect JPEG encoding variability (quantization, subsampling, progressive/baseline/scan patterns, DCT method/precision where applicable, entropy coding, restart markers, encoder-specific R-D tuning that changes output).
- Keep defaults realistic; rare options must remain rare.
- Favor correctness and maintainability over micro-optimizations.
