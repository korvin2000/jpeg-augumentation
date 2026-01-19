# jpeg-variety

A production-quality Python 3.11+ CLI tool that batch-encodes PNG images into
8-bit JPEG using **multiple encoder backends** and **realistically weighted
JPEG parameter sampling**. The goal is to maximize *plausible* JPEG bitstream and
artifact diversity for training JPEG artifact reduction models.

## Install

```bash
pip install -e .
```

Requirements:
- Python >= 3.11
- Pillow
- External encoders on `PATH`:
  - `cjpeg` (MozJPEG / libjpeg-turbo compatible)
  - `jpeg` (Thomas Richter / Fraunhofer reference encoder)

## Usage

Run as a module:

```bash
python -m jpeg_variety.cli 75 ./pngs ./out_jpgs
```

Or via the console script:

```bash
jpeg-variety 75 ./pngs ./out_jpgs
```

Optional flags:

```bash
jpeg-variety 75 ./pngs ./out_jpgs \
  --seed 1234 \
  --jobs 8 \
  --recursive --mirror-subdirs \
  --manifest ./out_jpgs/encoding_manifest.jsonl \
  --continue-on-error
```

Dry-run (samples options and writes a manifest without encoding):

```bash
jpeg-variety 75 ./pngs ./out_jpgs --seed 1 --dry-run --recursive
```

## Output naming

- Non-recursive: `src/foo.png -> dst/foo.jpg`
- Recursive:
  - With `--mirror-subdirs`: `src/a/b/foo.png -> dst/a/b/foo.jpg`
  - Without `--mirror-subdirs`: flattens into `dst/foo.jpg` (name collisions are possible)

## Metadata manifest

A JSONL file is written (default: `dst_dir/encoding_manifest.jsonl`) with one
record per processed image.

Key fields:
- `input`, `output`
- `encoder` (`cjpeg` or `jpeg`)
- `quality`, `subsampling`, `progressive`, `quant_table`, `dct`, `entropy`, `restart`
- `cmd` (full command-line as executed)
- `seed` (run seed) and `per_file_seed` (stable per-image seed)
- optional `error`

## Extending with new encoders

1. Add a new module under `jpeg_variety/encoders/` (e.g. `my_encoder.py`).
2. Implement `JPEGEncoder` from `jpeg_variety.encoders.base`.
3. Decorate the class with `@register_encoder`.

The factory auto-discovers modules in `jpeg_variety.encoders`.

## Notes on realism

The sampling distributions are intentionally **skewed**:
- baseline/sequential dominates overall, progressive remains substantial
- 4:2:0 subsampling dominates; 4:4:4 increases at higher quality
- standard-ish quant tables dominate; other tables and custom tables are rare
- arithmetic coding is extremely rare

Per-file RNG is derived from `(seed, relative_path)` so a fixed `--seed` yields
repeatable options per image.
