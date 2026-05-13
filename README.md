# Drift Dataset

Generate synthetic moving-circle samples and visualize them as videos.

Each generated sample is saved as a separate `.npz` file named from a content hash.
A sample file contains `frames`, `optical_flow`, `coords`, `path_name`,
`metadata_json`, and `sample_hash`.

## Setup

```bash
git clone https://github.com/lmartinez2001/drift-dataset.git
cd drift-dataset
uv sync
```

## 1. Generate Samples

Interactive:

```bash
uv run drift-dataset
```

Scripted:

```bash
uv run drift-dataset --frames 24 --samples 1000 --width 128 --height 128 --output-dir data/samples
```

This writes files like:

```text
data/samples/8f01f9c3a9e3d126.npz
```

Sample generation shows a progress bar.

## 2. Visualize Samples

Render one sample:

```bash
uv run drift-visualize data/samples/8f01f9c3a9e3d126.npz --output data/sample.mp4
```

Render every valid `.npz` sample in a directory:

```bash
uv run drift-visualize data/samples --output data/videos
```

Directory visualization shows a progress bar. Use `--fps` and `--scale` to adjust playback:

```bash
uv run drift-visualize data/samples --output data/videos --fps 12 --scale 4
```

## 3. Test

```bash
uv run python -m unittest discover -s tests
```
