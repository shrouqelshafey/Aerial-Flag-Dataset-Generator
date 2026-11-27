# Aerial-Flag-Dataset-Generator
Development of a synthetic and real-world aerial image dataset for drone-based flag detection and country identification. Includes a Python/OpenCV synthetic image generation pipeline, realistic environmental effects, image augmentation, and YOLO-format annotations for training and evaluating object detection models.


## Synthetic generator

Makes drone frames of 2 m x 1 m flags lying on the ground, exactly as the SIYI A8 mini
(3840 x 2160, HFOV 81 deg) sees them from 50-100 m, with YOLO labels.

### Quick start (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python generate.py --n 200
python tools/qa/validate_labels.py dataset/synthetic_n200_s42
python tools/qa/dataset_stats.py dataset/synthetic_n200_s42
```

Open `dataset/synthetic_n200_s42/previews/` to see the frames with their labels drawn on top.
`python -m generator.scale` prints the scale table (GSD and flag size per altitude).

Options: `--n` frames, `--name` run folder, `--seed`, `--workers` (processes; `--workers 1`
for debugging), `--start` (extend a run), `--no-tiles`, `--config`.

### How it works

One frame = 10 steps, one module each in `generator/`:

| Step | Module | What it does |
| --- | --- | --- |
| 1 Scale | `scale.py` | altitude -> GSD = altitude / f_px, f_px = 1920 / tan(40.5 deg) = 2248 px. 2 m flag = 90 px at 50 m, 45 px at 100 m |
| 2 Background | `background.py` | a ground photo resized to that GSD (its own GSD is in `assets/backgrounds/index.yaml`) |
| 3 Flag | `flags.py`, `flag_render.py` | artwork from `assets/flags` put on a 2:1 sheet without stretching, drawn 4x larger with fabric folds |
| 4-5 Placement, blending | `compose.py` | random position + rotation, one homography moves pixels AND corners, soft edge + contact shadow |
| 6 Distractors | `distractors.py` | unlabelled tarps, fabric, cars (layer, off by default) |
| 7 Camera | `view.py`, `camera.py` | camera tilt (gimbal not exactly straight down, `generator.view`: half the frames, 0-10 deg; flags on the far side smaller, near side bigger), then brightness / contrast, lens blur (40 %), motion blur from drone speed x exposure (30 %), and (when switched on) haze, noise |
| 8 Labels | `labels.py` | corners clipped to the frame; labelled if >= 50 % visible and box >= 10 x 5 px |
| 9 Tiles | `tiling.py` | 1024 px tiles with 20 % overlap |
| 10 Split + save | `split.py`, `writer.py` | train/val/test split BY BACKGROUND, JPEG quality 92 |

`generate.py` runs the steps; `generator/settings.py` reads `config/config.yaml`.

### Configuration

Everything is in `config/config.yaml`. The spec sections (camera, flight, flag, countries,
dataset, annotations, image_effects, training.tiling) are read as they are. The
`generator:` section at the end holds what the spec leaves open, including an on/off
switch for every realism layer under `generator.layers`. Phase 1 keeps the layers off;
switch them on one at a time and compare against real SIYI frames.

### Output: `dataset/<run>/`

```
frames/detect/{images,labels}/{train,val,test}/   class cx cy w h          (+ data.yaml)
frames/obb/{images,labels}/{train,val,test}/      class x1 y1 ... x4 y4    (+ data.yaml)
tiles/detect/...  tiles/obb/...                   the same, 1024 px tiles
metadata/flag_000000.json      altitude, GSD, background, every flag's corners, effects
previews/                      frames with labels drawn on top
generation_logs/generation_report.csv   one row per flag
config_used.yaml  summary.json
```

Train YOLO with `frames/detect/data.yaml` or `tiles/detect/data.yaml` (or the `obb` ones).
For Roboflow, `python tools/export_roboflow.py dataset/<run> --zip` writes
`roboflow_detect/` (train / valid / test + data.yaml, split kept) and its `.zip`.
`dataset/` is not committed; regenerate it with the same seed to get the same images.

### Batch on real backgrounds (Roboflow test set)

`generate_batch.py` pastes flags on REAL drone photos instead of building the ground.
With 200 photos in `assets/batch_backgrounds/`:

```powershell
python generate_batch.py --zip
```

| What | How it is decided (`generator.batch` in config.yaml) |
| --- | --- |
| 50 photos stay empty | `negative_fraction: 0.25`, spread evenly over train / valid / test |
| 150 photos x 3 images | `images_per_background: 3`, each with a random flip / 180 deg turn |
| 1, 2 or 3 flags per image | `flags_per_image`: exactly 150 / 150 / 150, in random order |
| Egypt in 60 % of them | `focus_share: 0.60`: exactly 270 of the 450 flag images, chosen at random |
| No country twice in an image | always; the other countries are balanced (least used goes first) |
| Split by photo | all images of one photo are in the same split: valid/test = ground never seen |

Total 50 + 150 x 3 = 500 images. The run checks all of this from the written label
files and stops with an error if anything is off.

Best practices built in:
- **True size:** the flag gets 2 m / GSD pixels. Put `gsd_m` or `altitude_m` per photo in
  `assets/batch_backgrounds/index.yaml` when known; otherwise the photo is taken as a full
  frame of the config camera (HFOV 81 deg) at a random 50-100 m.
- **Same camera as the photo** (`generator/match.py`): the flag gets the photo's black level
  (haze), its measured sensor noise and a little lens blur. A flag that is cleaner than the
  ground around it is a shortcut the model learns instead of the flag.
- Rendered 4x larger and shrunk (soft, correct edges), random angle 0-360 deg, folds and
  contact shadow, fully inside the picture, never overlapping.
- **Camera tilt and blur** (`generator.batch.view`, `generator.batch.blur`): half the images
  are seen 0-4 deg off straight down. A photo has no ground outside it, so the tilted picture
  is zoomed in to fill its corners (at most about 1.2x, never below the lowest altitude). Then
  30 % get extra lens blur and 30 % motion blur, on the whole picture: flag and ground alike.
- **Reference look** (`generator/look.py`, `generator.batch.look`): half of the images get the
  colour and contrast measured on the Roboflow Universe datasets uavc/flags_gamed_awy (bright,
  flat, pale sand; pastel flags) and uavc/uav2024-flags (airfield). Gains for white balance, then
  Lab contrast matching, applied to the whole image so the flags fade the same way.

Output `dataset/batch_<folder>_s42/`: `roboflow_detect/` and `roboflow_obb/` (Roboflow
layout `train|valid|test/images|labels` + `data.yaml`, `.zip` with `--zip`), `previews/`
(boxes + 1:1 zoom on each flag), `generation_report.csv` (one row per image), `summary.json`.
In Roboflow, upload `roboflow_detect` and keep the existing split.

**Real drone maps (default backgrounds).** `assets/background_sources.yaml` lists 5 free
drone maps from OpenAerialMap (CC-BY 4.0: credit the authors listed there): 4 flat desert
sites in Iraq (1.7-2.5 cm/px) and a small airstrip in Texas (5 cm/px). The GeoTIFFs are too
big for git; download them into `assets/batch_backgrounds_sources/` (each entry has its
`url`), then:

```powershell
python tools/cut_orthomosaics.py
```

It cuts 200 camera frames (3840 x 2160, random 50-100 m, exact `gsd_m` in `index.yaml`) into
`assets/batch_backgrounds/` and 14 large ground textures (37.5 ha of real ground) into
`assets/backgrounds/` for `generate.py`:
- one texture per map cell: the biggest clean square in it (up to 176 m, so even a 100 m
  frame is one piece of real ground, never patched together);
- 3 long strips (about 340 x 85 m) from the ground outside the cells, always train.

Each map is divided into cells bigger than a 100 m frame; a cell is one split group, and its
split is written into BOTH `index.yaml` files (`split:`). So a cell is in the same split in
`generate.py` and `generate_batch.py`: the ground of the batch's valid/test images is never in
`generate.py`'s training images (8 cells train, 2 valid, 1 test). Black map borders are never
used. It rewrites only its own marked block in each `index.yaml`, so your own photos stay listed.

No maps? `python tools/make_demo_backgrounds.py --n 200` builds 200 stand-in frames
from `assets/backgrounds`; then `python generate_batch.py --backgrounds assets/batch_backgrounds_demo`.

### Assets: what is still needed

- **Flags:** all 32 classes have an image in `assets/flags/<class name>.png`
  (united_states = `america.png`). The 26 added later are the public-domain flags from
  flagcdn.com (1280 px wide). Classes 28-31 (iraq, yemen, syria, syria_old) are the
  red-white-black tricolours that look like Egypt from the air: the model must learn the
  emblem, not just the stripes, so `generator.focus_classes` gives Egypt and these four 40 %
  of all flags in `generate.py` (the other 27 share the rest; `share: 0` = all equal). `syria` is the flag used since March 2025 (green-white-black,
  three red stars); `syria_old` (red-white-black, two green stars) is built from it by
  `python tools/flags/make_syria_old.py`. New classes are always added at the END of
  `countries.class_names`, so the ids of older labels never change. A class whose image is
  removed keeps its class id and is simply not drawn.
- **Backgrounds:** the 14 textures cut from the drone maps have exact GSDs and each covers a
  whole frame. The 9 older photos in `assets/backgrounds/index.yaml` have GSDs estimated by
  eye, and several cover only 15-30 m of ground, so they are patched together in an
  85-170 m frame. Ground types are weighted towards the competition site
  (`background_distribution` in config.yaml: desert sand 50 %, compacted soil 20 %, airfield
  8 %, the photo-only types 7-8 % each), and inside a type every file is used in proportion
  to the ground it covers, so every square metre repeats about equally. Still missing:
  a real desert AIRFIELD (the only airfield map is one small grass-and-concrete strip, hence
  only 8 %) and photos of the competition site itself. `germany.png` (a flag) and `Western Desert.png`
  (satellite scale) are excluded.
- **Distractors:** the photos have a painted checkerboard instead of transparency; only
  their clean centre is used (`assets/distractors/index.yaml`).
