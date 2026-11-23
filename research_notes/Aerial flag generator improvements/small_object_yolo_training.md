# Training Ultralytics YOLO (detect / OBB) on small flags in 4K aerial frames, and what it means for the synthetic generator

Scope reminder: 2 m x 1 m printed flags lying flat on desert, nadir drone view at 50-100 m, 3840x2160 frames, flag about 45-95 px long x 22-47 px wide at native resolution; 28 classes (Egypt is the focus class); Jetson deployment. Current set: 500 synthetic 4K frames, 50 empty, 1-3 flags each; plan is Roboflow 3x4 tiling.

Version note (important): as of late 2026 the current Ultralytics family is YOLO26 (OBB tables list only YOLO26 models; a YOLO27 preview is linked). Ultralytics added a native `tile` argument for train/val/predict in **ultralytics >= 8.4.204**, and the SAHI integration guide is now marked deprecated in favour of it. Much older advice (YOLOv5 "tips" page, SAHI guide) still applies conceptually but predates this.

Worked pixel arithmetic used throughout (derived from the user's numbers, not from a source):

| Pipeline | Downscale factor | Flag size seen by model (L x W px) | Flag width in stride-8 cells |
|---|---|---|---|
| Full 4K frame resized to imgsz=640 | 6.0x | 7.5-16 x 3.7-7.8 | 0.5-1.0 |
| Full frame at imgsz=1280 | 3.0x | 15-32 x 7.3-16 | 0.9-2.0 |
| Full frame at imgsz=1920 | 2.0x | 22-48 x 11-23 | 1.4-2.9 |
| Roboflow 3x4 grid (960x720 tiles) then resized to 640 | 1.5x | 30-63 x 15-31 | 1.8-3.9 |
| 3x4 grid tiles trained at imgsz=960 / Ultralytics `tile=640, imgsz=640` (native) | 1.0x | 45-95 x 22-47 | 2.8-5.9 |

---

## 1. Tiling vs full-frame high imgsz vs SAHI-style sliced inference (accuracy, Jetson speed, tile size, overlap, edge cuts, empty-tile ratio)

### Takeaway
What matters is the effective downscale factor, not "tiling vs not": compute scales with pixels processed, so native-resolution tiling of a 4K frame costs about the same as full-frame inference at imgsz=3840. Ultralytics' own benchmarks show full-frame imgsz=1920 can beat tile=640 on VisDrone, while tiling beat full-frame imgsz=4096 on DOTA OBB. For this case, prefer Ultralytics' native `tile` (with 20% overlap, cut-box filtering and seam-aware merging, OBB supported) over Roboflow's 3x4 Grid, which has no overlap. Then benchmark a 2x-downscaled full-frame model (imgsz=1920) against a native tiled model on the Jetson.

### Cited Findings
- `tile=True` uses `imgsz` as the window size. `tile=1024` sets a square window and `tile=[h, w]` a rectangular one. It works in train, val, predict and track, and prediction returns one Results object per original image — [Ultralytics High Resolution Tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Overlap is set by `tile_overlap`, default 0.2 (20%) on both axes. The last row and column align to the image boundary and can overlap more — [Ultralytics High Resolution Tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- In training, each tile is one sample, so `batch` and epoch length count tiles. Boxes and polygons are cropped to the window, and **OBB clipping preserves orientation**. **Instances that keep less than 10% of their original box area are removed**, and **empty windows are kept as negative examples** — [Ultralytics High Resolution Tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Validation with `tile=True` merges predictions per original image before scoring against the original (untiled) annotations — [Ultralytics High Resolution Tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Seam handling: a box cut by an interior window edge is dropped if another window sees the object fully and a kept box covers it. A cut box is kept only if no kept box of the same class covers more than half of it. Objects larger than the overlap can still be split, and fragments are not rejoined. OBB merging uses rotated NMS — [Ultralytics High Resolution Tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Benchmarks (YOLO26s, ultralytics 8.4.216): **VisDrone, full-frame imgsz=1920 scored 23.7 mAP at 5.8 ms/frame, vs 21.7 mAP at 13.5 ms with tile=640.** **DOTAv1 OBB, tile=1024 scored 64.4 mAP at 40.8 ms and 2.2 GB, vs 61.1 mAP at 60.9 ms and 9.0 GB with imgsz=4096** — [Ultralytics High Resolution Tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling). The hardware for these timings is not given in the page as summarised.
- Window counts: 3840x2160 at tile=640 gives 32 windows (8 columns x 4 rows), and at tile=1280 gives 8 windows. imgsz=1280 costs about 4x the compute of imgsz=640 on a whole image — [Ultralytics blog: small object detection with image tiling](https://www.ultralytics.com/blog/small-object-detection-image-tiling)
- Prediction runs one window at a time unless `batch` is set. On 1920x1080 frames at tile=640, `batch=16` cut YOLO26n from 26.6 ms to 10.6 ms per frame — [Ultralytics High Resolution Tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Tiling removes global context. Ultralytics recommends comparing 2-3 tile sizes on your validation set, watching precision as well as recall, and comparing against a whole-image baseline on the same split. Use `cache="ram"` for large sources — [Ultralytics blog](https://www.ultralytics.com/blog/small-object-detection-image-tiling); [Ultralytics tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Exported models (TensorRT etc.) keep their normal input graph, and tiling runs in the Python prediction pipeline. In the Rust inference runtime, tiling is available from Ultralytics Inference 0.0.57 (`--tile`) but is off by default — [Ultralytics tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling); [Ultralytics blog](https://www.ultralytics.com/blog/small-object-detection-image-tiling)
- Example, prediction counts rather than scored mAP: on a VisDrone 1080p frame, YOLO26s returned 35 detections at imgsz=640 whole-image vs 161 with tile=640 — [Ultralytics blog](https://www.ultralytics.com/blog/small-object-detection-image-tiling)
- SAHI paper (Akyon et al. 2022, VisDrone and xView): slicing-aided inference alone added +6.8 AP (FCOS), +5.1 (VFNet) and +5.3 (TOOD). Slicing-aided fine-tuning plus sliced inference added +12.7, +13.4 and +14.5 AP cumulatively — [arXiv 2202.06934](https://arxiv.org/abs/2202.06934). These are not YOLO results, as Ultralytics staff also note — [Ultralytics SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/)
- The Ultralytics SAHI guide is deprecated in favour of `tile`. Its examples use 256x256 slices with 0.2 overlap ratio, and comments use 512-1024. Staff say runtime depends mainly on slice count. The SAHI author suggests Non-Maximum-Merging when large objects split into several boxes — [Ultralytics SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/)
- **Roboflow Tile preprocessing:** Grid mode (rows x columns, default 2x2) has **no overlap setting**. Overlap exists only in Fixed Size and Fixed Aspect Ratio modes, which are app-enabled only for selected workspaces (all workspaces via API) and pad right/bottom edges with black or white. The docs do **not** say how boxes cut by tile edges are handled, or whether empty tiles are kept. Tiling runs before resize, and resize only downsizes — [Roboflow image preprocessing docs](https://docs.roboflow.com/datasets/dataset-versions/image-preprocessing)
- Roboflow guidance: set tile overlap to at least the size of the largest object in pixels. Typical overlap is 0.1-0.3, or 0.3-0.5 for critical detection (Workflows Image Slicer, SAHI-based) — [Roboflow answers: tile overlap](https://roboflow.com/answers/tile-overlap-for-small-objects); [Roboflow search summary of Image Slicer docs](https://docs.roboflow.com/workflows/blocks/blocks/advanced-blocks/image-slicer)
- Jetson reference speeds (YOLO26n, imgsz=640, TensorRT, model-only time excluding pre/post-processing): Orin NX 16GB FP16 4.13 ms / INT8 3.49 ms; Orin Nano Super FP16 4.57 / INT8 3.80; AGX Orin 64GB FP16 2.62 / INT8 2.30. INT8 loses about 1-3 mAP points (e.g. 0.4800 to 0.4490 mAP50-95 on Orin Nano Super). Use `nvpmodel -m 0` and `jetson_clocks` for max performance — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Inferences
- **Do not resize full 4K frames to 640.** The flag would be 3.7-7.8 px wide, at or below one stride-8 cell, and the central emblem that tells Egypt from look-alikes would be 1-3 px. That is unrecoverable.
- **Compute budget on Jetson (rough linear-in-pixels scaling of the 4.13 ms Orin NX FP16 YOLO26n figure; measure on the device):**
  - Full-frame imgsz=1920 is about 9x the 640 cost, roughly 37 ms model time.
  - Native tiling with tile=640 is 32 windows, roughly 132 ms (about 7 fps before pre/post-processing; batching helps).
  - A 3x4 grid at 960 is 12 windows x 2.25, roughly 110 ms.
  - Larger model sizes (s/m) scale this up proportionally.
  - The 2x-downscaled full frame (imgsz=1920, flag 11-23 px wide) is the cheapest option that still keeps flags at about 1.4-2.9 stride-8 cells. It is the cheapest candidate worth testing, and Ultralytics' VisDrone result shows it can beat tile=640 there.
  - The two candidates are A) full-frame imgsz=1920 and B) native tiling at 640 with 0.2 overlap. Test them on the Jetson with FP16 TensorRT, comparing recall on the smallest flags (100 m altitude) and Egypt-vs-look-alike confusion.
  - Because tiling happens in Python around the exported engine, tiled inference on Jetson needs the Ultralytics Python pipeline (or a custom slicer) on top of the TensorRT engine.
- **Tile size and overlap:**
  - The largest flag diagonal is about 105 px (95 x 47 at 45 deg). Roboflow's rule (overlap >= largest object) is satisfied by tile=640 with tile_overlap=0.2 (128 px) or tile=960 with 0.15.
  - The Roboflow 3x4 Grid gives 960x720 non-square tiles with zero overlap. That guarantees cut flags at seams, and Roboflow does not document how it clips or drops them.
  - Recommendation: upload/keep untiled 4K frames and let Ultralytics `tile=True` handle tiling at train and val time, or have the generator emit overlapping tiles with its own clipping rule. If staying with Roboflow, use Fixed Size tiling with overlap via the API rather than Grid.
- **Edge-cut flags:**
  - Ultralytics drops clipped instances keeping less than 10% of their box area. For a 2:1 flag, a 10% sliver is a few pixels of one colour band, which is unclassifiable.
  - If the generator does its own tiling, use a higher keep threshold: keep the label only if at least about 40-60% of the flag area is visible.
  - For slivers below that threshold, either drop them or mask/inpaint the sliver region so the tile does not teach "red pixels at the edge are background". With overlap, every flag is fully visible in at least one tile anyway.
  - These thresholds are a judgement call, not from a source.
- **Empty-tile ratio:**
  - A 3x4 grid on frames with 1-3 flags gives at least 9 of 12 tiles empty per frame, plus 12 empty tiles from each of the 50 empty frames. That is roughly 80-90% empty tiles, far above the 0-10% "background image" guidance (see section 2).
  - Ultralytics tile=640 on 4K (32 windows/frame) would push this above 90%.
  - Either subsample empty tiles in a pre-tiling step to about 10-30% of the training tiles (keeping preferentially the ones with hard-negative distractors), or put many more flags per frame (see section 2). Ultralytics does not document a switch to subsample empty windows.

### Gaps
- No Jetson benchmarks found for YOLO11/YOLO26 at imgsz 1280/1920 or for tiled 4K inference. The scaling above is an extrapolation.
- The hardware used for the Ultralytics VisDrone/DOTA tiling timings was not identified.
- Roboflow's handling of boxes cut by Grid tiles, and of empty tiles, is undocumented. Verify by exporting a version and inspecting it.
- The SAHI paper's slice sizes/overlaps and per-size (APs) gains were not retrieved (only the abstract).

---

## 2. Dataset size: images/instances per class and background-only images

### Takeaway
Ultralytics' long-standing guidance is >=1500 images and >=10,000 labelled instances per class, with 0-10% background images. The current set (~1,000 instances over 28 classes, about 36 per class) is two to three orders of magnitude below that. Because the data is synthetic, the fix is cheap: many more flags per frame, class-balanced sampling with Egypt and its look-alikes oversampled, and empty imagery controlled to about 10% (by tiles, not frames) built mainly from hard negatives.

### Cited Findings
- ">= 1500 images per class recommended" and ">= 10000 instances (labeled objects) per class recommended". Background images (no objects) at about 0-10% of the dataset reduce false positives and need no labels. Every instance must be labelled (partial labelling does not work), with tight boxes. Check `train_batch*.jpg` mosaics to verify labels — [Ultralytics tips for best training results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/) (originally YOLOv5 docs, still the referenced Ultralytics guidance)
- Same page: start from pretrained weights for small/medium datasets. Train 300 epochs to start (600/1200 if not overfitting). Use `imgsz 1280` "if the dataset has many small objects" and run inference at the training size. Use the largest batch possible, `patience 50`, and defaults first — [Ultralytics tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- A maintainer confirmed background images help reduce false positives — [YOLOv5 discussion #2670 (via search summary)](https://github.com/ultralytics/yolov5/discussions/2670). Glenn Jocher (Ultralytics) said on Reddit that background images "slightly reduce recall but greatly improve precision", recommending 1-10% of total images. Note the slight conflict with the "0-10%" in the docs — [search summary of r/Ultralytics thread and YOLOv5 discussion #12242](https://github.com/ultralytics/yolov5/discussions/12242)
- Ultralytics community advice: add hard negatives (images that look like the target but contain none), and add real-world false-positive images to training and retrain — [Ultralytics community: reducing false recognition](https://community.ultralytics.com/t/how-does-yolov8-reduce-false-recognition-how-does-it-reduce-false-recognition-of-similar-objects/1143)
- A widely shared tutorial on adding background images cites the YOLOv8 guide's "10% negative images" but runs no experiment, so it gives no measured effect — [y-t-g: background images for YOLO (Aug 2023, updated Dec 2024)](https://y-t-g.github.io/tutorials/bg-images-for-yolo)
- Ultralytics tile training keeps empty windows as negatives — [Ultralytics tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)

### Inferences
- **Current numbers:**
  - 450 non-empty frames x about 2 flags is about 900-1,000 instances, or about 32-36 instances per class across 28 classes. At 2 flags/frame, reaching 10k instances/class for 28 classes would need about 140k frames.
  - A practical synthetic target is to raise flags per frame to about 8-20 (still physically plausible for a scattered-flag field; sparse frames remain useful for realism) and generate about 5-15k frames. That reaches about 2k-10k instances per class.
  - Weight Egypt and the classes most confusable with it (other red-white-black tricolours, and any pair that becomes identical under 180-deg rotation) at about 2-3x.
- **Background share:**
  - The 10% guidance was written for whole images, where negatives are a minority. For a tiled small-object task the deployment distribution is overwhelmingly empty ground, so precision depends on hard negatives more than on raw empty count.
  - Suggestion: aim for about 10-20% fully empty training tiles, most containing distractors the generator can render (red/white/black tarps, plastic sheeting, painted rocks, vehicles, tents, tyre tracks, debris, shadows).
  - Validate precision on a realistic test set where most area is empty. This is a judgement call combining the sources above.
- **Sim-to-real:** with an all-synthetic training set, hold out some real drone frames (even a few dozen flags) as the test set. The 1500/10000 rule says nothing about the synthetic-to-real gap, and that gap usually dominates.

### Gaps
- No controlled study found that quantifies false-positive reduction vs background-image percentage for YOLOv8/11/26. The 0-10% figure is a rule of thumb.
- No source found on the number of synthetic images needed for sim-to-real transfer for small aerial objects.

---

## 3. Built-in augmentations: which help or hurt, and what to leave out of the generator

### Takeaway
For nadir flags:
- **Keep** mosaic (helps small objects), moderate HSV saturation/value, and translate.
- **Set fliplr=0 and flipud=0.** Mirroring a flag changes its identity, and the docs say to disable fliplr where left-right orientation matters.
- **Use rotation only from one place.** Ideally the generator does it with exact OBB labels and training uses degrees=0, or a small value.
- **Shrink scale** so altitude variation is not doubled.
- copy_paste only works with polygon/OBB labels.
- The generator should supply what YOLO cannot: realistic photometric and physical variation and hard negatives.

### Cited Findings
Defaults and ranges are from the [Ultralytics data augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/):

| Parameter | Default | Guidance from the docs |
|---|---|---|
| `hsv_h` | 0.015 | Helps when lighting changes object colour |
| `hsv_s` | 0.7 | Weather and camera-setting variation |
| `hsv_v` | 0.4 | Lighting variation |
| `degrees` | 0 (range 0-180) | "Useful when objects appear at arbitrary orientations, such as in aerial imagery" |
| `translate` | 0.1 | |
| `scale` | 0.5 (float 0-1, or a (min, max) tuple for geometric tasks) | |
| `shear` | 0 | Start small |
| `perspective` | 0 (0-0.001) | Foreshortening, e.g. drone footage |
| `flipud` | 0 | Use only when objects can appear upside down |
| `fliplr` | 0.5 | "can be disabled where left-right orientation carries meaning" |
| `mosaic` | 1.0 | "Helps small-object detection and context" |
| `close_mosaic` | 10 | Turns mosaic off for the last N epochs, which also turns off copy_paste, mixup and cutmix |
| `mixup` | 0 | |
| `cutmix` | 0 | |
| `copy_paste` | 0 | Mode `flip` (within image) or `mixup` (across images) |

- `copy_paste` needs polygon labels: it works on segment and OBB datasets and on detect datasets with polygon labels, and **has no effect on box-only datasets**. OBB and detect share the standard geometric/colour pipeline, and a low-probability built-in Albumentations set also runs when the package is installed (`augmentations=[]` removes it). `rect=True` disables mosaic, mixup and cutmix — [Ultralytics data augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- Ultralytics staff suggest mosaic and MixUp among small-object remedies — [Ultralytics discussion #8227](https://github.com/orgs/ultralytics/discussions/8227)

### Inferences
- **Flips are label-corrupting for flags:**
  - A horizontal or vertical mirror of a printed flag is not a real flag appearance, unless the back of a see-through print is visible, which the team should decide. Flag designs with asymmetric emblems or text become invalid.
  - Some mirrored flags become other real flags. Example from flag designs, not from a fetched source: Ireland (green-white-orange) mirrored is Côte d'Ivoire (orange-white-green).
  - Set `fliplr=0.0` and `flipud=0.0`.
  - Rotating 180 deg (= fliplr + flipud together) is physically valid for a flag lying flat, and should come from rotation, not flips.
- **Inherent 180-deg ambiguity:**
  - Because flags lie flat at arbitrary yaw, any two classes related by a 180-deg rotation cannot be told apart from imagery (examples from flag designs: Indonesia/Monaco vs Poland; Ireland vs Côte d'Ivoire). Check the 28-class list for such pairs and merge or flag them.
  - Egypt rotated 180 deg is black-white-red with an upside-down eagle. Its main look-alikes (other red-white-black horizontal tricolours, which differ only in the white-band emblem) are only separable by an emblem that is about 7-16 px tall at native resolution.
- **Rotation, done once:**
  - The generator already controls flag yaw and can write exact OBB corners. Sample yaw uniformly over 0-360 deg there and set `degrees=0`, or at most about 10-15, in training.
  - For the axis-aligned `detect` task without polygons, train-time rotation of a box label can only produce a loosened enclosing box. That is another reason to bake rotation into the generator. This is based on how box-only affine augmentation generally works, not verified in Ultralytics source here.
- **Scale, not doubled:**
  - The generator's 50-100 m altitude range already spans a 2x size range (22-47 px width). Default `scale=0.5` (0.5-1.5x) on top would push flags down to about 11 px wide (about 1.4 stride-8 cells at native resolution, worse after downscaling).
  - Either keep generator altitude at the real band and set `scale` to about 0.2-0.3, or narrow the generator band and let `scale` cover it. Do not do both at full strength.
- **HSV:**
  - Colour is class identity (red vs orange, dark green vs black), so keep `hsv_h` at the default 0.015 or lower (e.g. 0.005-0.01).
  - Moderate `hsv_s`/`hsv_v` are fine for sun, haze and exposure.
  - The generator should not add large global hue shifts. It should model physically plausible colour change instead: fading, dust, white balance, shadows, sun glint on vinyl or polyester.
- **Mosaic:**
  - Keep mosaic=1.0 with `close_mosaic=10` (default) so the last epochs train on realistic single tiles/frames. The generator need not compose collage-like layouts itself.
  - Mosaic on 4K frames without tiling would mix 4 full frames into an imgsz canvas, adding more downscaling. That is another reason to train on tiles or at high imgsz.
- **copy_paste:** with OBB labels this becomes available (try 0.1-0.3) for oversampling Egypt. But the generator can do physically consistent pasting (shadows, ground contact, lighting) better, so prefer generator-side placement and leave `copy_paste` low.
- **Perspective/shear:** small perspective (up to about 0.0005) can mimic slight off-nadir gimbal tilt. It is better if the generator renders true camera tilt (e.g. 0-15 deg off-nadir) and leaves these at 0.
- **What only the generator can provide:**
  - Cloth wrinkles and folds, partial sand cover, edge curl, wind ripples, motion blur from drone speed, rolling shutter, sensor noise, JPEG/H.265 compression artefacts at the deployment bitrate, sun angle and shadows, varied desert textures, and hard-negative distractors.
  - Altitude-correct ground sampling distance: render at true GSD rather than resizing a flag sprite, so emblem detail blurs the way optics do.

### Gaps
- No ablation found of specific augmentation values for nadir small-object flag/marker detection.
- Not verified in source code how Ultralytics transforms axis-aligned boxes under `degrees` for box-only detect data.

---

## 4. OBB vs axis-aligned boxes, P2 head / model size, stride limits

### Takeaway
Use OBB. A 2:1 flag at 45 deg fills only about 44% of its axis-aligned box, and OBB gives rectified crops for a second-stage classifier. Ultralytics tiling supports OBB end to end. With native or near-native tiles the flag spans about 3-6 stride-8 cells across its width, so the standard P3-P5 head is adequate. A P2 (stride-4) head is worth trying only if you run full frames at about 1280 or lower, where flag width drops to about 1-2 stride-8 cells.

### Cited Findings
- OBB label format is `class x1 y1 x2 y2 x3 y3 x4 y4` (normalised corners), stored internally as xywhr. Training uses a long-edge form with angle in [-pi/4, 3pi/4). An OBB equals its 180-deg rotation (angle modulo 180 deg). Predictions are not canonicalised, so w may be the short edge, and `result.obb.xyxyxyxy` is correct either way — [Ultralytics OBB task docs](https://docs.ultralytics.com/tasks/obb/)
- YOLO26-OBB on DOTAv1 at imgsz 1024 — [Ultralytics OBB task docs](https://docs.ultralytics.com/tasks/obb/):

| Model | mAP50 (e2e) | T4 TensorRT10 (ms) | Params (M) |
|---|---|---|---|
| YOLO26n-obb | 78.9 | 2.8 | 2.4 |
| YOLO26s-obb | 80.9 | 4.9 | 9.8 |
| YOLO26m-obb | 81.0 | 10.2 | 21.2 |
| YOLO26l-obb | 81.6 | 13.0 | 25.6 |
| YOLO26x-obb | 81.7 | 30.5 | 57.6 |

- Ultralytics tiling supports OBB: clipping preserves orientation, and merging uses rotated NMS — [Ultralytics tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Standard heads are P3/8 to P5/32. P2 adds a stride-4 head that "preserves detail for tiny objects" at more memory and compute cost. P6 (stride 64) "generally will not improve 10 px targets". Staff give no hard minimum object size, and note that sub-4 px boxes don't break the loss but 1 px label errors matter a lot at that scale. The example P2 training uses imgsz=1920. Raising imgsz, tiling and cropping are alternatives, and SAHI helps most when targets are much smaller than the training scale. Earlier staff replies giving strides "2 and 6" were incorrect and later corrected — [Ultralytics discussion #8227](https://github.com/orgs/ultralytics/discussions/8227)
- A user in that thread reported about a 2x slowdown with P2 (user figure, not Ultralytics) — [Ultralytics discussion #8227](https://github.com/orgs/ultralytics/discussions/8227)
- `yolov8-p2.yaml` (outputs P2/4-P5/32) is the reference for adding a P2 head, including to YOLO11. P2 variants have caused problems with seg heads in YOLO26 due to hard-coded mask stride — [Ultralytics community: adding P2 head to YOLO11n](https://community.ultralytics.com/t/adding-a-new-head-to-the-yolo11n-model-to-detect-very-small-objects/876); [Ultralytics community: YOLO26-seg P2](https://community.ultralytics.com/t/yolo26-seg-p2-available-possible/1760)

### Inferences
- **Box fill ratio:**
  - For a 2:1 rectangle at angle t, the axis-aligned box is (2cos t + sin t)(2sin t + cos t). At 45 deg that is 4.5 units^2 against an object area of 2, a 44% fill.
  - Loose HBBs include more sand, merge adjacent flags under NMS, and give classifier crops dominated by background.
  - OBB also lets the generator export exact ground-truth corners at no cost, and enables rectified crops.
- **OBB's 180-deg ambiguity:** a rectified crop still comes out in one of two orientations, so any crop classifier must be trained with 180-deg rotation augmentation. Class pairs that are 180-deg rotations of each other remain inseparable (see section 3).
- **Stride arithmetic:**
  - At native tiles the flag's short side is 22-47 px, or 2.8-5.9 cells at stride 8. That is fine for P3.
  - At full-frame imgsz=1280 the short side is 7-16 px (0.9-2 cells). P2 (stride 4, giving 1.8-4 cells) would plausibly help there.
  - P2 adds compute at the highest-resolution level. On Jetson, it is likely better to spend that budget on less downscaling rather than on P2. This is a judgement, with no benchmark found.
- **Model size:** the DOTA table shows n to s gains +2.0 mAP50 for about 1.75x latency, and s to m only +0.1. YOLO26s-obb (or YOLO11s-obb) is the natural accuracy/speed point for Jetson Orin, with n as the fallback if the tiled window count makes latency too high.
- **Generator implication:** export OBB corner labels (Ultralytics OBB txt), not only HBBs. Keep the corner order consistent; Ultralytics canonicalises angle internally.

### Gaps
- No published YOLO P2 vs P3 comparison found at Jetson scale or for 20-50 px objects specifically.
- Ultralytics issue #981 ("small object under 15px") was referenced by a secondary blog but not retrieved.

---

## 5. Fine-grained country classification at low resolution: one detector with 28 classes vs detect-then-classify

### Takeaway
There is no controlled benchmark for this exact case. Aerial fine-grained benchmarks (FAIR1M) show that most errors are misclassification between similar sub-categories, not missed objects. Domain papers report gains from detect-then-crop-classify. The pragmatic path:
1. Train a single 28-class OBB detector first and read its confusion matrix.
2. If Egypt-vs-look-alike confusion is the dominant error, move to a class-agnostic or coarse-class "flag" detector plus a crop classifier.
3. Run that classifier on native-resolution 4K crops, not downscaled tiles.

### Cited Findings
- FAIR1M (more than 1M instances, more than 15k images, 5 categories / 37 fine-grained sub-categories): error analysis shows classification error accounts for most errors. Removing similar-object (fine-grained) confusions yields larger gains than removing dissimilar-object errors for most categories. Some fine-grained classes (e.g. aircraft C919, ARJ21) have very low AP. Missed detections hit small objects (ships, vehicles) hardest — [FAIR1M, arXiv 2103.05569](https://arxiv.org/pdf/2103.05569)
- A 2024 hierarchical plant-disease framework (detector finds regions, then a separate classifier on crops) improved macro-F1 by 1.1-7.2 points over an EfficientNetV2 baseline across 21 classes, and the second-stage classifier reduced over-detection — [arXiv 2407.17906](https://arxiv.org/html/2407.17906v1)
- Single-stage YOLO on visually similar fine-grained classes can confuse heavily. In a cotton growth-stage study, the best YOLOv8-X reached 47.34% per-class accuracy, and accuracy rose when stages were merged into one class (as reported in a search summary of [arXiv 2307.15816](https://arxiv.org/pdf/2307.15816v1); not independently verified)
- A weed-detection study observed higher precision on a dataset with fewer classes, consistent with fewer similar classes reducing confusion (search summary; [arXiv 2501.17387](https://arxiv.org/pdf/2501.17387))
- Ultralytics tiling supports classification models (probabilities averaged across tiles), but Ultralytics warns tiling removes global context and recommends validating classification carefully — [Ultralytics tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)

### Inferences
- **Why two-stage suits this case:**
  - The detector can run on downscaled or tiled input for speed. The classifier then sees an OBB-rectified crop taken from the original 4K frame (45-95 x 22-47 px), resized to e.g. 64x128 or 96x192 with some context.
  - That is up to 2-6x more emblem pixels than the detector saw if the detector ran at imgsz 640-1920 on the full frame.
  - It also decouples class balance: the classifier can be trained on millions of cheaply generated synthetic crops per class (including 180-deg rotations, blur and compression), far beyond the 10k-instances-per-class bar.
  - A class-agnostic "flag" detector effectively gets all 28 classes' instances (tens of thousands) for one class, which helps localisation recall on the smallest flags.
- **Cost on Jetson:** with 1-3 (or even 20) flags per frame, a small classifier (e.g. YOLO26n-cls or a MobileNet/EfficientNet-B0 at 64-128 px) adds a few ms per frame in a batched TensorRT call. This is an estimate, not benchmarked.
- **Middle ground:** a detector with a few coarse super-classes (e.g. "red-white-black horizontal tricolour", "vertical tricolour", "other") plus a classifier only for the ambiguous groups. Egypt then gets a dedicated Egypt-vs-look-alikes classifier.
- **Generator implication:** export per-instance crops, rectified via OBB, with class labels and a native-resolution flag (taken at render resolution, not from downscaled tiles), alongside the detection labels. Then a classification dataset is produced for free.

### Gaps
- No head-to-head study found of a single many-class YOLO vs YOLO plus crop classifier for small aerial objects, or for flags/insignia specifically. The evidence is from adjacent domains (FAIR1M aircraft, agriculture).
- No data found on the minimum pixel size at which flags with similar tricolour layouts and different emblems can be reliably distinguished.

---

## 6. Evaluation for small objects: mAP50 vs mAP50-95, per-size recall, confusion matrix

### Takeaway
Evaluate on untiled full frames: Ultralytics `val(tile=True)` scores merged predictions against the original annotations, while Roboflow pre-tiled validation would score tile fragments. Report:
- **mAP50 and recall** as primary metrics (localisation is coarse at 20-50 px).
- **mAP50-95** as secondary.
- **Recall per altitude/size bin** and per class, especially Egypt.
- **The normalised confusion matrix**, for Egypt vs look-alikes and for flag vs background.
- **Results on a real-image test set.**

### Cited Findings
- `val(tile=True)` merges predictions per original image before scoring against original annotations, while training loss is computed on tiles — [Ultralytics tiling guide](https://docs.ultralytics.com/guides/high-resolution-tiling)
- Ultralytics recommends comparing tiled vs whole-image baselines on the same validation split, and watching precision as well as recall when choosing tile size — [Ultralytics blog](https://www.ultralytics.com/blog/small-object-detection-image-tiling)
- Ultralytics notes that at tiny scales a 1 px annotation error matters a lot. Remove only ambiguous or inconsistently labelled boxes, not boxes filtered by size alone — [Ultralytics discussion #8227](https://github.com/orgs/ultralytics/discussions/8227)
- In FAIR1M, small objects' main failure mode is missed detection, while fine-grained classes fail through misclassification. The two error types should be separated in analysis — [FAIR1M, arXiv 2103.05569](https://arxiv.org/pdf/2103.05569)
- On Jetson, INT8 TensorRT cost about 1-3 mAP50-95 points vs FP16 for YOLO26n, so re-validate the deployed engine on the device — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Inferences
- **Why mAP50 first:**
  - With a 22 px short side, a 2 px offset already costs substantial IoU. mAP50-95 penalises small-object localisation disproportionately and is less tied to the mission goal of finding and identifying flags.
  - Track mAP50-95 anyway to catch OBB angle errors.
- **Per-size metrics:**
  - Ultralytics val does not print COCO-style APs/APm/APl by default (not verified in this session). Have the generator write metadata per instance: altitude, flag pixel length/width, yaw, occlusion %, distance to tile seam.
  - Then compute recall per bin offline from the saved predictions (`save_json=True` or `save_txt`).
  - Under the standard COCO definition (small < 32^2 px area, medium 32^2-96^2; from memory, not fetched this session), native-resolution flags are about 990-4,465 px^2, so small-to-medium. After any downscaling they are all "small".
- **Confusion matrix:**
  - Use Ultralytics' normalised confusion matrix (saved in the run directory).
  - Check the Egypt row/column against Yemen/Syria/Iraq-type look-alikes and against any 180-deg-rotation pairs.
  - Check the "background" row and column: FP from distractors and FN from missed small flags.
- **Operating point:** pick the confidence threshold for Egypt from the precision-recall curve to meet the mission's FP tolerance per km^2, not the default 0.25.
- **Test sets:**
  - A synthetic test set in which generator parameters are held out (unseen desert textures, sun angles, altitudes at the band edges).
  - A real-drone test set, even if small. Synthetic val mAP alone will overestimate field performance.

### Gaps
- No source fetched confirming whether current Ultralytics val can emit per-size AP natively.
- No published small-object recall-vs-pixel-size curves found for YOLO26/YOLO11 to set expectations for 20-50 px objects.
