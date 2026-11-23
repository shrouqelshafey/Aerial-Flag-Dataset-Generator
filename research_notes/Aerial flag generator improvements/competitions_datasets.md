# Lessons from UAV ground-target competitions and public datasets (for an ICMTC aerial flag generator)

Research date: 2026-11-23. Scope: AUVSI/RoboNation SUAS ODLC, Teknofest, ICMTC UAVC and comparable events; public flag and background datasets; ICMTC flag-task rules.
Source quality: very few teams publish quantitative sim-to-real results. Most of the material is team docs, READMEs and generator repos, not peer-reviewed papers. The most directly relevant source is a 2026 GitHub repo from another ICMTC UAVC-9 team (Maher-star-eng/ICMTC-2026). It contains real numbers on look-alike flag confusion. Its rule statements come from one team and are not official.

## Q1. How did SUAS / Teknofest / ICMTC teams generate synthetic target data, what accuracy did they get, and what failed on the field?

### Takeaway
Most teams paste targets onto real aerial photos (2D copy-paste) or render them in Blender/AirSim. The failure mode they name most often is that the detector learns compositing artifacts, such as sharp pasted edges and too-clean colours, instead of the target. Published field accuracy numbers are almost non-existent. In the one ICMTC flag pipeline with numbers, the failure was "not detected at all" for look-alike red-white-black tricolour flags, not wrong labels.

### Cited Findings
**Open-source SUAS generators**
- UAV Austin "hawk-eye" (SUAS, Python):
  - Pipeline: take a base shape (circle, square), change its colour, paste a letter or number on it, then paste the shape into an aerial image.
  - Motivation: real target images are scarce ("one competition per year and a limited number of flight tests").
  - Stated failure: synthetic images contain "artifacts in the shape edges" and predictable colour characteristics, which models learn quickly to spot fake targets.
  - Planned fixes: capture real targets in flight tests or from building rooftops; add pixel-level colour variation and blur; explore generative models to make synthetic data look real.
  - Source: [hawk-eye data generation docs](https://uavaustin.github.io/hawk-eye/data_generation/overview.html) (undated page; the repo predates 2022, exact year unverified).
- dec4234 "TargetGeneratorV2" (Rust, MIT licence, SUAS):
  - Varies terrain fidelity, brightness, contrast, ambient light, angle, target size, shape and colour.
  - Explicitly adds distractor objects ("tents, trees, runway markers, roads, people").
  - Simulates about 150-300 ft altitude.
  - Annotation format not documented; no results reported.
  - Source: [GitHub dec4234/TargetGeneratorV2](https://github.com/dec4234/TargetGeneratorV2) (year not shown on page).
- amitibo "auvsi-targets": an older Python shape and letter sample generator used for the 2015-2016 competitions (outside the preferred date range, historical only). Source: [GitHub amitibo/auvsi-targets](https://github.com/amitibo/auvsi-targets).

**A 2025 SUAS team write-up (uas2030.github.io; team identity not stated)**
- Dataset generation:
  - Blender renders target objects onto UAV-captured aerial backgrounds.
  - Project AirSim generates photorealistic images: a 3D model per class is placed randomly without overlap, with automatic annotation.
  - About 12,000 "synthetic and real composite images" for training and validation.
  - Augmentation: noise, transformations, lighting changes.
- Model and performance:
  - YOLOv11 runs at 32 FPS on a Jetson Orin NX.
  - "High mAP" in every class except one (bats). No numeric mAP is given.
- Trade-off: input image size was reduced for speed, and this "hampers the model's ability to detect smaller objects".
- Source: [UAS 2030 SUAS 2025 ODLC page](https://uas2030.github.io/docs/suas2025/object-detection-localisation-and-classification/).

**AUVSI-SUAS 2022 lightweight pipeline (Poddar & Jain, arXiv 2022)**
- Classical CV region proposals, then unsupervised segmentation plus a false-positive removal stage, then a light alphanumeric and shape classifier.
- The abstract gives no numbers. Source: [arXiv 2208.06804](https://arxiv.org/abs/2208.06804).

**Teknofest**
- Poznan University (PUT) team, Teknofest 2022 search-and-rescue mission:
  - The team reportedly used YOLOv5 and YOLOv7, and the mission failed in the competition round.
  - Only a search snippet supports this: [putvision.github.io/competition/teknofest-2022](https://putvision.github.io/competition/teknofest-2022). The page returned HTTP 404 when fetched on 2026-11-23, so the exact failure cause is UNVERIFIED.
- Other Teknofest target-detection repos found in search use classical colour or circle detection (OpenCV red circle), not learned detectors: for example [yilmaz0734/auto-mission-updater-opencv](https://www.github.com/yilmaz0734/auto-mission-updater-opencv).
- A community Roboflow dataset "TEKNOFEST-2024" (1,105 images) exists. Its content and task are not described. Source: [Roboflow TEKNOFEST-2024](https://universe.roboflow.com/neo-stellar-3kknq/teknofest-2024/dataset/1).

**ICMTC 2026 UAVC-9, team "Maher-star-eng" (GitHub, created 2026-07-20, updated 2026-10-03)**
- Data and model:
  - YOLO11n trained on a private Roboflow project `flag-detection-0swrg`: 196 country classes, 27,812 images (24,138 train / 2,290 valid / 1,384 test).
  - 2,126 of those images are no-flag backgrounds taken from soccer-field, tarp, airport and desert Roboflow datasets.
  - 640x640 stretch resize. Roboflow 10x augmentation: 90° rotation, ±40% brightness, ±27% exposure, 0-1.8 px blur, 1.92% salt-and-pepper noise, ±45° bounding-box rotation.
  - Source: [DATASET.md](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/training/DATASET.md).
- Look-alike flags fail as missed detections:
  - Egypt, Iraq, Syria and Yemen (red-white-black horizontal tricolours) were mostly missed outright, not mislabelled.
  - Correct recall at conf 0.6: Egypt 0/30, Iraq 0/31, Yemen 2/28, Syria 10/31.
  - Class-agnostic recall at conf 0.25: 0.97 / 1.00 / 0.89 / 0.94.
  - Correct-class recall at 0.25: only 0.23 / 0.68 / 0.43 / 0.71.
  - Across 120 ground-truth boxes, only one was "matched but mislabelled".
  - Source: [resolution_and_remap_report.md](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/training/resolution_and_remap_report.md).
- imgsz sweep (640 / 960 / 1280):
  - Larger input helped only Yemen (correct@0.25 0.46 → 0.79) and Syria (0.74 → 0.90/0.84).
  - Egypt and Iraq stayed flat.
  - Mac latency was 13 / 20 / 32 ms per image. Same source.
- Deployed design:
  - Raspberry Pi 5 running ncnn YOLO11n, SIYI A8 mini gimbal camera (RTSP/H.265), Cube/ArduPilot flight controller.
  - Class masking restricts outputs to the 4 "briefed" flags before argmax.
  - Thresholds: 0.25 for those 4 flags, 0.60 for all others.
- Reported results:
  - Worst-case F1 ≥ 0.936 on the other 192 classes.
  - Zero false positives on 444 background validation images.
  - Live RTSP capture is not implemented and no real flight was tested.
  - Source: [README.md](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/README.md).
- Dry run:
  - The team's synthetic "schematic tricolour" test frames gave 0 detections (max class score 0.0064).
  - The report concludes "a real recording is still needed" before any accuracy claim.
  - A one-frame gimbal error of −1° moved a single-frame geolocation by about 1,086 m. Median aggregation over a track rejected it.
  - Source: [dry_run_report.md](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/training/dry_run_report.md).

**Second ICMTC UAVC-9 repo (zakariahmedd34/uav-gps-localization, fixed-wing)**
- The author warns that a 2 m flag is "~6-10 px at YOLO's old 640 input". The repo runs YOLO at `--imgsz 1920` by default.
- Camera: GoPro Hero 13, Linear lens mode only (the calibration is invalid otherwise).
- Source: [GitHub zakariahmedd34/uav-gps-localization](https://github.com/zakariahmedd34/uav-gps-localization).

**General sim-to-real evidence (adjacent domains)**
- A 2026 Isaac Sim fruit-detection study found real-only training gives the highest accuracy, synthetic-only loses accuracy to the domain gap, and hybrid training clearly beats synthetic-only. Source: [arXiv 2603.28670](https://arxiv.org/html/2603.28670v1).
- A 2023 detector study found models get their best mAP on the data type they were trained on (synthetic vs real). Source: [arXiv 2312.00694](https://arxiv.org/html/2312.00694v1).
- A UAV paper argues that a "meta gap" (altitude, viewing angle, time of day) is a separate source of synthetic-real mismatch. It aligns these metadata between simulator and real UAV. Source: [arXiv 2112.12252](https://arxiv.org/pdf/2112.12252.pdf).
- MIT Media Lab (2022) used Unreal Engine domain randomization (lighting, materials, pose) for drone gate detection. It notes that performance in simulation "often does not directly transfer to the physical world" and reports no real results. Source: [MIT Media Lab project](https://www.media.mit.edu/projects/domain-randomization-synthetic-data-auav/overview/).

### Inferences
- The other ICMTC team's numbers point to the main risk for this team. The four look-alike tricolours are told apart only by small central emblems:
  - Egypt: eagle
  - Iraq: green Takbir script
  - Syria: three red stars on a green-white-black field since 2025, not the old two green stars on red-white-black
  - Yemen: plain tricolour
- At 640 input and 100 m, those emblems are a few pixels wide. The generator should therefore:
  - (a) oversample these confusable classes;
  - (b) render emblems at realistic, degraded resolution (downsample → blur → JPEG/H.265), not as crisp vectors;
  - (c) emit high-resolution tiles or crops so training sees enough pixels per flag;
  - (d) report per-class recall on the confusable group, not only overall mAP.
- hawk-eye's edge-artifact lesson applies directly to pasting flag PNGs:
  - Use alpha feathering or soft edges, colour and illumination matching to the background, shadows, fabric wrinkle or warp, and noise added after compositing.
  - Otherwise the detector may key on paste seams. Seams will not exist on real fabric lying on sand.
- Adding realistic distractors (TargetGeneratorV2) and many no-flag backgrounds (Maher team: about 7.6% background images, zero background FPs) appears to be a cheap, effective guard against false positives:
  - runway markings, tarps, tents, vehicles, people, cones, shadows, and other rectangular coloured fabric.
- Reading of the Maher threshold results: a lower class-specific threshold for a confusable group, combined with class masking to the briefed flags, recovered most detections. This only works if the briefed flag list is known before flight (see Q4).

### Gaps
- No team published a field-measured detection rate versus altitude for synthetic-trained flag or ODLC detectors.
- No SUAS team journal paper with quantitative synthetic/real mixing ratios was retrievable. The robonation.org and suas-competition.org paper archives did not appear in search, and suas-competition.org/rules returned 404.
- PUT's Teknofest 2022 failure cause is unverified (page 404).
- Whether the Maher team's Roboflow data used the new (2025) Syrian flag or the old one is not stated.

## Q2. What do team papers and reports say about dataset size, synthetic/real mixing, tiling, onboard inference (Jetson) and altitude vs reliability?

### Takeaway
Reported dataset sizes run from about 12k (SUAS 2025) to about 28k images (ICMTC flags, 196 classes). Teams on Jetson Orin NX or Raspberry Pi 5 shrink the input for speed and openly report that this hurts small-target recall. Two ICMTC teams independently flag the 640 px input as too small for a 2 m flag at 50-100 m.

### Cited Findings
- SUAS 2025 team: about 12,000 synthetic and real composite images. YOLOv11 at 32 FPS on Jetson Orin NX. A reduced input size "hampers the model's ability to detect smaller objects". No tiling mentioned. Source: [UAS 2030 ODLC](https://uas2030.github.io/docs/suas2025/object-detection-localisation-and-classification/).
- ICMTC (Maher team):
  - 27,812 images, 640 px, YOLO11n on a Raspberry Pi 5 via ncnn.
  - imgsz 1280 roughly 2.4x slower than 640 on Mac (32 vs 13 ms).
  - Gains from 1280 only for some classes.
  - Source: [resolution report](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/training/resolution_and_remap_report.md); [README](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/README.md).
- ICMTC (zakariahmedd34): 2 m flag ≈ 6-10 px at 640 input, so the repo defaults to imgsz 1920. Its geolocation gates detections to ≤30° off-nadir and clusters with DBSCAN (12 m) plus a geometric median. Source: [uav-gps-localization](https://github.com/zakariahmedd34/uav-gps-localization).
- Tiling:
  - Ultralytics describes tiling as running the model over overlapping windows of the full-resolution frame and merging the results. Objects near seams can be detected twice and need merging or NMS. Source: [Ultralytics blog](https://www.ultralytics.com/blog/small-object-detection-image-tiling).
  - "The Power of Tiling for Small Object Detection" (2019, Jetson TX1/TX2, VisDrone) is the classic reference. Source: [ResearchGate](https://www.researchgate.net/publication/332547902_The_Power_of_Tiling_for_Small_Object_Detection).
- Jetson Orin + TensorRT FP16 is the usual deployment path for YOLOv9-v11 in 2025 comparisons. Source: [Even3 2025 paper](https://static.even3.com/anais/brahurbrasero2025_paper_3.9ea63eaa84804c93ac7b.pdf?v=639010982278771247).
- TargetGeneratorV2 targets 150-300 ft (about 45-90 m), roughly the ICMTC band. Source: [TargetGeneratorV2](https://github.com/dec4234/TargetGeneratorV2).

### Inferences
- Rough GSD arithmetic. This is my own calculation and the camera FOV is assumed, not sourced:
  - Assume a 4K frame (3840 px wide) with about 85° horizontal FOV. Ground width is about 2·h·tan(42.5°) ≈ 1.83·h.
  - At 100 m: about 4.8 cm/px, so a 2 m flag is about 42x21 px at native resolution and about 7x3.5 px after resizing the full frame to 640. This matches the "6-10 px" claim.
  - At 50 m: about 2.4 cm/px, so about 84x42 px native.
- Recommendation: the generator should parametrise flag pixel size from (altitude, sensor width, FOV, output tile size), not a fixed scale range. It should also produce training images as tiles (for example 640 or 1024 crops from virtual 4K frames), so that one consistent strategy is used for train and test: tiling or SAHI at inference, or high imgsz.
- Rooftop test scaling (inference):
  - The flag-size to altitude ratio of a 30x15 cm flag at 15 m (0.02) equals that of a 2 m flag at 100 m. Same camera and resolution gives the same pixel footprint, so the rooftop test is a valid stand-in for the top of the altitude band.
  - To represent 50 m (ratio 0.04), film the 30 cm flag from about 7.5 m.
  - The test does not reproduce haze, wind motion blur, H.265 compression at a given bitrate, large-flag fabric wrinkles or desert glare. Printed paper colour (CMYK gamut, gloss) also differs from dyed fabric.
- "Synthetic + some real" mixing is the consistent recommendation in adjacent literature ([arXiv 2603.28670](https://arxiv.org/html/2603.28670v1)). Even a few hundred real rooftop or field frames are best used for (a) the validation/test set and (b) fine-tuning, not mixed blindly. This is inference; no flag-specific ratio was found.

### Gaps
- No source gives measured recall versus altitude for flags. The FOV and GSD figures above are illustrative and must be replaced by the team's actual camera specs.
- No SUAS team in the retrieved material reported using SAHI or tiling at inference with numbers.

## Q3. Public aerial datasets of flags or flag-like markers, and open aerial imagery of desert/airfield ground at 2-15 cm/px usable as backgrounds (with licences)

### Takeaway
No public dataset of real aerial country flags at 50-100 m was found. Roboflow flag datasets are mostly ground-level photos or small competition sets (uavc), and Roboflow blocked automated access, so their contents could not be verified. Backgrounds: OpenAerialMap (CC-BY 4.0, but check each image), GeoNadir on AWS (CC-BY 4.0 orthomosaics), and USGS UAS orthoimagery of drylands (public domain) are the best-licensed sources found.

### Cited Findings
**Flag datasets (Roboflow Universe; pages returned HTTP 403 to automated fetch, so details are from search snippets only)**
- `uavc` workspace:
  - `flags_gamed_awy`: 132 images, updated about a year ago.
  - `flags_uavc`: 0 images.
  - A `uav2024-flags` citation entry (2024) exists. Contents, class list and licence unverified.
  - Source: [Roboflow uavc workspace](https://universe.roboflow.com/uavc); [uav2024-flags](https://universe.roboflow.com/uavc/uav2024-flags/dataset/).
- Other general flag datasets, mostly not aerial:
  - Example: "countries flags" by mohamed-hanafy.
  - Example: a "country flags" project with 7 classes (Canada, Germany, Malaysia...).
  - Sources: [countries-flags](https://universe.roboflow.com/mohamed-hanafy-k650m/countries-flags); [country-flags-9bb1h](https://universe.roboflow.com/flowers-oav0e/country-flags-9bb1h); [flags-poltc](https://universe.roboflow.com/theokala/flags-poltc).
- Maher team's private 196-class top-down flag set (27,812 images) is not public. Its backgrounds came from soccer-field, tarp, airport and desert Roboflow datasets. Source: [DATASET.md](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/training/DATASET.md).
- GitHub flag detectors (YOLOv3/v8/v11) are ground-level flag recognition, not aerial: for example [Ws-Syx/Flag-detection-Yolo_v3-PyTorch](https://github.com/Ws-Syx/Flag-detection-Yolo_v3-PyTorch) (2021) and [Muhammad-Muzammil-Shah PAK-PALESTINE YOLOv11](https://github.com/Muhammad-Muzammil-Shah/PAK-PALESTINE-FLAGE-DETECTION-Using-Yolov11) (2025).

**Background imagery**
- OpenAerialMap:
  - Drone imagery is usually 1-30 cm resolution.
  - The AWS registry says all imagery is CC-BY 4.0 (attribution to Open Imagery Network contributors). Google Earth Engine's catalogue says each image has its own licence in its metadata, so verify per image.
  - Coverage is uneven; desert coverage is not guaranteed.
  - Sources: [Bellingcat OAM guide](https://bellingcat.gitbook.io/toolkit/more/all-tools/openaerialmap); [AWS Open Data: OpenAerialMap](https://registry.opendata.aws/openaerialmap/); [GEE community catalog OAM](https://gee-community-catalog.org/projects/oam).
- GeoNadir FAIR data on AWS: drone RGB orthomosaics as Cloud-Optimized GeoTIFFs, CC-BY 4.0. Desert sites not confirmed. Source: [AWS registry GeoNadir](https://registry.opendata.aws/geonadir-fair-data/).
- USGS:
  - Moab, Utah dryland UAS natural-colour imagery (2023), US public domain. Source: [data.gov USGS UAS 2023](https://catalog.data.gov/dataset/natural-color-and-multispectral-aerial-imagery-collected-during-uncrewed-aircraft-sys-2023).
  - USGS EROS UAS orthoimagery archive and a Mojave Desert UAS orthomosaic (public domain). Sources: [USGS EROS UAS Orthoimagery](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-unmanned-aircraft-systems-uas-orthoimagery); [USGS Mojave orthomosaic image](https://www.usgs.gov/media/images/contours-overlaid-orthomosaic-mojave-desert-generated-uas-acquired-data).
- Desert waste-detection UAV set (DJI Mavic 2 Pro, 2,088 geotagged RGB images in desert-like terrain). Licence not shown; region and authors per the ASREN record. Source: [ASREN record](https://asren-aggregator.asren.net/Record/Manara2_c2a7666e932f0fd084a585a963977a89).
- Bardenas Reales (Spain) semi-desert drone images on Zenodo. Licence not shown in results. Source: [ICAERUS Zenodo datasets](https://github.com/ICAERUS-EU/Zenodo_Datasets).
- Voxel51 "Ariel Scans" (Hugging Face): desert search imagery. Code and annotations are MIT, but the imagery comes under informal permission from the original poster, so it is risky for redistribution. Source: [Voxel51 ariel_scans](https://docs.voxel51.com/dataset_zoo/datasets_hf/ariel_scans.html).
- A UC Davis/IGIS Mojave survey at 400 ft gave about 2.25 in/px (about 5.7 cm) RGB. This is project-specific, not an open listing. Source: search snippet via [UCANR](https://www.ucanr.edu/node/159981) (unverified detail).

### Inferences
- For 2-15 cm/px desert or airfield backgrounds:
  - Pull OAM and GeoNadir tiles over arid regions (Egypt, Gulf, Sahel, US Southwest), plus USGS UAS orthomosaics.
  - Resample to the target GSD for each simulated altitude (for example 2.4-4.8 cm/px for 50-100 m with a 4K camera).
  - Keep per-image licence and attribution metadata in the generator output.
  - Avoid satellite basemaps (Google, Bing) because of licence restrictions.
- Airfield-specific distractors (runway paint, taxiway signs, windsocks, tyre marks, tarmac/sand transitions) are worth harvesting from OAM airport mosaics as negative tiles.
- Roboflow uavc sets are small (around 132 images). Their best use is probably as a sanity-check real or near-real test set, if they turn out to be real aerial frames from a previous UAVC. They need manual verification and a licence check in the browser.

### Gaps
- Licences and contents of `uavc/uav2024-flags` and `uavc/flags_gamed_awy` could not be verified (Roboflow returned 403).
- No specific OAM listing of an Egyptian desert airfield with confirmed GSD was identified. The catalogue must be browsed manually.
- Common UAV datasets (VisDrone, Semantic Drone Dataset, UAVid, DOTA) were not checked for licence in this pass.

## Q4. ICMTC competition rules for the flag task (flag list, scoring, altitude) that would change the generator configuration

### Takeaway
No official rulebook was retrievable. The official UAVC site (uavc.conferences.ekb.eg) refused connections. Two independent ICMTC 2026 UAVC-9 team repos agree on the following (all team-reported, unofficial):
- 2 m x 1 m flags lying flat on a flat desert airfield
- flight at 50-100 m AGL
- 2 flags in the search area plus 1 bonus flag inside the geofence
- full localisation points within 20 m, partial credit up to 60 m
- flag identity worth 10 points
- submission of GPS coordinates plus a cropped image per flag on USB

One team says only four flags (Egypt, Iraq, Syria, Yemen) were "briefed".

### Cited Findings
- From [zakariahmedd34/uav-gps-localization](https://github.com/zakariahmedd34/uav-gps-localization) (2026, team doc, unofficial):
  - Flags are 2 m x 1 m, flat on the ground, no pole.
  - "Flat desert airfield, 50-100 m AGL".
  - Mission 1: "2 flags in the search area + 1 bonus flag" in the geofence.
  - Full points within 20 m error, partial credit up to 60 m.
  - Flag identity worth 10 points. That team plans a human reading the three crops and typing country names.
  - Submission: GPS plus a cropped image per flag on a USB drive.
- From the Maher-star-eng README (2026): the system is restricted to 4 "briefed" flags (Egypt, Iraq, Syria, Yemen), with class IDs 50/78/170/193 masked. Mission 2 is described as detection, identification, geolocation and payload-drop target selection. Source: [README](https://raw.githubusercontent.com/Maher-star-eng/ICMTC-2026/main/icmtc/README.md).
  - Note: the two repos use different mission numbering, "Mission 1" vs "Mission 2". The tasks may differ between the fixed-wing and multirotor tracks, or the teams may number them differently. UNRESOLVED.
- The official site describes the 9th ICMTC "Lt. General Ibrahim Selim Award" Unmanned Aerial Vehicle Challenge, sponsored by the Egyptian Armed Forces, with a concept stage and PDR deadline. Rules text was not accessible. Sources: [uavc.conferences.ekb.eg](https://uavc.conferences.ekb.eg); [ICMTC umsic site](https://umsic.conferences.ekb.eg/cnf/about).
- A Studocu document titled "ICMTC 2026 Multi Rotors Challenge: Competition Details & Guidelines" exists. Full text not visible. Source: [Studocu](https://www.studocu.com/row/document/assiut-university/mechanics-of-machine/icmtc-2026-multi-rotors-challenge-competition-details-guidelines/157117175).
- Syria's national flag: the interim constitution of 13 March 2025 (Art. 6) defines green-white-black with three red stars, replacing the 1980-2024 red-white-black flag with two green stars. It was raised at the UN on 25 April 2025. Sources: [FOTW Syria](https://crwflags.com/fotw/flags/sy.html); [Al Jazeera 2025-04-25](https://www.aljazeera.com/program/newsfeed/2025/4/25/syrias-new-three-starred-flag-raised-at-un); [Al-Monitor 2025-07](https://www.al-monitor.com/originals/2025/07/syria-unveils-gold-eagle-emblem-what-it-means).

### Inferences (generator configuration implications)
- **Flag list:**
  - If the briefed list really is Egypt, Iraq, Syria, Yemen (unverified; confirm from the team's official rulebook or brief), the generator should support a "mission subset" mode. That mode heavily oversamples these four classes and their hard negatives.
  - Keep a broader distractor set: other tricolours such as Sudan, UAE, Jordan, Palestine, Libya, Germany, plus generic red/white/black fabric. The other 190+ classes can act as distractors or an "other flag" class, so the model is not forced to pick one of the four for any rectangle.
  - Configure both Syrian variants, with the 2025 three-red-star flag as primary. The physical flags printed by organisers could be either. Ask the organisers.
- **Geometry:**
  - Fixed 2:1 aspect (2 m x 1 m), lying flat. Use near-nadir views (the other team gates at ≤30° off-nadir), so mostly mild perspective.
  - Fabric on sand will wrinkle, fold at the edges, cast thin shadows, and may be partly covered by sand or weighted with stones. Model these.
- **Scale:** sample altitudes uniformly over 50-100 m (or match the planned flight altitude), converted to pixels through the real camera intrinsics.
- **Orientation:** flags can lie at any heading, so sample a full 0-360° in-plane rotation, not just 90° steps. Also include mirrored placement, since a flag may be laid face-down or reversed. Whether organisers ever lay flags mirrored is UNVERIFIED.
- **Scoring:**
  - Localisation is worth more than ID: a 20 m tolerance with partial credit to 60 m, versus 10 points for identity.
  - So detection recall ("is there a flag here") matters most. Classification can be backed up by a human reading the cropped image.
  - Export high-quality crops. Consider a two-stage design: a generic "flag" detector trained on all flags, then a classifier on high-resolution crops. This is the same lesson as the Maher team: failures were misses, not mislabels.
- **Count:** 2-3 flags per mission, so false positives are costly when choosing the top-K. Include many flag-free desert and airfield frames and distractors.

### Gaps
- The official ICMTC 2026 (10th edition?) UAVC rulebook, the flag list, the exact scoring split, and the altitude limits could not be accessed. The ekb.eg site gave ECONNRESET and no PDF was indexed.
- Whether flags are fabric or printed vinyl, whether they are weighted, and whether the briefed list is published in advance or revealed on site: all unknown.
- Edition numbering is inconsistent: one site says "9th ICMTC UAVC", the repos say "UAVC-9" for 2026, and an ICMTC 2024 page says "8th ICMTC". Needs confirmation.
