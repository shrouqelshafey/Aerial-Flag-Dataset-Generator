# Closing the synthetic-to-real gap for cut-paste aerial flag detection (YOLO, 50-100 m UAV)

Scope: which techniques measurably reduce the synthetic-to-real gap for cut-and-paste / procedurally composited detection data, with numbers where found, and what a small Python/OpenCV team should add next. The generator already does metric scale, rotation, folds, contact shadows, per-photo camera matching (black level, haze, noise, blur), look transfer, negatives, class balancing and a split by background photo.

## 1. Cut-paste synthesis, domain randomization (DR) and structured DR: which blending modes and randomizations matter most?

### Takeaway
Mixing several blending modes beats any single mode (about +8 mAP over no blending in Cut, Paste and Learn, 2017). Occlusion/truncation and rotation are the largest single augmentation gains (about 4 to 10 AP). Distractors add about 2.5 to 3 AP. In rendered DR, randomized lighting and texture matter most (6.1 and 4.7 AP), and brightness/contrast jitter alone does almost nothing. For aerial/UAV synthetic data, texture randomization gave the largest reported jump (+17 mAP YOLO, 2021), and matching capture metadata (viewing angle, time of day) helps synthetic-only models a lot but adds little once real fine-tuning is done.

### Cited Findings
**Cut, Paste and Learn (Dwibedi et al., ICCV 2017), GMU Kitchen mAP, Table 1** — [ar5iv 1708.01642](https://ar5iv.labs.arxiv.org/html/1708.01642)
- Blending: no blending 65.9; Gaussian blur only 68.9; Poisson only 58.4 (worse than no blending); "All Blend" (each object rendered with all modes as separate samples) 72.4; All Blend + same-image variant 73.7. That is about +8 AP from mixing modes rather than picking one. Motion and box blur are NOT ablated in the paper, so any claim that "motion blur blending helps" is not backed by its tables.
- Augmentation ablations against the full model (73.7): without 2D rotation 69.7 (-4.0); without 3D rotation 68.3 (-5.4); without truncation 71.8 (-1.9); without occlusion 63.1 (-10.6); adding distractor objects 76.2 (+2.5).
- Final GMU results: real only 86.3; synthetic only 76.2 (vs 51.7 for an earlier synthesis baseline); synthetic + real 88.8.
- Scale of data: about 6,000 synthetic images from 1,548 background images (each background used about 4 times).
- The authors argue that patch-level realism, not global scene layout, is what matters for region-based detectors. Pasting at random positions is acceptable.
- The abstract says naive cut-paste creates pixel artifacts, and that the blend mixture trains the detector to ignore boundary artifacts — [arXiv 1708.01642](https://arxiv.org/abs/1708.01642)

**Tremblay et al., NVIDIA, CVPRW 2018, DR for car detection on KITTI** — [ar5iv 1804.06516](https://ar5iv.labs.arxiv.org/html/1804.06516)
- 100K DR images (up to 14 cars each). KITTI AP for Faster R-CNN: VKITTI (photo-realistic clone, 2.5K images) 79.7 vs DR 78.1. For R-FCN: 64.6 vs 71.5. For SSD: 36.1 vs 46.3.
- DR + fine-tuning on all 6,000 real images: AP 98.5, about 2.1 above real-only (about 96.4, derived).
- Ablation (baseline 73.7): fixed light instead of random lights 67.6 (-6.1); no texture 69.0 (-4.7); 4K vs 8K texture pool 71.5 (-2.2); no data augmentation 72.0 (-1.7); no flying distractors about 72.6 (-1.1); brightness/contrast augmentation off 73.6 (-0.1).
- Performance saturates at about 10K synthetic images with pretrained weights and about 50K without. COCO + DR 83.7 vs COCO alone 56.1. Freezing pretrained layers hurt by up to 13.5% (Faster R-CNN 66.4 frozen vs 78.1 full).

**Structured DR (Prakash et al., ICRA 2019)** — [arXiv 1810.10093](https://arxiv.org/abs/1810.10093)
- Places objects and distractors using context-aware, problem-specific distributions instead of uniform ones. The abstract says SDR beats VKITTI, Sim 200k, DR and BDD100K on KITTI easy/moderate/hard, and SDR + real KITTI beats real KITTI alone. Exact AP values were not in the abstract (see Gaps).

**Simple Copy-Paste (Ghiasi et al., CVPR 2021)** — [arXiv 2012.07177](https://arxiv.org/abs/2012.07177v2)
- Random placement with no context modelling plus large-scale jittering is "good enough". It gave +1.5 box AP on COCO at state of the art, and its largest gains were in the low-data regime.

**Aerial/UAV-specific evidence**
- Sim2Air (2021): texture randomization of synthetic aerial (UAV-to-UAV) data raised real-test mAP by 17 points for YOLO and 20 for Faster R-CNN on one real test set, and by 3.7 / 1.1 on a second. Rationale: it forces shape-based rather than texture-based features — [arXiv 2110.05145](https://arxiv.org/abs/2110.05145v2)
- Kiefer, Ott, Zell (ICPR 2022), DeepGTA-UAV, 40-90K synthetic 4K images per task, Table VII — [ar5iv 2112.12252](https://ar5iv.labs.arxiv.org/html/2112.12252)
  - Aligning camera angle (a ~10% subset, 3,954 of 40k images, matched to the real dataset's angles) raised synthetic-only mAP@50 on Cattle from 29.2 to 36.6. After real fine-tuning there was no change (85.8 both ways).
  - Time-of-day alignment raised synthetic-only SeaDronesSee from 10.5 to 14.6, and fine-tuned only from 60.3 to 60.5.
  - Graphics quality: low vs high rendering gave synthetic-only 8.4 vs 10.2 on VisDrone, and no difference after fine-tuning (45.1 vs 45.0).
- 2026 small-UAV detection study (lightweight YOLO-class nano model, ICEE 2026), mAP@50 — [alphaXiv 2604.19999](https://www.alphaxiv.org/abs/2604.19999.md)
  - Copy-paste with global distortions scored 61.8 / 85.2 / 84.2 / 52.9 on four datasets vs a no-augmentation baseline of 50.6 / 79.9 / 82.6 / 40.5.
  - Rough copy-paste with scale jitter fell below baseline on one dataset (78.3 vs 79.9).
  - Authors report the nano model overfits to the "unnatural edges" of pasted instances. Copy-paste also gave high precision but low recall (80.9% precision, 44.5 mAP on FDD).
  - Mosaic + HSV pixel-level augmentation did best (68.0 on DvB), with Mosaic disabled for the last 10 epochs.
- UAV Austin "Hawk-Eye" (an SUAS team) pastes generated shape targets onto aerial images. They note that edge artifacts and predictable colours let the model learn to spot fakes quickly, and they plan to add real target photos from flight tests or rooftops — [Hawk-Eye docs](https://uavaustin.github.io/hawk-eye/data_generation/overview.html)

### Inferences
- The generator's single "camera matching + small Gaussian blur" path is roughly the "Gaussian only" row (68.9 vs 72.4 for All Blend). The cheapest likely win is to render each flag with a random choice among: hard paste, Gaussian-feathered alpha (several sigma), motion-blur-matched alpha (UAV forward motion and vibration), and a small mask erosion/dilation jitter. Avoid relying on Poisson (`cv2.seamlessClone`) as the sole mode: it was the worst single mode and shifts the object's colours toward the background, which is bad for flag identity. Use it only as a minority mode, if at all.
- Sim2Air's texture-randomization result does not transfer directly, because flag identity is colour/pattern-based, so the texture IS the class. The analogous safe randomizations are:
  - illumination (sun angle, specular sheen on fabric)
  - per-flag white balance and saturation jitter within print/fabric tolerances
  - print fading
  - dirt or sand on the sheet
  - background texture randomization of the airfield ground
- Tremblay's ablation suggests global brightness/contrast jitter adds almost nothing. Lighting that varies spatially (shadows across the flag, sun-glint) and variety of textures and backgrounds are what count.
- Occlusion/truncation was the largest Dwibedi factor (-10.6 AP). For flags on the ground this maps to:
  - flags partly cut by the image edge (truncation; YOLO tiles also cut flags)
  - partial cover by sand, debris or shadows of poles/people/aircraft
  - folds that hide part of the pattern
- Kiefer's metadata result implies: sample altitude, camera pitch (nadir vs oblique), sun elevation and time of day from the distribution expected at the competition. This matters most while there is no real data; once real fine-tuning exists, it matters less.

### Gaps
- Exact SDR AP tables (KITTI easy/moderate/hard) were not retrieved, only the qualitative abstract claims.
- No source found that ablates motion-blur blending for pasted objects specifically.
- No source found that ablates the fabric-fold or contact-shadow effects for ground targets.
- Dvornik et al. 2018 ("Modeling Visual Context is Key...", arXiv 1807.07428) reportedly found that random placement can hurt in some settings while context-aware placement helps. This was not verified in this session; check before citing.

## 2. How much real data is needed and how to mix it

### Takeaway
Across driving, indoor and aerial domains, synthetic-only lags real-only by a large margin. Synthetic pretraining (or mixing) plus a small real set (about 5-20% of a "full" real set, or tens of images) recovers most or all of the real-only performance and often beats it. Pretrain-then-fine-tune is the most commonly recommended schedule, and some reports explicitly prefer it over blended batches.

### Cited Findings
- Cut, Paste and Learn (2017), cross-domain Active Vision test — [ar5iv 1708.01642](https://ar5iv.labs.arxiv.org/html/1708.01642)
  - Real only 41.9; synthetic only 36.5; synthetic + real 51.1.
  - 10% real alone 15.8 vs 10% real + synthetic 43.2, which beats 100% real-only (41.9).
  - 40% real 38.2 → 50.8 with synthetic.
- RarePlanes (Shermeyer et al., WACV 2021, satellite aircraft): synthetic-only was not enough. Fine-tuning with ~10% of the real training data reached roughly 91-96% of real-only mAP, though the false-positive rate stayed slightly higher than with full real training — [arXiv 2006.02963](https://arxiv.org/pdf/2006.02963); [WACV paper](https://openaccess.thecvf.com/content/WACV2021/papers/Shermeyer_RarePlanes_Synthetic_Data_Takes_Flight_WACV_2021_paper.pdf)
- RarePlanes follow-up blog (2020): recommends isolating real and synthetic into separate training stages rather than blending. The gains were largest for the rarest classes (≤10 real examples) — [DownLinQ part 2](https://medium.com/the-downlinq/rareplanes-exploring-the-value-of-synthetic-data-part-2-2198c334b383)
- Kiefer et al. (ICPR 2022), mAP@50, synthetic-only / real-only / synthetic pretrain + real fine-tune — [ar5iv 2112.12252](https://ar5iv.labs.arxiv.org/html/2112.12252)
  - YOLOv5m6: Cattle 64.2 / 88.8 / 86.9 (slightly worse than real-only); SeaDronesSee 10.5 / 55.8 / 60.3; VisDrone 10.2 / 43.9 / 45.0.
  - Faster R-CNN: SeaDronesSee 14.6 / 54.7 / 59.0.
  - Pretraining gains shrink as the real set grows, and preliminary experiments found pretrain + fine-tune superior to combined (mixed) training.
- Burdorf et al. (2022): mixed datasets with 5-20% real data cut the need for real data by up to 70% without losing detection performance. They did not confirm the fine-tuning advantage reported by Nowruzi et al. for a similar task, so fine-tune vs mix is not settled — [arXiv 2202.00632](https://arxiv.org/abs/2202.00632v1)
- Nowruzi et al. (2019): systematic study of hybrid sets with low real ratios. They found fine-tuning on a small real set after synthetic pretraining generally beats mixing — [arXiv 1907.07061](https://arxiv.org/abs/1907.07061v1)
- Tremblay et al. (2018): DR + fine-tune on 6,000 real KITTI images reached AP 98.5 vs about 96.4 real-only — [ar5iv 1804.06516](https://ar5iv.labs.arxiv.org/html/1804.06516)
- A UAV YOLO study (CEUR-WS vol. 4155, about 2025): fine-tuning a synthetic-pretrained model on 50 real images matched a model needing 400 real images. Real-data baseline mAP@0.5 was 0.672 — [CEUR-WS paper03](https://ceur-ws.org/Vol-4155/paper03.pdf)
- An SUAS team (IIT Madras) used three stages: synthetic only → synthetic + real mix → fine-tune on 90% real / 10% synthetic — [SUAS-IITM dataset notes](https://suas-iitm.vercel.app/resources/dataset-generation)

### Inferences
- Recommended schedule for this team:
  - (a) Train on synthetic from COCO-pretrained YOLO weights. Do not freeze the backbone (Tremblay: freezing cost up to 13.5%).
  - (b) Fine-tune on rooftop + any real drone frames, mixing in about 10-30% synthetic to avoid forgetting classes absent from the real set.
  - (c) If real data stays tiny (<100 frames), also try one run with real frames oversampled into the synthetic set and compare.
- Expect that even 50-200 labelled real frames per condition will matter more than further generator polish. Kiefer's results show synthetic-side improvements largely wash out after fine-tuning.
- Real data that covers all classes is the bottleneck. RarePlanes found the biggest synthetic gains on the rarest classes, so synthetic data is most valuable for flags the team cannot print or film.
- Don't overtrain on synthetic. Tremblay saw saturation at about 10K images with pretrained weights. Diversity of backgrounds probably matters more than count (Dwibedi used ~1.5K backgrounds × 4).

### Gaps
- No source gives a ratio specific to tiny (45-95 px) flat ground targets.
- Nowruzi's exact numeric thresholds were not retrieved in this session.
- No Ultralytics documentation or GitHub discussion on synthetic/real mixing was retrieved. Ultralytics has no official sim2real guide that I could confirm.

## 3. Image harmonization, photometric matching and learned refinement (CycleGAN, SimGAN, diffusion img2img)

### Takeaway
Photometric and metadata matching helps synthetic-only performance but has little effect after real fine-tuning. GAN and diffusion translation can close appearance gaps, but they risk label drift (hallucinated or removed objects, shifted colours). For colour-identity classes like flags this risk is serious. Evidence for these methods in aerial detection is mostly qualitative or not retrievable here.

### Cited Findings
- Kiefer et al. (2022): better rendering quality and time-of-day alignment raised synthetic-only mAP (8.4→10.2; 10.5→14.6) but had ≈0 effect after real fine-tuning (45.1 vs 45.0; 60.3→60.5) — [ar5iv 2112.12252](https://ar5iv.labs.arxiv.org/html/2112.12252)
- Tremblay et al. (2018): VKITTI's photorealism did not beat cheap DR with Faster R-CNN (79.7 vs 78.1), and DR beat it on R-FCN and SSD — [ar5iv 1804.06516](https://ar5iv.labs.arxiv.org/html/1804.06516)
- Liu, Liu, Luo (2020): used CycleGAN to translate synthetic remote-sensing images toward real, using unlabelled real images, then fine-tuned on limited real data (NWPU VHR-10, UCAS-AOD, DIOR). The abstract claims it augments limited real data, but the numbers were not retrieved — [arXiv 2006.05015](https://arxiv.org/abs/2006.05015)
- RGB/IR drone detection (2022): CycleGAN applied to AirSim renders before YOLOv4 training. No isolated before/after mAP was found — [arXiv 2201.03786](https://arxiv.org/pdf/2201.03786.pdf)
- Diffusion translation for UAV human detection (CFHA, 2025): a three-stage diffusion pipeline that keeps the original synthetic labels, with an explicit "hallucination removal" stage. This shows label preservation needs special handling — [arXiv 2512.13869](https://arxiv.org/html/2512.13869)
- A 2025-2026 photorealism-enhancement paper reports that diffusion methods often hallucinate even with depth/edge control. Outputs can stop matching the ground-truth boxes, so the authors check label fidelity with a pretrained segmenter — [arXiv 2605.02291](https://arxiv.org/pdf/2605.02291) (venue/date not verified)
- A 2025 study of conditioning synthetic data (prompt vs layout) found that naively sampled diffusion data can be worth less than the same amount of real data — [arXiv 2507.02217](https://arxiv.org/html/2507.02217v1)
- SimGAN (Shrivastava et al., CVPR 2017): refiner GAN with a self-regularization L1 loss so annotations stay valid. Demonstrated on gaze and hand pose, not on detection — [arXiv 1612.07828](https://arxiv.org/abs/1612.07828)
- Hawk-Eye/SUAS: predictable colours and edge artifacts are learnable "fake" cues — [Hawk-Eye docs](https://uavaustin.github.io/hawk-eye/data_generation/overview.html)

### Inferences
- The existing per-photo camera matching and look transfer is the right, label-safe level of harmonization. Two cheap OpenCV/numpy additions are worth testing:
  - Fourier Domain Adaptation-style low-frequency amplitude swap, or simple histogram/Reinhard colour matching, applied to the whole composite rather than the flag alone.
  - Re-applying the camera pipeline to the composite after pasting: demosaic-like softening, JPEG/H.264 re-compression at drone bitrates, sharpening halos, chromatic aberration. Then flag edges carry the same artifacts as the background.
- Do not apply CycleGAN or diffusion img2img to whole frames without a label-fidelity check. At 45-95 px the flags occupy a tiny area, and a generative model can easily recolour stripes, change one flag into another, or erase it. If tried:
  - use low denoising strength, apply it only to background tiles, and re-paste the flags afterwards
  - or use it to generate new backgrounds and distractors, never the labelled flags
- Expect harmonization gains to mostly disappear once rooftop/real fine-tuning exists (Kiefer). Prioritize real data over learned refinement.

### Gaps
- No quantitative mAP result was retrieved for CycleGAN or diffusion refinement on small aerial targets.
- No evidence was found on whether FDA or histogram matching helps YOLO on UAV data specifically.

## 4. Hard negatives and distractors

### Takeaway
Pasting unlabelled distractors gives modest but consistent AP gains: +2.5 AP in cut-paste (2017) and about +1.1 AP from flying distractors in DR (2018). Papers attribute this partly to stopping the detector from latching onto paste-boundary artifacts. No study was found that isolates the false-positive reduction itself. Copy-paste without distractors can make a detector over-conservative, or let it key on paste edges.

### Cited Findings
- Cut, Paste and Learn: adding distractor objects 73.7 → 76.2 AP. The authors say distractors also keep the detector from latching onto boundary artifacts — [ar5iv 1708.01642](https://ar5iv.labs.arxiv.org/html/1708.01642)
- Tremblay et al.: removing flying distractors cost about 1.1 AP (derived from the ablation table) — [ar5iv 1804.06516](https://ar5iv.labs.arxiv.org/html/1804.06516)
- Rajpura et al. (2017): a detector trained on synthetic data without distractors produced false positives on look-alike vegetables. The authors link this to the missing background distractors, but did not test it directly — [arXiv 1706.06782](https://arxiv.org/pdf/1706.06782)
- 3D-rendered face detection (2018): false positives were the main limit and came from confusing look-alike objects that the renders did not cover — [arXiv 1812.07363](https://arxiv.org/pdf/1812.07363)
- SimD3 (2026): a synthetic drone dataset that explicitly models bird distractors for robust detection — [arXiv 2601.14742](https://arxiv.org/html/2601.14742v1)
- Small-UAV copy-paste study (2026): high precision with low recall when copy-paste was used — [alphaXiv 2604.19999](https://www.alphaxiv.org/abs/2604.19999.md)

### Inferences
- Paste unlabelled look-alikes with the same blending pipeline as the flags. Then "has a pasted edge" stops being a positive cue. Candidates:
  - plain-coloured tarps and sheets, striped mats, runway markings
  - vehicles, solar panels, rectangular roofs, debris
  - printed rectangles in random colours
  - flags of countries NOT in the class list (very hard negatives)
  - partially hidden or oversized/undersized flags left unlabelled? Avoid this one: it creates label noise.
- Keep the existing empty images and add real-background-only hard negatives. Mine false positives from inference on unlabelled real drone footage and add those crops or frames as background images (YOLO treats images with an empty label file as negatives).
- Measure false positives per image on empty real frames separately from mAP. The literature reports AP, not FP rate.

### Gaps
- No controlled study was found that reports false-positive rates with and without distractors for aerial cut-paste data.
- No Ultralytics-specific recommendation on the background-image ratio was retrieved. The commonly cited ~0-10% background images figure comes from Ultralytics tips but was not verified this session.

## 5. Validation strategy without real target data

### Takeaway
The literature consistently measures the gap as synthetic-only vs real-only vs synthetic+real on a held-out REAL test set, plus per-component ablations (drop one randomization at a time). Even a small real test set is essential. Synthetic validation mAP is not a proxy for real performance (e.g. Kiefer: VisDrone synthetic-only 10.2 vs real-only 43.9).

### Cited Findings
- Per-factor ablation (remove one augmentation/randomization, retrain, test on real) is the standard protocol in Cut, Paste and Learn (Table 1) and Tremblay et al. — [ar5iv 1708.01642](https://ar5iv.labs.arxiv.org/html/1708.01642); [ar5iv 1804.06516](https://ar5iv.labs.arxiv.org/html/1804.06516)
- Cross-domain testing (train on one real set, test on another) exposed much larger benefits of synthetic data (Active Vision: +9.2 AP from adding synthetic) than in-domain testing (GMU: +2.5) — [ar5iv 1708.01642](https://ar5iv.labs.arxiv.org/html/1708.01642)
- Sim2Air reports that the harder the generalization test set, the larger the benefit of the shape-biased randomization. Gains differed widely across two real test sets (+17 vs +3.7 mAP) — [arXiv 2110.05145](https://arxiv.org/abs/2110.05145v2)
- Kiefer et al.: synthetic-side changes visible in synthetic-only evaluation vanished after fine-tuning, so evaluate both regimes — [ar5iv 2112.12252](https://ar5iv.labs.arxiv.org/html/2112.12252)
- SUAS teams plan or use rooftop and flight-test photos as the real check — [Hawk-Eye docs](https://uavaustin.github.io/hawk-eye/data_generation/overview.html); [SUAS-IITM](https://suas-iitm.vercel.app/resources/dataset-generation)

### Inferences
- Build three real test sets and never train on them:
  - (A) rooftop small-flag footage, scaled so the pixel lengths match 45-95 px. It tests flag appearance and printing but not desert background, altitude haze or the drone camera.
  - (B) real drone frames over any sandy/desert-like terrain with printed flags, even a few sheets, flown at 50-100 m with the competition camera.
  - (C) empty real aerial frames, to measure false positives per image.
- Report mAP50, per-class recall and FP/image on each.
- Run per-layer ablations against (A)/(B): turn off folds, shadows, camera matching, look transfer, distractors, or the blend mixture one at a time. Use a fixed seed and the same number of images, and ideally 2-3 seeds per setting, because small test sets are noisy.
- Keep a "real-only" baseline (trained on part of the rooftop/real data) and a "synthetic + real fine-tune" model. The gap between synthetic-only and these is the number to drive down.
- Rooftop footage is a proxy with its own gap: different lens, altitude, atmosphere and background. Treat gains on (A) as necessary but not sufficient, and prioritize getting even 20-50 frames of type (B).
- Cheap gap diagnostics with no labels:
  - compare detector confidence histograms on synthetic vs real frames
  - FID/KID or simple colour, noise and sharpness statistics on flag crops from synthetic vs real
  - eyeball a grid of real vs synthetic crops (a "spot the fake" test: if a human can tell, a CNN probably can too)

### Gaps
- No published protocol was found for validating with a rooftop proxy as a stand-in for altitude data.
- No study quantifies how many real test images are needed for a stable mAP estimate at these object sizes.
- No AUVSI SUAS team report with quantified synthetic-to-real mAP results was found. The team pages found describe pipelines only.
