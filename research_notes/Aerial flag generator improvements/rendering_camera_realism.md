# Rendering and Camera Realism for Synthetic Aerial Flag Images (NumPy/OpenCV)

Scope: making pasted 2 m x 1 m printed polyester flags on desert sand / tarmac, seen at 50-100 m by a SIYI A8 mini class gimbal camera, look physically plausible; deciding which effects are worth implementing in NumPy/OpenCV. The generator already has supersampling, sine+noise folds with shading, small perspective tilt, contact shadow, alpha feathering, matched black level / noise / Gaussian blur on the flag, and a global colour/contrast look transfer.

Notation used below: `h` = altitude (m), `GSD` = ground sample distance (m/px), `v` = ground speed (m/s), `t_exp` = exposure time (s).

---

## 0. Camera geometry baseline (needed to judge which effects survive at this pixel size)

### Takeaway
At 4K (4096 px wide) with HFOV 81 deg, the nadir GSD is about 2.1 cm/px at 50 m and 4.2 cm/px at 100 m, so a 2 m flag spans about 96 px down to 48 px. The 45-95 px flag length in the brief matches this. The "5-10 cm/px" figure in the brief does not. It would only hold for frames downscaled to about 2K or for strongly oblique views. Each pixel subtends about 0.02 deg, so the gimbal's stated +/-0.01 deg vibration is about +/-0.5 px.

### Cited Findings
- SIYI A8 mini: Sony 1/1.7-inch, 8 MP effective sensor; 21 mm equivalent fixed focal length with 6x digital zoom; F2.8; DFOV 93 deg, HFOV 81 deg; video 4096x2160 / 2560x1440 / 1920x1080 / 1280x720 at 25 fps; **H.265 only, 20 Mbps for 4K/2K, 15 Mbps for 1080p/720p**; photos are JPG; gimbal "angular vibration range +/-0.01 deg" — [SIYI A8 mini spec page](https://siyi.biz/en/product/tri-axis-single-camera-gimbal/a8-mini/spec/)
- SIYI's FAQ lists the A8 mini shutter as a **rolling shutter**, and the exact Sony sensor model is not published. An ArduPilot forum user asked SIYI for the model and found none — [SIYI A8 mini FAQ PDF](https://siyi.biz/siyi_file/FAQ/20260327/A8%20mini%20-%20FAQ.pdf); [ArduPilot forum: sensor model used in SIYI A8](https://discuss.ardupilot.org/t/sensor-model-used-in-siyi-a8/102869)

### Inferences
- Footprint width at nadir = 2 h tan(40.5 deg) = 1.708 h. At 4096 px: GSD(50 m) = 2.09 cm/px, GSD(75 m) = 3.13 cm/px, GSD(100 m) = 4.17 cm/px. The flag is therefore 96 / 64 / 48 px long and 48 / 32 / 24 px wide. Oblique gimbal pitch makes the far side coarser.
- Angular pixel size (IFOV) is about 81 deg / 4096 = 0.0198 deg/px. The gimbal's +/-0.01 deg vibration spec therefore means about +/-0.5 px frame-to-frame jitter at any altitude. That is negligible as blur within a short exposure.
- Bitrate budget: 20 Mbit/s / 25 fps = 0.8 Mbit/frame. Over 4096x2160 = 8.85 Mpx, that is about **0.09 bits/pixel**. This is very low for natural texture. Fine sand texture and the small high-contrast details inside a 50 px flag (stars, coats of arms, thin stripes) will be smoothed or smeared on video frames. This makes compression one of the most important effects to model when frames are pulled from video.
- Diffraction: Airy disk diameter = 2.44 x lambda x N = 2.44 x 0.55 um x 2.8 = 3.8 um. A 1/1.7-inch sensor is about 7.6 mm wide, so at about 4096 px the pitch is about 1.9 um. The Airy disk therefore covers about 2 px. Diffraction plus lens aberration plus ISP denoising gives an effective PSF Gaussian sigma of roughly 0.6-1.2 px. That supports the existing Gaussian blur step but suggests sampling sigma in that range rather than a fixed value. Pixel pitch is estimated from the nominal sensor format, not from a datasheet.

### Gaps
- No published rolling-shutter readout time, pixel pitch or MTF chart was found for the A8 mini sensor.
- No independent lens sharpness or chromatic aberration test of the A8 mini was found.

---

## 1. Fabric appearance at about 2-4 cm/px (brief says 5-10 cm/px): fading, dye differences, sheen, wrinkles vs taut

### Takeaway
At 2-4 cm/px the weave is invisible. Four things still show at 50 px: (1) overall colour and chroma shifts from fading and printer differences, (2) large folds and creases (10 cm or more) as shading bands, (3) occasional specular glints on fold facets for satin-like polyester, and (4) dust, dirt and the reverse-side print where a corner flips over. Colour variation, packaging creases and dust are the cheapest high-value additions. A full specular sheen model is medium priority.

### Cited Findings
- Dye-sublimation inks on polyester generally have **lower light fastness than selected acid dyes on nylon**, and "you can expect the shortest life outdoors from dye sublimation on polyester." UV-absorber treatment and a tighter weave improve outdoor fastness — [Specialty Fabrics Review / textiles.org: Creating images built to last](https://www.textiles.org/?p=1458)
- A flag retailer (vendor-grade claim) says a nylon flag flown daily in full sun typically shows visible fading within 60-90 days, and describes polyester as the more fade-resistant choice. Wet/dry cycles accelerate fading. This contradicts the textile source above on sublimated polyester specifically. Treat the vendor claim as marketing — [Colonial Flag: Why Flags Fade](https://www.colonialflag.com/blogs/blog/why-flags-fade-and-how-to-slow-it-down); contradicted by [textiles.org](https://www.textiles.org/?p=1458)
- Measured cloth BRDFs (including **polyester charmeuse**) show appearance as a mix of diffuse reflectance, specular reflectance, shadowing/masking, and grazing-angle sheen. Microcylinder models of fibres reproduce this — [Sadeghi et al., microcylinder cloth model (UCSD)](https://datarepo.eng.ucsd.edu/visiondata/obisker/public_html/cloth/microcylinder-sg13.pdf)
- Woven-cloth specular reflection arises from specular scattering off fibres in twisted yarns. It is anisotropic and spatially varying — [Irawan & Marschner, Specular Reflection from Woven Cloth](https://discmaster.textfiles.com/file/16677/SIGGRAPH%202012%20-%20Disc%201.iso/content/papers/034a-irawan.pdf)
- Recent neural cloth reflectance work models fabric as an anisotropic specular term plus a sheen lobe that adds the "characteristic fabric edge glow" — [RoboCloth (arXiv 2610.21090)](https://arxiv.org/pdf/2610.21090)

### Inferences
- **What is visible at 50 px.** Yarn and weave structure (sub-mm) and the anisotropic specular texture average out within one pixel. What remains is a per-facet BRDF: a mostly Lambertian albedo plus a broad, weak specular lobe. Sheen matters only on fold facets whose normal bisects the sun and camera directions. Looking down at nadir with the sun at elevation e, a flat flag is in mirror geometry only when e is near 90 deg (sun near zenith). Tilted fold facets can glint at other sun angles.
- **Colour fading / print variation (high value, cheap).** Work in Lab (`cv2.cvtColor(..., cv2.COLOR_BGR2LAB)` on float32):
  - Fading: `C* <- C* * U(0.65, 1.0)`, `L* <- L* + U(0, 8)` (washed-out look). Optional small hue drift `h <- h + N(0, 4 deg)`. Fading is often uneven, so apply a low-frequency mask (Gaussian-blurred noise, sigma about 20-40 % of flag size) and fade the exposed or upper side more.
  - Printer / dye differences: per-flag global offsets `a* += N(0,3)`, `b* += N(0,3)`, saturation scale `U(0.85, 1.1)`, and occasional ink-limit clipping (reduce the darkest blacks to L* about 15-25, because sublimated polyester rarely gives deep blacks). White areas: `b* += U(0, 5)` for yellowed or dusty white. These magnitudes are judgement values. No measured distribution was found.
  - Clamp back into the sRGB gamut after the shifts.
- **Wrinkles vs taut.** Wrinkles with wavelength under about 3-4 px (under about 10 cm at 3 cm/px) disappear after PSF and compression, so do not spend effort there. They also add aliasing when not supersampled. Two features are visible: (a) **packaging creases**. Flags are shipped folded, so a regular grid of straight crease lines at 1/2 and 1/4 of length and width gives thin bright/dark line pairs. This is cheap to add as displacement plus shading along straight lines. (b) **Large drapes** over uneven sand at 20-60 cm wavelength, which the existing sine+noise model already covers. Taut flags (pinned with stakes or stones) should exist too, with almost zero fold amplitude, so the detector does not learn "folds = flag".
- **Specular sheen (medium value).** Using the existing fold normal map `n`, the sun direction `L` and the view direction `V` (about (0,0,1) at nadir, or from the gimbal pitch), compute `H = normalize(L+V)`, `spec = ks * max(0, n·H)^p` with `p` in U(5, 30) (broad lobe; polyester satin is glossier, spun or knitted polyester is matte) and `ks` in U(0, 0.15) (often 0). Add it as a white-ish term before the ISP chain: `I = albedo*(ka + kd*max(0,n·L)) + spec*sun_color`. Clip highlights to the background's white level, because sensor saturation is common on bright sand.
- **Dust / dirt (high value in desert).** `flag = lerp(flag, sand_rgb, d*M)`, where `sand_rgb` is the median background colour around the paste location, `d` is in U(0, 0.35), and `M` is low-frequency noise that is stronger near edges (distance transform from the alpha edge: `M *= exp(-dist/edge_scale)` plus a uniform base). This also matches flag contrast to the hazy, dusty scene.
- **Reverse side.** Single-sided dye-sublimation flags typically show a mirrored, slightly lighter and lower-saturation image on the back because the dye penetrates the fabric. This is common industry knowledge but no source was retrieved. Use it for flipped-over corners: mirror the texture, `C* *= 0.7-0.9`, `L* += 5`.

### Gaps
- No controlled lightfastness data (Delta-E vs days of desert UV exposure) was found for dye-sublimated polyester flags, so fading magnitudes are judgement-based.
- No measured BRDF parameters were found specifically for printed flag-grade (knitted or spun, 110-120 g/m2) polyester, only for charmeuse. The `ks` and `p` ranges above are guesses.
- Print-through (reverse side) percentage for single-sided flags is unsourced.

---

## 2. Lighting consistency: sun direction, cast shadows, shading match, colour temperature

### Takeaway
A flag lying flat casts almost no shadow itself. What makes it look pasted is (a) fold shading lit from a direction that disagrees with the scene shadows, (b) a flag pasted over a background shadow region (from a building, vehicle, person or the drone) that is not darkened and blue-shifted, and (c) a flag whose brightness and colour do not follow local illumination such as vignetting, cloud shadow or exposure gradients. The cheapest robust fix is to annotate or estimate one sun azimuth (and roughly elevation) per background photo, drive fold shading, lifted-corner shadows and specular from it, and modulate the flag with a local shading/shadow map taken from the background.

### Cited Findings
- Sun azimuth can be estimated from a single image using the angle between an object's vertical edge and its shadow line, measured from the image Y axis. In aerial imagery, the angle of the shadow line (from footprint corner to shadow end) relative to the image Y direction gives the sun azimuth — [USPTO 8437554: extracting 3D object information from a single image](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8437554)
- A physics-based approach for aerial images combines detected shadows with capture date/time and GPS (sun position) and a sun-and-sky irradiance model. It estimates illumination from lit/shadow point pairs that share the same albedo — [ISPRS Annals V-2-2022-23](https://isprs-annals.copernicus.org/articles/V-2-2022/23/2022/isprs-annals-V-2-2022-23-2022.pdf)
- No annotated dataset existed for aerial shadow detection or removal. Ufuktepe et al. generate shadow masks automatically from 3D point clouds plus known solar angles — [Ufuktepe et al., ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/papers/Ufuktepe_Learning-Based_Shadow_Detection_in_Aerial_Imagery_Using_Automatic_Training_Supervision_ICCVW_2021_paper.pdf)
- Metadata-provided sun azimuth can be inaccurate. One cloud/shadow method searches over a small range around the metadata azimuth to correct it — [USPTO 11126886: Cloud detection on remote sensing imagery](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11126886)
- Daylight spectra measured in Granada (about 2,600 spectra) span CCT from **3758 K to 34,573 K**. Solar elevation, site altitude, atmospheric conditions, pollution and detector orientation all change daylight colour — [Hernández-Andrés et al., JOSA A 2001](https://colorimaginglab.ugr.es/pages/pdfs/josaa_2001_412/!)
- Natural-light CCT measured worldwide varies from about 3000 K to about 10^6 K, and illumination chromaticity varies systematically with solar elevation — [Hernández-Andrés / UGR JOSA A 2016 preprint](https://digibug.ugr.es/bitstream/10481/47128/1/2016-JOSAA-Colorimetric-PREPRINT.pdf); [Sci. Rep. figure: chromaticity vs solar elevation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4895134/figure/f3)
- In a diffusion-based UAV sim-to-real study, the appearance gap (style, lighting) is distinguished from the content gap (layout, object types) — [CFHA, arXiv 2512.13869](https://arxiv.org/html/2512.13869)

### Inferences
- **Sun direction per background (high value, cheap).** Options, from most to least reliable per effort:
  1. If EXIF/SRT has time + GPS + camera yaw, compute solar azimuth and elevation with a solar-position routine (for example `pysolar` or an NREL SPA port, a few lines of NumPy). Convert to image coordinates using the gimbal yaw. Refine with the shadow check below, because the metadata can be off.
  2. Manually click one object-to-shadow-tip vector per background image (about 5 s per image), store azimuth and shadow length / object height for elevation, and cache it in a JSON sidecar. For a dataset of a few hundred backgrounds this is the most reliable option.
  3. Automatic: dark-region segmentation (for example L* below 0.6 x local median and b* lower than its surround) followed by the dominant orientation of the shadow blobs' long axes (`cv2.findContours` + `cv2.fitEllipse` or a PCA of the edge gradient directions). This is error-prone on tarmac with dark patches.
- **Use the sun vector everywhere:**
  - Fold shading: replace any fixed-direction shading with `shade = ka + kd*max(0, n·L)`, where `L = (cos e sin az, cos e cos az, sin e)` in the image frame, and add the matching self-shadow on the far side of tall folds (cast along `-L_xy` by `height/tan(e)` px, computed with a simple 2D heightfield ray-march over the supersampled fold height map).
  - Lifted corner or edge: shadow polygon = the lifted region's footprint projected along the azimuth by `z/tan(e)/GSD` px, darkened by the scene shadow ratio (below), with a soft edge (penumbra about `z * 0.0093 / GSD` px, since the sun's angular diameter is about 0.53 deg).
  - Contact shadow: keep it, but make it anisotropic (offset slightly away from the sun) and very weak for a flat sheet.
- **Background shadows falling on the flag (high value).** If the flag footprint overlaps a shadowed background area, multiply the flag by the per-pixel ratio `r = bg_shadowed / bg_lit`, estimated per channel from the background itself: median of shadow pixels divided by median of nearby lit pixels of the same surface. On sand this is typically about 0.25-0.5 in luminance and bluer, because only skylight reaches it. This is an expectation, not a sourced measurement. The shadow edge on the flag must be continuous with the background shadow edge. Use the background-derived mask, eroded and blurred by about 1 px. Alternatively, randomly synthesize shadows of poles, people, vehicles or the drone (polygon or blob masks projected along the sun azimuth) that cover both the flag and the surrounding ground, so the shadow crosses the flag boundary consistently.
- **Local illumination / vignetting / exposure gradient (high value, cheap).** The global look transfer cannot capture position-dependent brightness. Compute `S(x) = G_sigma(lum_bg)(x) / median(lum_bg)` with a large sigma (for example 3-5x the flag size), and multiply the flag's linear RGB by `S` at its location. This captures vignetting, cloud shadows and exposure falloff without an explicit vignetting model. Pitfall: use a robust low-pass (median filter on a downscaled image) so that the flag does not darken just because it sits next to a dark object.
- **Colour temperature.** The background photo already contains the scene's white balance (the camera AWB result). The flag must receive the same illuminant and AWB, not a "true" CCT. Practical method: treat the flag texture as reflectance, multiply by an illuminant estimated from the background (for example the white-patch or gray-world colour of sand in a neighbourhood, normalized against an assumed sand albedo), or more simply apply per-channel gains matching a known neutral surface (tarmac or concrete is roughly gray) near the paste. For random augmentation, perturb per-channel gains by about +/-5 % to mimic AWB errors. In low-sun images (golden hour) the lit parts get warmer and shadows much bluer. The sun/shadow ratio above handles this automatically if estimated per channel.
- **Pitfall:** do all shading multiplications in linear light (`(x/255)^2.2`, or the sRGB EOTF), then re-encode. Multiplying in gamma space makes shadows on the flag too dark relative to the background.

### Gaps
- No measured sunlit-to-shadow luminance ratio or chromaticity for desert sand was found. The 0.25-0.5 range is an expectation and should be measured from the user's own background photos.
- No measured CCT-vs-solar-elevation table specifically for desert sites was retrieved. The Granada data is a mid-latitude site.
- No accuracy figures were found for automatic sun-azimuth-from-shadow methods on drone imagery.

---

## 3. Camera / ISP effects for small drone gimbal cameras

### Takeaway
Ranked by expected impact for 50-100 px flags from an A8 mini class camera: (1) **compression**: H.265 at about 0.09 bpp on video frames, JPEG on stills; (2) **correct noise model and spatial correlation**: Poisson-Gaussian in linear space, then denoised or sharpened by the ISP, so the noise ends up spatially correlated; (3) **PSF + sharpening halos** with randomized sigma and an unsharp mask; (4) small motion blur from gimbal panning and vibration; (5) lateral chromatic aberration and demosaic artifacts (minor); (6) rolling shutter, which is negligible at flag scale. All effects should be applied in a crop around the flag, or to the whole composite, in ISP order. Applying them to the flag alone in arbitrary order produces a "different-camera" signature that a detector can learn.

### Cited Findings
- **Poisson-Gaussian / shot+read noise.** Brooks et al. invert the camera pipeline (sRGB -> inverse tone map -> inverse gamma -> inverse CCM -> inverse WB gains -> mosaic), add heteroscedastic noise, and reprocess. A simple additive model is unrealistic, and photon arrival is better modelled as shot plus read noise — [Brooks et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.pdf)
- Exact constants from the reference implementation: `variance = image*shot_noise + read_noise`; `log(shot_noise) ~ U(log 1e-4, log 0.012)`; `log(read_noise) = 2.18*log(shot_noise) + 1.20 + N(0, 0.26)`; inverse tone map `x = 0.5 - sin(asin(1-2y)/3)`; gamma expansion `x^2.2`; WB gains `red ~ U(1.9, 2.4)`, `blue ~ U(1.5, 1.9)`, `rgb_gain = 1/N(0.8, 0.1)`; CCM sampled as a random convex combination of four camera XYZ->cam matrices with rows normalized — [timothybrooks/unprocessing, unprocess.py](https://raw.githubusercontent.com/timothybrooks/unprocessing/master/unprocess.py); [repo](https://github.com/timothybrooks/unprocessing)
- **Physically based sensor augmentation** (Carlson et al.): chromatic aberration as per-channel affine scale + translation (`x' = S x + t_x`, with S scaling green for longitudinal and translations for lateral CA); Gaussian out-of-focus blur; exposure via inverting and shifting a sigmoid response `I = 255/(1+e^{-A S})`; Poisson-Gaussian noise sampled on a GBRG Bayer pattern with bilinear demosaicing; colour shifts as translations in CIELAB. Parameter bounds were "chosen experimentally" and are not published in the text. Gains on KITTI car AP50 when training on synthetic data: Virtual KITTI 54.60 -> 61.88, GTA 46.83 -> 51.24 (2975-image subsets); smaller gains (+1 to +2.9) when training on real data — [Carlson et al., Modeling Camera Effects to Improve Visual Learning from Synthetic Data (arXiv 1803.07721)](https://ar5iv.labs.arxiv.org/html/1803.07721) (code released as the "SensorEffectAugmentation" GitHub repo per the paper; repo URL not verified)
- **Motion blur (photogrammetry rule).** Maximum exposure time = GSD / drone speed (one-GSD rule). Example: 2 cm/px at 5 m/s gives 1/250 s — [NTNU, Preventing motion blur in drone mapping](https://www.ntnu.no/blogger/richard-hann/2021/10/07/preventing-motion-blur-in-drone-mapping/). Equivalent pixel form: blur_px = v x t_exp / GSD, with a half-pixel rule of thumb — [Pix-Pro: Shutter speed, ISO and motion blur in drone photogrammetry](https://www.pix-pro.com/blog/shutter-speed-iso-motion-blur-photogrammetry). Vision-systems guidance suggests keeping blur under 0.5 px — [Vision Systems Design](https://www.vision-systems.com/directory/blog/14064933/the-challenge-of-aerial-imaging-achieving-a-clear-and-sharp-image)
- **Video compression hurts small objects most.** Detection of small, low-contrast and fast-moving objects is particularly hampered at higher compression levels. Retraining YOLOv5 on compressed images gave up to about 1 % improvement, and compression-based augmentation is described as effective — [Impact of Video Compression on Object Detection for Surveillance (arXiv 2211.05805)](https://arxiv.org/pdf/2211.05805)
- Concrete ffmpeg settings used to create realistic codec artifacts: H.264 and H.265 at CRF 24, 36, 48. HEVC gives smoother textures, finer detail suppression and mild blocking/ringing — [VITAL (arXiv 2511.17962)](https://arxiv.org/pdf/2511.17962). In H.264/H.265, QP is controlled via CRF in ffmpeg — [VCISR (arXiv 2311.00996)](https://arxiv.org/pdf/2311.00996). QP ranges 0-51, about 4 is near-lossless, 51 is extreme. Frames can be encoded with ffmpeg and decoded to lossless PNG for training — [Video compression artifacts on fisheye perception (arXiv 2403.16338)](https://arxiv.org/pdf/2403.16338)
- **Rolling shutter.** Measured readouts range from about 3 ms to 16 ms on modern full-frame cameras (Sony a7S III 4K about 8.7 ms), and readout depends on mode and frame rate — [CineD lab test](https://www.cined.com/?p=296166). Example: a 30 fps camera with a 27 ms readout has a rolling-shutter fraction of 0.81 — [SynthEyes manual: rolling shutter](https://borisfx.com/documentation/syntheyes/SynthEyesUM_files/rolling_shutter.html). High-frequency vibration on drones is a main driver of "jello" — [same](https://borisfx.com/documentation/syntheyes/syntheyesum_files/rolling_shutter/)

### Inferences
- **Exposure and motion blur in desert sun.** By the sunny-16 rule (f/16, 1/ISO), an F2.8 lens at ISO 100 needs about 1/3200 s on bright sand (5 stops faster than 1/100 at f/16). At 10 m/s that gives 10 x 1/3200 = 3 mm of motion, about 0.1 px at 2-4 cm/px. **Translational motion blur is therefore negligible in normal sunny conditions** with auto exposure. It becomes significant (a) if ND filters or a fixed 1/50 s "cinematic" shutter are used (0.2 m = 5-10 px at 10 m/s, which is unlikely for the A8 mini), (b) at dawn/dusk or in overcast (1/250-1/500 s gives 1-2 px), or (c) during gimbal yaw/pitch slews: angular blur = omega x t_exp / IFOV, so 30 deg/s at 1/1000 s = 0.03 deg = 1.5 px. **Implementation:** with probability about 0.2, apply a linear kernel of length `U(0.5, 3)` px at a random angle (or along the flight direction, consistent across the whole frame), built with `cv2.line` on a supersampled kernel and normalized, then `cv2.filter2D`. Apply it to the whole composite, or to the flag crop **only if the background shows the same blur**. Estimate background blur roughly (for example with the variance of the Laplacian) and skip flag blur for sharp backgrounds. Pitfall: blurring the flag but not the background is a strong shortcut cue.
- **Noise (high priority, refine what exists).** The existing "matched sensor noise" should (a) be signal-dependent and (b) match the spatial correlation of the background's noise. ISP denoising plus H.265 leave low-frequency, blotchy (often chroma) noise, not i.i.d. pixel noise. Cheap method:
  1. Estimate the background noise level function: high-pass the background (`bg - cv2.GaussianBlur(bg, (0,0), 1.5)`), bin pixels by local mean intensity in flat regions (low gradient), and fit `var = a*I + b` per channel.
  2. Estimate the noise power spectrum shape: take the FFT of high-pass residual patches from flat sand and average them, or more simply fit a Gaussian correlation length `s` by comparing the variance of the residual before and after a 2x2 box filter.
  3. Synthesize for the flag: `n = G_s(N(0,1))` renormalized to unit variance, scaled by `sqrt(a*I_lin + b)`, with separate luma and chroma components (chroma noise is usually lower-resolution: generate at half resolution and upsample).
  For video frames, noise is mostly removed by the encoder, so keep the added noise small and let the codec step (below) do the rest.
- **PSF and sharpening (high priority, cheap).** Order: supersampled render -> downsample with area interpolation (`cv2.INTER_AREA`) -> PSF blur `sigma ~ U(0.6, 1.2)` px (optionally a slightly elliptical kernel to mimic off-axis astigmatism near frame corners) -> noise -> **unsharp mask** `I + k*(I - G_r(I))` with `r ~ U(0.8, 2.0)` px and `k ~ U(0.2, 1.0)`. Small-sensor ISPs sharpen strongly, producing the 1-2 px bright/dark halos visible around high-contrast flag edges and stripes. Match `k` by looking at edge overshoot on background objects (runway paint edges, vehicle edges).
- **Compression (highest priority for video-frame data).**
  - Stills (JPEG): re-encode the **whole composite** with `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])`, `q ~ U(70, 95)`. Alternatively re-encode only a crop around the flag whose origin is aligned to the 8x8 (or 16x16 with chroma subsampling) grid of the original, and paste the crop back, to avoid double-compressing the full background.
  - Video (H.265 at 20 Mbps): run actual x265 through ffmpeg for fidelity. For each composite (or each tiled crop), write a short 3-8-frame clip with small sub-pixel shifts between frames (mimicking flight, so the encoder uses inter prediction like real footage), encode with `-c:v libx265 -crf 28..38` (or `-b:v 20M` at the full 4K size to match the real bitrate), decode the middle frame to PNG. Use `-pix_fmt yuv420p` to get the 4:2:0 chroma subsampling, which smears thin red/blue stripes. That is a very visible effect on flags at 1-2 px chroma resolution.
  - A cheap proxy without ffmpeg: convert to YCrCb, downsample Cr/Cb by 2 (`cv2.resize(..., INTER_AREA)` then upsample with `INTER_LINEAR`), apply a mild edge-preserving smooth (`cv2.bilateralFilter`, d=5, sigmaColor 10-20) to mimic the encoder's texture loss, then JPEG at q 50-75. This is less faithful (no 4x4-32x32 transform blocks, no deblocking filter signature) but captures the 4:2:0 chroma smear and texture loss.
  - Pitfall: the background is already compressed. Recompressing the whole frame is realistic (re-encoding is common), but recompressing only the pasted flag with a different quality creates a block-grid or texture-statistics mismatch. Always compress a region containing both flag and background.
- **Chromatic aberration (low-medium).** Lateral CA grows with image radius. Wide 81 deg HFOV small-sensor lenses typically show 0.5-2 px of red/blue fringing near the corners and nearly none at the centre (a typical value, not measured for the A8). Implement on the composite or on a crop with knowledge of the crop's position: per-channel radial scaling about the optical centre with `cv2.remap`, `r_R = r*(1+k)`, `r_B = r*(1-k)`, `k ~ U(0, 8e-4)` (gives about 1.9 px at the 2350 px corner radius). Many ISPs partly correct CA, so sample k = 0 often.
- **Demosaicing (low).** At 50 px flags with PSF sigma of about 1 px, demosaic zipper/false colour appears mainly on very high-contrast thin edges (white star on dark blue). Cheap Carlson-style method: mosaic the flag crop to Bayer (keep one channel per pixel in an RGGB pattern) and demosaic with `cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)` (bilinear) or `..._EA` (edge-aware) / `..._VNG`. This should run in the noise stage (before sharpening and compression). Expected impact is small, so do it only after the items above.
- **Rolling shutter (skip).** Even with a pessimistic 30 ms readout and 10 m/s flight, the full frame skews by 0.3 m = 7-14 px. A flag covering about 50 of 2160 rows (2.3 %) sees about 0.2-0.3 px skew. Vibration-induced jello at the 0.01 deg gimbal spec is about 0.5 px. **Not worth implementing for detection of flags.** It would only matter for long straight objects spanning many rows (runway markings), which are background anyway.
- **Vignetting (fold into local shading).** Rather than a parametric `cos^4` model, use the local background luminance ratio (Section 2), which captures vignetting automatically. If a parametric model is wanted: `V(r) = 1 - a*(r/r_max)^2 - b*(r/r_max)^4`, `a ~ U(0, 0.3)`.
- **Recommended ISP order (on a crop of about 3x flag size containing background):** linear-light composite (flag albedo x illumination + shadows + specular, alpha-composited over the linearized background) -> optional motion kernel -> PSF -> Poisson-Gaussian noise (only on the flag pixels, since the background already has noise; or on everything at low strength) -> optional mosaic/demosaic -> CA -> gamma/tone curve back to sRGB -> unsharp mask -> codec (JPEG or x265) applied to the crop or whole frame.

### Gaps
- No published A8 mini-specific noise parameters, sharpening strength or CA amount were found. All should be estimated from the user's own A8 mini frames (flat sand regions, runway paint edges).
- Carlson et al. did not publish numeric parameter ranges in the paper text.
- No study was found that quantifies the detection benefit of specific ISP effects for objects under 100 px in UAV imagery.

---

## 4. Partial occlusion and environment interaction

### Takeaway
Occlusion and interaction are cheap to synthesize in 2D and plausibly important. Real flags laid on desert ground will be weighted with stones or sand along the edges, partially buried, stepped on, or near people and vehicles. Implement them as alpha-mask edits plus pasted small objects, and keep labels consistent (visible fraction). The main pitfall is creating artifacts that occur only on flags, so apply the same occluders to background-only areas and distractors.

### Cited Findings
- Naive cut-and-paste leaves pixel-level boundary artifacts that hurt trained detectors. Training on the same scenes with multiple blending modes (none, Gaussian, Poisson, motion-blur) makes detectors ignore seams and gave about +8 AP overall versus no blending. Poisson blending alone scored lower than no blending on GMU (58.4 vs 65.9). Cut-Paste-Learn also adds truncation and occlusion of pasted objects — [Dwibedi et al., Cut, Paste and Learn, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dwibedi_Cut_Paste_and_ICCV_2017_paper.pdf)
- Poisson blending can make large parts of pasted objects effectively vanish into the background. DeePaste proposes inpainting to fill the blending gap — [DeePaste (arXiv 2112.10600)](https://arxiv.org/pdf/2112.10600)
- Without instance masks, pasted patches carry mismatched background. SAM segmentation is suggested before pasting — [Stable Diffusion for Aerial Object Detection (arXiv 2311.12345)](https://arxiv.org/pdf/2311.12345)
- Diffusion-based "paste, inpaint and harmonize" methods have been used to improve foreground/background matching for UAV vehicle detection data — [TNO 2025, Snel et al.](https://publications.tno.nl/publication/34645222/07v7xkq9/snel-2025-data.pdf)

### Inferences
- **Sand on edges / partial burial (high value).** Build an occlusion mask `O` along a random subset of edges: `O = (distance_from_edge < w(s)) & (noise > t)`, where `w(s)` is 1D fractal noise along the edge (0 to 15 % of flag width) and the noise is 2D Perlin/blurred noise for an irregular boundary. Inside `O`, replace flag pixels with background texture: either the background pixels at that location (the sand "over" the flag is the same sand), or `cv2.inpaint` / a shifted background patch for buried centre areas. Add a slight brightening rim (wind-blown sand ridge) on the windward side if desired. Update the label: compute visible area, and optionally drop or mark the instance as "difficult" when more than about 60-70 % is hidden.
- **Stones / weights (medium-high, cheap).** Stones of 10-30 cm are 3-15 px at 2-4 cm/px. Synthesize irregular convex polygons (`cv2.fillPoly` on supersampled canvas) with sand/rock colour (sample from background darker or lighter pixels), Lambertian shading from the scene sun vector, and a small cast shadow along the azimuth (length `height/tan(e)/GSD`). Place them at 2-4 corners or mid-edges. Also place similar stones randomly on the background without flags, so stones-at-corners is not a flag-only cue, though in reality it is a real-world correlate.
- **Tent pegs / tape / ropes (low).** At 2-4 cm/px, ropes are sub-pixel and only visible as faint 1 px lines. Skip or add very rarely.
- **Wind-lifted corner / folded-over edge (medium).** In 2D: choose a corner, a fold line across it, reflect the corner triangle across the line (`cv2.getAffineTransform` / `warpAffine` on the texture and alpha) and render it with the reverse-side appearance (Section 1). Add a thin highlight/shadow along the fold line and a soft cast shadow on the ground offset along the sun azimuth by `lift_height/tan(e)/GSD`. A true 3D lifted corner (curled, raised 10-30 cm) is better done as a displacement plus perspective in the existing fold heightfield; this is where Blender cloth simulation would be clearly superior.
- **People / vehicles nearby (medium).** People from 50-100 m nadir are about 10-25 px blobs (shoulders about 0.5 m) with long shadows at low sun. Vehicles are about 110-220 x 45-90 px. Paste real cut-outs (from the user's own background photos or open aerial datasets such as VisDrone/DOTA crops) near and occasionally overlapping flags. Their shadows, cast along the scene azimuth, should fall across flags (Section 2).
- **Drone shadow (low).** Only visible when the sun is behind the camera at a high elevation (near the anti-solar point). Small, rare. Skip unless real backgrounds show it.
- **Footprints / tyre tracks on and around the flag (low-medium).** Darkened, slightly textured strips across the flag, which are mostly the same multiplicative darkening as on surrounding sand.
- **Blending variety (high value, trivial).** Following Dwibedi, randomize the edge treatment per instance (hard alpha, Gaussian-feathered alpha 0.5-2 px, slight morphological erode/dilate of the alpha, motion-blurred alpha) so that no single seam signature exists. The generator already has random feathering. Extend it to include hard edges and sub-pixel alpha from supersampling. Avoid Poisson blending (`cv2.seamlessClone`) as the main method: it shifts the flag's colours toward the background and can wash out flags, as reported above.

### Gaps
- No source quantified how often real ground-laid flags are partially covered or weighted in operational settings. Occlusion frequencies must be chosen by judgement or from real test footage.
- No aerial-specific study of occlusion augmentation for objects of this size was found.

---

## 5. Atmospheric effects at 50-100 m in desert (haze, heat shimmer)

### Takeaway
Both are minor at 50-100 m slant range in clear conditions. Haze reduces contrast by about 1-4 % at 10-20 km visibility, but can reach 15-35 % in dust events (1-2 km visibility). Shimmer adds at most sub-pixel warping for a mostly vertical path. Haze matters mostly as a consistency requirement: the flag must receive the same contrast loss and airlight as the surrounding ground, which the existing black-level matching plus a transmission/airlight step handles. Shimmer can be skipped, or added as an optional sub-pixel warp for midday video frames.

### Cited Findings
- Koschmieder: extinction coefficient `beta = 3.912 / V` (V = meteorological visibility), where 3.912 = -ln(0.02) from a 2 % contrast threshold. Contrast decays exponentially with distance (Lambert-Bouguer), `C = C0 * exp(-beta*d)` — [WMO CIMO Guide, visibility chapter](https://old.wmo.int/extranet/pages/prog/www/IMOP/publications/CIMO-Guide/Prelim_2018_ed/8_I_9_en_MR_tc.pdf); [Horvath review (pp.bme.hu)](https://pp.bme.hu/ci/article/download/4169/3274)
- The Koschmieder model assumes homogeneous illumination, constant extinction along the path, and a black target against the horizon sky, so it is only accurate in limited conditions — [Lee & Shang 2016](https://cocc.xmu.edu.cn/Lee_2016_Visibility.pdf); [USPTO 7899207](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7899207)
- Turbulence near hot surfaces such as desert terrain and summer roads causes shimmering, distortion and blurring. Distance and heat are the two dominant factors — [CMU, Depth from Optical Turbulence](https://www.ri.cmu.edu/publications/depth-from-optical-turbulence)
- In arid/desert environments the diurnal change in Cn2 can span three to four orders of magnitude or more — [SPIE 3494, atmospheric turbulence monitoring](https://spiedigitallibrary.org/conference-proceedings-of-spie/3494/1/Atmospheric-turbulence-monitoring-in-conjunction-with-imager-designator-operation/10.1117/12.332424.full)
- Over very short distances, small refractive-index changes usually cause negligible problems for all but the most demanding optical systems — [SPIE 6747 (search snippet)](https://astronomicaltelescopes.spiedigitallibrary.org/conference-proceedings-of-spie/6747/67470D/Impact-of-turbulence-on-the-measurement-of-CCD-materials-in/10.1117/12.740710.full)

### Inferences
- **Haze magnitude.** Transmission `t = exp(-3.912 d / V)` at slant range d (= h / cos(off-nadir angle)):
  - V = 20 km, d = 100 m: t = 0.981 (1.9 % contrast loss)
  - V = 10 km, d = 100 m: t = 0.962
  - V = 2 km (dusty): t = 0.822
  - V = 1 km (dust storm): t = 0.676
  Implementation (in linear light): `I = J*t + A*(1-t)`, where `A` is the airlight colour, estimated as the colour of the brightest, least-saturated distant region or from the dark-channel prior on the background. Since the background already contains the haze, apply it to the flag with `t` estimated from the background (compare local black level with the darkest object shadows). This essentially generalizes the existing black-level match. Vary `t` slightly with the off-nadir angle across a tilted frame.
- **Heat shimmer.** The view path is mostly vertical. The strongest turbulence is in the first few metres above hot ground, so only a short part of the 50-100 m path is strongly turbulent. Shimmer is much weaker than for horizontal ground-level imaging at the same range. Expected effect: random sub-pixel (about 0.1-0.5 px) low-frequency warping, varying between video frames, plus slight extra blur at midday. Optional implementation: `cv2.remap` with displacement fields `dx, dy = G_sigma(N(0,1)) * amp`, `sigma ~ 5-20 px`, `amp ~ U(0, 0.4)` px, applied to the whole crop including the background. Priority: low.

### Gaps
- No measurement of turbulence-induced blur (Fried parameter r0 or MTF) for near-vertical 50-100 m paths from a UAV over desert was found. The sub-pixel estimate is a reasoned inference.
- No desert-specific visibility statistics were retrieved for the user's operating area.

---

## 6. Distractors / hard negatives on airfields and desert

### Takeaway
The most flag-like objects from above are rectangular, colourful, roughly 1-4 m objects: tarps (blue/orange/green PE), banners and signs lying flat, vehicles (cars at about 4.5 x 1.8 m have a similar aspect ratio of about 2.5:1), runway/taxiway paint (threshold bars, numbers, chevrons, holding-position lines), mats/rugs, solar panels, container and shed roofs, debris, and windsocks or marker panels. Synthesize "anti-flags" with the **same rendering pipeline** (folds, shading, shadows, noise, compression) so the detector cannot use rendering artifacts as the positive cue, and include real-photo distractors where available.

### Cited Findings
- Aerial vehicle detection suffers from hard negatives (lookalike background regions). Tang et al. add a cascade of boosted classifiers with hard negative mining to reject them and report fewer false detections — [Tang et al., Vehicle detection in aerial images (Remote Sensing 2017, via Crossref)](https://api.crossref.org/works/10.3390%2FRS10010124)
- "Hard chip mining" for VisDrone/DOTA augments hard examples beyond random negative chips to reduce false positives — [Hong et al., Patch-Level Augmentation for Object Detection in Aerial Images, ICCVW 2019](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Hong_Patch-Level_Augmentation_for_Object_Detection_in_Aerial_Images_ICCVW_2019_paper.pdf)
- Naively adding low-confidence teacher predictions as negatives increased false alarms, so confidence-weighted negatives work better (some low-confidence boxes are true positives) — [Rethinking Scale Imbalance in Semi-supervised Object Detection for Aerial Images (arXiv 2310.14718)](https://arxiv.org/pdf/2310.14718)
- Runway threshold markings are either eight equal longitudinal stripes or a count tied to runway width (4 stripes for 60 ft, 6 for 75 ft, 8 for 100 ft, 12 for 150 ft, 16 for 200 ft) — [FAA AIM 2-3, Airport Marking Aids and Signs](https://www.faa.gov/air_traffic/publications/ATpubs/AIM/aim0203.html). The current marking standard is AC 150/5340-1M Change 1 — [FAA AC 150/5340-1M Chg 1](https://www.faa.gov/documentLibrary/media/Advisory_Circular/150-5340-1M-Chg-1-Airport-Markings.pdf)
- Detectors learn to exploit pasting artifacts unless blending is randomized — [Dwibedi et al., ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dwibedi_Cut_Paste_and_ICCV_2017_paper.pdf)

### Inferences
- **Distractor catalogue and synthesis recipes (NumPy/OpenCV):**
  - *Tarps / plastic sheets*: solid blue, orange, green, silver or brown rectangles, 2x3 m, 3x4 m, 4x6 m (sizes are common retail values). Same fold model but higher gloss (`ks` up to 0.3, since PE is shinier), grommet dots at the edges, often weighted by stones. These are the strongest hard negatives for single-colour or two-colour flags.
  - *Banners / signs lying flat*: white or yellow rectangles with text-like dark blobs (random glyph strokes via `cv2.putText` with random fonts, then blur).
  - *"Anti-flags"*: tricolour or striped rectangles in colour combinations that are not target flags (if the task is flag detection of specific countries), or real flags of non-target countries if the classifier must discriminate. Reuse the full flag renderer so render artifacts are shared.
  - *Vehicles*: real crops are best. Otherwise draw a rounded rectangle with darker windshield/rear-window bands and a specular roof highlight, plus a hard cast shadow. Aspect ratio 2.2-2.6.
  - *Runway / apron paint*: white or yellow bars, numbers and chevrons with worn edges (multiply by noise), drawn with correct sizes relative to GSD. Mostly already present in real airfield backgrounds, so ensure the backgrounds contain them and do not paste flags only away from them.
  - *Mats / rugs / blankets*: about 0.7x1.2 m to 2x3 m patterned rectangles (random repeating geometric textures), with low saturation and fringed edges.
  - *Solar panels, container roofs, sheds*: dark blue grid rectangles, corrugated metal (sinusoidal stripes with specular highlight). Usually larger (6 m+), so they are mainly size-based negatives.
  - *Debris, cardboard, plywood*: brown or beige rectangles with folds.
- **Hard-negative mining loop:** after a first training round, run the detector on real flag-free airfield/desert frames, collect false positives, and (a) add those frames as negatives, (b) crop the false positives and reuse them as paste-in distractors in new synthetic images.
- **Pitfall:** if only flags receive folds, contact shadows, feathered alpha and matched noise, the detector learns "anything with pasted-object statistics is a flag". Every distractor class should pass through the identical pipeline, and some real flags should receive minimal processing (taut, no folds, no shadow).

### Gaps
- No published dataset or study was found that lists false-positive categories for small ground-laid fabric targets on airfields. The catalogue above is based on reasoning.
- Exact FAA stripe dimensions (bar width and length) were not retrieved from the AC figures.

---

## 7. Overall prioritization: which effects are worth implementing in NumPy/OpenCV

### Takeaway
Highest return per effort: (1) real codec simulation (x265 via ffmpeg for video-frame data, JPEG for stills) applied to a crop that includes the background; (2) a linear-light ISP-ordered chain with PSF sigma sampling, signal-dependent and spatially correlated noise, and unsharp-mask halos; (3) a per-background sun vector driving fold shading plus local shading/shadow-ratio modulation so flags darken under scene shadows and follow vignetting; (4) colour fading, print variation and dust; (5) edge occlusion by sand and stones, plus blending-mode randomization; (6) a distractor set rendered with the same pipeline. Rolling shutter, heat shimmer and drone shadow are not worth implementing at this pixel size. Blender is only clearly better for wind-lifted 3D drape and cloth self-shadowing.

### Cited Findings
- Physically motivated sensor-effect augmentation (CA, blur, exposure, Poisson-Gaussian noise, colour shift) improved synthetic-to-real car detection by about +4 to +7 AP50 on KITTI — [Carlson et al. (arXiv 1803.07721)](https://ar5iv.labs.arxiv.org/html/1803.07721)
- Randomized blending gave about +8 AP over no blending in cut-paste detection training — [Dwibedi et al., ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dwibedi_Cut_Paste_and_ICCV_2017_paper.pdf)
- Small objects suffer most from compression, and compression augmentation helps — [arXiv 2211.05805](https://arxiv.org/pdf/2211.05805)
- Unprocessing to raw and adding shot/read noise is more realistic than additive noise — [Brooks et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.pdf)

### Inferences
Priority table (realism at 45-95 px; effort in NumPy/OpenCV):

| # | Effect | Visible at 50 px? | Method (cheap) | Effort | Priority |
|---|---|---|---|---|---|
| 1 | Video codec (H.265 20 Mbps, 4:2:0) / JPEG | Yes, strongly: smears thin stripes and chroma, kills texture | ffmpeg x265 CRF 28-38 on crop/frame, decode PNG; proxy: YCrCb chroma 2x down/up + bilateral + JPEG q50-75 | Low-Med | **1** |
| 2 | ISP-ordered chain in linear light (PSF sigma 0.6-1.2 px -> noise -> gamma -> USM halos) | Yes: edge profile and halo signature | `GaussianBlur`, Poisson-Gaussian `var=a*I+b` fitted to background, correlated noise via blurred N(0,1), USM k 0.2-1.0 | Low | **1** |
| 3 | Local illumination match (vignetting, cloud shadow, exposure gradient) | Yes: brightness mismatch is obvious | ratio of large-scale low-pass background luminance; multiply in linear light | Low | **2** |
| 4 | Scene shadows crossing the flag + sun-consistent fold shading | Yes when present; very strong cue | per-background sun azimuth/elevation (metadata or click); per-channel shadow ratio from background; Lambert `n·L` on fold normals | Med | **2** |
| 5 | Fading, printer colour variation, dust/dirt | Yes (colour and contrast are what remains at 50 px) | Lab chroma/L/hue jitter with low-freq masks; lerp toward local sand colour | Low | **2** |
| 6 | Sand/stone occlusion of edges, folded-over corner with reverse print | Yes | noise-driven alpha erosion, background re-fill, `fillPoly` stones with sun shadow; affine corner flip | Med | **3** |
| 7 | Blending-mode randomization (hard/feathered/eroded alpha) | Subtle but learnable shortcut | random per-instance alpha ops | Trivial | **3** |
| 8 | Distractors through the same pipeline + hard-negative mining | N/A (dataset-level) | tarps, anti-flags, banners, vehicles, paint, mats | Med | **3** (high impact on false positives) |
| 9 | Packaging crease grid; taut vs draped variety | Partly: creases show as 1-2 px lines at 50 m | straight-line displacement and shading | Low | 4 |
| 10 | Specular sheen on folds | Occasionally (glints) | Blinn-Phong on fold normals, ks 0-0.15, p 5-30 | Low | 4 |
| 11 | Haze / airlight consistency | Weak (2-4 %), strong only in dust | `I=J t + A(1-t)` with t from background | Trivial | 4 |
| 12 | Motion blur (pan/vibration/low light) | Rarely (sub-px in sun at about 1/3000 s) | linear kernel 0.5-3 px, p about 0.2, only when background is equally blurred | Low | 5 |
| 13 | Lateral chromatic aberration | Only near frame corners, 0.5-2 px | per-channel radial `remap`, k up to 8e-4 | Low | 5 |
| 14 | Bayer mosaic + demosaic | Barely | `cv2.cvtColor(bayer, COLOR_BayerRG2BGR[_EA/_VNG])` | Low | 6 |
| 15 | Heat shimmer | Sub-pixel | low-freq `remap` field, amp 0.1-0.4 px | Low | 7 (skip) |
| 16 | Rolling shutter | about 0.2-0.5 px across a flag | row-dependent shift | Low | skip |
| 17 | Drone shadow | Rare | blob shadow near anti-solar point | Low | skip |

- **When Blender is clearly better:** wind-lifted or curled corners and flags draped over rough ground with physically correct self-occlusion and self-shadowing (cloth simulation + heightfield terrain), glossy sheen with correct BRDF (Principled BSDF "Sheen"), and fully consistent cast shadows from 3D props (stones, poles, people). A hybrid works well: pre-render a library of a few hundred flag drape meshes as normal/height/UV maps once in Blender (orthographic top-down), then texture, light (with the per-background sun vector) and composite them in NumPy at generation time. This keeps generation fast while gaining realistic geometry.
- **Global pitfalls:**
  1. Apply all degradations in ISP order, in linear light where physics applies (shading, haze, noise), and to a region containing both flag and background (blur, sharpening, compression). Flag-only processing that differs statistically from the background is a learnable shortcut.
  2. Do not double-degrade: the background already carries PSF, noise, sharpening and compression. Add to the flag only what it lacks (PSF, noise, sharpening), then apply a mild shared step (compression) to the crop or frame.
  3. Supersample geometry and alpha (already done), but downsample with `INTER_AREA`, not `INTER_LINEAR`, to avoid aliasing on thin stripes.
  4. Keep labels in sync with occlusion and fold-over: recompute the tight box or OBB from the final visible alpha.
  5. Validate realism quantitatively: compare per-instance statistics of real vs synthetic flags (edge overshoot amplitude, noise residual std vs intensity, chroma bandwidth via Cr/Cb spectrum) and train a small real-vs-synthetic patch classifier. If it reaches high accuracy, inspect its saliency to find the remaining giveaway.

### Gaps
- No ablation study was found that ranks these specific effects for small (<100 px) fabric targets in drone imagery. The priority order is a reasoned judgement from the camera geometry and the general findings above, and should be checked with ablations on real A8 mini test footage.
- Sun-shadow ratios, A8 mini noise/sharpening parameters, and real compression settings (photo JPEG quality) need to be measured from the user's own captures.
