Full drone frames (no flags in them) for `python generate_batch.py --zip`.

The 200 frames `<map>_cNN_KK.jpg` are cut from the drone maps in
`assets/background_sources.yaml` by `python tools/cut_orthomosaics.py` (not in git; re-run
the tool to get the same files). Their lines in `index.yaml` sit between the
`cut_orthomosaics` begin/end marks and are rewritten by the tool.

Your own photos: copy them here and add a line OUTSIDE that block when you know more:

    - {file: IMG_0012.jpg, altitude_m: 80}            # flown altitude -> GSD from the camera HFOV
    - {file: IMG_0013.jpg, gsd_m: 0.031}              # or the GSD itself (m per pixel)
    - {file: IMG_0014.jpg, group: flight2}            # photos of one place/flight stay in one split
    - {file: IMG_0015.jpg, use: false}                # keep the file, leave it out

A photo not listed is taken as a full frame at a random 50-100 m.
With more than 200 photos here the batch gets bigger; use `python tools/cut_orthomosaics.py
--frames N` to keep the total at 200.
