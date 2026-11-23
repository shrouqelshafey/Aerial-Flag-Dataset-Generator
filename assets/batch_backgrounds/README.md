Put the real background photos (full drone frames, no flags in them) here,
then run `python generate_batch.py --zip`.

Optional `index.yaml` in this folder, one line per photo, when you know more:

    - {file: IMG_0012.jpg, altitude_m: 80}            # flown altitude -> GSD from the camera HFOV
    - {file: IMG_0013.jpg, gsd_m: 0.031}              # or the GSD itself (m per pixel)
    - {file: IMG_0014.jpg, group: flight2}            # photos of one place/flight stay in one split
    - {file: IMG_0015.jpg, use: false}                # keep the file, leave it out

A photo not listed is taken as a full frame at a random 50-100 m.
