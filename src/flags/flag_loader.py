from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

print("I AM RUNNING FLAG_LOADER.PY")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLAGS_FOLDER = PROJECT_ROOT / "assets" / "flags"

ALLOWED_EXTENSIONS= {".png", ".jpg", ".jpeg"} #in lower case


def load_flag(path):
    """
    Loads one flag image and returns it as an RGBA image.
    RGBA means 4 values for every pixel:
    Red, Green, Blue and Alpha (0 = transparent, 255 = solid).
    Every flag comes out in this same format, whatever
    format the file was saved in (PNG, JPG, grayscale, etc.).
    """

    with Image.open(path) as image: #load one flag image

        #Some photos store a "rotated" flag inside the file. This turns the image the right way up. For normalflag files it changes nothing.
        image= ImageOps.exif_transpose(image)

        #Convert any color mode to RGBA.
        #A JPG has no transparency, so every pixel gets alpha 255 (solid)
        flag= image.convert("RGBA")

    return flag


def load_all_flags(folder=FLAGS_FOLDER):
    """
    Loads every flag in the folder.
    Returns a dictionary: class name -> RGBA image. Example: {"egypt": <image>, "germany": <image>}
    Files that cannot be loaded are skipped and listed in a printed message, so nothing fails silently.
    """

    folder= Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Flags folder was not found or is not a folder: {folder}"
        )

    #All allowed image files in the folder, in any letter case, sorted.
    paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )

    #An empty list counts as False, so "not paths" means "no supported images were found". Say so and return an empty dictionary.
    if not paths:
        print(f"No supported flag images were found in: {folder}")
        print(f"Looked for: {sorted(ALLOWED_EXTENSIONS)}")
        return {}

    flags= {}
    skipped= []

    for path in paths:
        #The class name is the file name without the extension, made lowercase: "Egypt.png" -> "egypt".
        class_name= path.stem.lower()

        #Two files with the same class name would overwrite each other.
        if class_name in flags:
            skipped.append(f"{path.name}: same class name as another file") #append a msg to the skipped list but dont raise an error
            continue
        try:
            flags[class_name] = load_flag(path)
        except Exception as error:
            skipped.append(f"{path.name}: {error}")

    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for message in skipped:
            print(f"  - {message}")
    return flags


def flag_to_array(flag):
    """
    Turns an RGBA image into a NumPy array of shape (height, width, 4). #4 channels
    The channel order stays R, G, B, A.
    Careful: OpenCV normally expects B, G, R order. convert with: cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    """

    return np.array(flag, dtype=np.uint8)

# It loads the flags and prints a short summary.
if __name__ == "__main__":
    flags = load_all_flags()
    print(f"\nLoaded {len(flags)} flags.\n")
    for class_name, flag in flags.items():
        print(f"{class_name}: size={flag.size}, mode={flag.mode}")