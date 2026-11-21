
#You have a folder full of flag images (assets/flags), before using them to build your synthetic dataset, this script checks them for problems, for example:
#   - a file that is corrupt or cannot be opened
#   - an image that is too small, blank, or badly named
#   - the same flag saved twice, or two very similar flags

# It produces two files in output/metadata/:
#   1. flag_validation_report.csv  -> one row per image, with all the checks
#   2. flag_contact_sheet.png      -> all flags in one picture, to look at by eye

# Every image ends up with one of three statuses:
#   PASS   = no problems found
#   REVIEW = something looks odd, a human should look at it
#   ERROR  = the file could not be read at all

# The script never deletes or changes your flag images BUT it passes the right information to the dataset generator


from pathlib import Path

#Pillow (PIL): opens and edits images.
#Image= load, resize and create images
#ImageDraw= draw text and shapes on an image
from PIL import Image, ImageDraw

import numpy as np
import hashlib
import csv
import re

# for example: project/src/tools/validate_flags.py
PROJECT_ROOT= Path(__file__).resolve().parents[2]

FLAGS_FOLDER= PROJECT_ROOT / "assets" / "flags"
REPORT_FOLDER= PROJECT_ROOT / "output" / "metadata" #Where the reports will be saved.

#The two output files.
REPORT_FILE= REPORT_FOLDER / "flag_validation_report.csv"
CONTACT_SHEET_FILE= REPORT_FOLDER / "flag_contact_sheet.png"

ALLOWED_EXTENSIONS= {".png", ".jpg", ".jpeg"}

MIN_WIDTH= 256 #An image smaller than this (in pixels) gets a LOW RESOLUTION warning.
MIN_HEIGHT= 256 

# DESIGN NOTE: this script only CHECKS the shape; it never stretches anything. 
# Stretching, padding or cropping a flag to fit the real
# 2 m x 1 m rectangle happens later, in the dataset generator, so that
# is where you must decide how to handle flags that are not 2:1.
MIN_ASPECT_RATIO= 0.5
MAX_ASPECT_RATIO= 3.0

BORDER_TOLERANCE= 10 #solid backround or solid border is suspicious?
MIN_FILE_SIZE=1024 #file may be corrupt or blank, or it may be a tiny icon instead of a real flag (suspiciously small file size)


CLASS_NAME_PATTERN= re.compile(r"^[a-z0-9_]+$")
#DESIGN NOTE: this pattern already rejects capital letters, so "Egypt.png" gets a BAD CLASS NAME warning by itself (check 1)
#check 2 looks at all the file names together and asks whether two files stand for the same country. It ignores capital letters when comparing.

#For comparing flags to each other, Exact copies: a flag that exists twice gets pasted into twice as many images, so the model sees it more than the others (memorize it instead of learning to generalize)
THUMBNAIL_SIZE= (32, 16) 

#Two flags are "very similar" if their average pixel difference is below this threshold.
#the warning it gives is useful to know about because those are the pairs your detector may confuse
NEAR_DUPLICATE_THRESHOLD= 8

#Two flags are only compared if the area they share is at least this fraction of the smaller flag (0.5 = half). 
MIN_OVERLAP_FRACTION= 0.5

SHEET_COLUMNS= 6     
CELL_WIDTH= 200       
CELL_HEIGHT= 100      
LABEL_HEIGHT= 20      
PADDING= 10           



# Small functions, each doing one job. 
def calculate_hash(image_path):
    #Calculates a hash to identify exact duplicate files.
   
    with open(image_path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def load_on_white(image):
    #Returns an RGB copy of the image. Transparent areas are filled with white so every flag can be compared and displayed consistently.
    rgba_image= image.convert("RGBA")

    white_background = Image.new(
        "RGBA",
        rgba_image.size,
        (255, 255, 255, 255)
    )
    combined = Image.alpha_composite(
        white_background,
        rgba_image
    )
    return combined.convert("RGB")


def check_transparency(image):
    #Checks whether an image contains actual transparent pixels.
    rgba_image = image.convert("RGBA")
    alpha = np.array(rgba_image.getchannel("A"))
    transparent_pixels = np.sum(alpha == 0)
    partial_pixels = np.sum(
        (alpha > 0) & (alpha < 255)
    )
    total_pixels = alpha.size
    transparency_percentage = (
        (transparent_pixels / total_pixels) * 100
    )

    partial_percentage = (
        (partial_pixels / total_pixels) * 100
    )
    if transparent_pixels > 0 or partial_pixels > 0:
        transparency = "YES"
    else:
        transparency = "NO"
    return (
        transparency,
        round(transparency_percentage, 2),
        round(partial_percentage, 2)
    )


def check_border(image):
    # Checks whether the image border has nearly uniform colors. This can indicate a solid background, but it cannot prove that the background is removable.
   

    rgba_image = image.convert("RGBA")
    array = np.array(rgba_image)
    visible_pixels = array[:, :, 3] > 10
    rgb = array[:, :, :3]
    border_pixels = np.concatenate([
        rgb[0, :, :],rgb[-1, :, :],rgb[:, 0, :],rgb[:, -1, :]
    ])
    border_alpha = np.concatenate([
        visible_pixels[0, :],visible_pixels[-1, :],
        visible_pixels[:, 0],visible_pixels[:, -1]
    ])

    border_pixels = border_pixels[border_alpha]
    if len(border_pixels) == 0:
        return "TRANSPARENT BORDER"
    channel_std = np.std(border_pixels, axis=0)
    if np.all(channel_std <= BORDER_TOLERANCE):
        return "UNIFORM BORDER - REVIEW"

    return "NON-UNIFORM BORDER"


def check_image_content(image):
    #Checks whether an image is empty or has very little variation between its visible pixels.
    rgba_image = image.convert("RGBA")
    array = np.array(rgba_image)
    visible_mask = array[:, :, 3] > 10 #True where the pixel is visible (alpha above 10).
    visible_pixels = array[:, :, :3][visible_mask]

    #No visible pixel at all means the image is empty
    if len(visible_pixels) == 0:
        return "EMPTY / FULLY TRANSPARENT"
    channel_std = np.std(visible_pixels, axis=0)
    if np.max(channel_std) < 3:
        return "LOW COLOR VARIATION - REVIEW"

    return "CONTENT DETECTED"


def check_class_name(image_path):
    #Checks whether the file name can be used as a class name. Example: egypt.png -> egyptو saudi_arabia.png -> saudi_arabia
    class_name = image_path.stem
    # .match checks the name against the pattern from the settings.
    # "not" means: if it does NOT match, return a warning message.
    if not CLASS_NAME_PATTERN.match(class_name):
        return (
            "BAD CLASS NAME - USE LOWERCASE LETTERS, "
            "NUMBERS AND UNDERSCORES ONLY"
        )
    return 


def make_thumbnail(image):
    """
    Creates a small version of the flag for near-duplicate
    comparison.
    Preserves the original aspect ratio instead of stretching
    every flag to the same proportions.
    """

    rgb_image = load_on_white(image)
    rgb_image.thumbnail(THUMBNAIL_SIZE)
    canvas = Image.new(
        "RGB",
        THUMBNAIL_SIZE,
        (255, 255, 255)
    )

    x = (THUMBNAIL_SIZE[0] - rgb_image.width) // 2
    y = (THUMBNAIL_SIZE[1] - rgb_image.height) // 2

    canvas.paste(rgb_image, (x, y))

    box = (x,y,x + rgb_image.width,y + rgb_image.height)

    #Return two things:
    #   1. the canvas as a grid of numbers (as decimals, so that
    #      subtracting two thumbnails later cannot go wrong)
    #   2. the bounding box of the real flag
    return np.array(canvas, dtype=float), box


def validate_flag(image_path):
    #Validates one flag. Returns the validation result and a thumbnail. The thumbnail is None if the image cannot be read.
  
    result = {
        "filename": image_path.name,"class_name": image_path.stem,"width": "","height": "","format": "","mode": "",
        "aspect_ratio": "","file_size_kb": "","transparency": "","transparent_pixels_percent": "","partial_transparency_percent": "",
        "border_check": "","content_check": "","duplicate_of": "","similar_to": "","status": "PASS",
        #Starts as an empty list. Each problem found is added to it.
        "warnings": []
    }

    thumbnail = None #Will stay None if the image cannot be read.

    # try/except: if anything inside "try" crashes (for example acorrupt file), the program jumps to "except" instead of stopping.
    try:
        #First pass: verify() checks that the file is not corrupt.Open and verify the image.
        with Image.open(image_path) as image:
            image.verify()

        # Reopen after verify because verify() invalidates the original image object.
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            result["width"] = width
            result["height"] = height
            result["format"] = image.format   #for example PNG or JPEG
            result["mode"] = image.mode       
            result["file_size_kb"] = round(
                image_path.stat().st_size / 1024,
                2
            )
            aspect_ratio = width / height
            result["aspect_ratio"] = round(
                aspect_ratio,
                3
            )
            name_warning = check_class_name(image_path)
            if name_warning:
                result["warnings"].append(name_warning)
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                result["warnings"].append(
                    "LOW RESOLUTION"
                )
            if not (
                MIN_ASPECT_RATIO
                <= aspect_ratio
                <= MAX_ASPECT_RATIO
            ):
                result["warnings"].append(
                    "UNUSUAL ASPECT RATIO - REVIEW"
                )
            #Check transparency.
            (
                transparency,
                transparent_percent,
                partial_percent
            ) = check_transparency(image)

            result["transparency"] = transparency

            result["transparent_pixels_percent"] = (
                transparent_percent
            )

            result["partial_transparency_percent"] = (
                partial_percent
            )

            # Check border appearance.
            result["border_check"] = check_border(image)

            if result["border_check"] == "UNIFORM BORDER - REVIEW":
                result["warnings"].append(
                    "POSSIBLE SOLID BACKGROUND - INSPECT MANUALLY"
                )

            # Check whether the image contains useful content.
            result["content_check"] = check_image_content(image)

            # Anything other than "CONTENT DETECTED" is a problem, and the message itself is used as the warning.
            if result["content_check"] != "CONTENT DETECTED":
                result["warnings"].append(
                    result["content_check"]
                )

            # Check file size.
            if image_path.stat().st_size < MIN_FILE_SIZE:
                result["warnings"].append(
                    "VERY SMALL FILE"
                )

            # Create a thumbnail for near-duplicate comparison.(It is a pair: the pixel grid and the real flag's bounding box.)
            thumbnail = make_thumbnail(image)

    except Exception as error:
        result["status"] = "ERROR"
        result["warnings"].append(
            f"UNABLE TO READ IMAGE: {error}"
        )

    # If there is at least one warning (and it is not an ERROR),
    # the status changes from PASS to REVIEW.
    if result["warnings"] and result["status"] != "ERROR":
        result["status"] = "REVIEW"
    return result, thumbnail



def check_duplicate_class_names(results):
    """
    Warns if two files have the same class name.Comparison ignores capitalization. Example:
    Egypt.png and egypt.jpg are treated as the same class.
    """

    # A dictionary remembering which class names we have already met:
    # class name -> the file that had it first.
    seen = {}
    for result in results:
        class_name = result["class_name"].lower()
        #"in" checks if the name is already a key in the dictionary.
        if class_name in seen:
            result["warnings"].append(
                f"SAME CLASS NAME AS {seen[class_name]}"
            )

            result["status"] = "REVIEW"
        else:
            seen[class_name] = result["filename"] # First time we see this name: remember it.


def check_near_duplicates(results, thumbnails):
    """
    Compares every pair of flags using small thumbnails.

    Only the area where both flags exist is compared.
    Pairs with very different shapes are skipped.

    Similar flags are reported for manual inspection.
    Similarity does not mean the flags are identical.
    """

    names = list(thumbnails.keys())
    result_by_name = {
        result["filename"]: result
        for result in results
    }

    # Two loops to visit every PAIR of flags exactly once: i picks the first flag, j picks any flag after it. (So we compare A-B but not B-A, and never A-A.)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]
            grid_a, box_a = thumbnails[name_a]
            grid_b, box_b = thumbnails[name_b]

            left = max(box_a[0], box_b[0])
            top = max(box_a[1], box_b[1])
            right = min(box_a[2], box_b[2])
            bottom = min(box_a[3], box_b[3])

            #Size of the shared area. If the boxes do not touch at all, the width or height would be zero or negative.
            overlap_width = right - left
            overlap_height = bottom - top

            if overlap_width <= 0 or overlap_height <= 0:
                continue
            overlap_area = overlap_width * overlap_height

            area_a = (
                (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            )
            area_b = (
                (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            )

            if overlap_area < MIN_OVERLAP_FRACTION * min(area_a, area_b):
                continue

            region_a = grid_a[top:bottom, left:right]
            region_b = grid_b[top:bottom, left:right]

            difference = np.mean(
                np.abs(region_a - region_b)
            )
            #Below the threshold means "very similar".
            if difference < NEAR_DUPLICATE_THRESHOLD:
                #The warning is put on the second flag of the pair.
                result_b = result_by_name[name_b]
                result_b["similar_to"] = name_a
                result_b["warnings"].append(
                    f"VERY SIMILAR TO {name_a} - REVIEW"
                )
                result_b["status"] = "REVIEW"




def save_contact_sheet(flag_images):
    """
    Saves one image showing every flag with its file name.
    Each flag is centered inside its cell and keeps
    its original proportions.
    """
    cell_total_height = CELL_HEIGHT + LABEL_HEIGHT
    columns = SHEET_COLUMNS
    rows = (
        len(flag_images) + columns - 1
    ) // columns

    sheet_width = (
        columns * (CELL_WIDTH + PADDING) + PADDING
    )
    sheet_height = (
        rows * (cell_total_height + PADDING) + PADDING
    )
    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        (230, 230, 230)
    )

    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(flag_images):
        row = index // columns
        column = index % columns
        x = PADDING + column * (CELL_WIDTH + PADDING)
        y = PADDING + row * (cell_total_height + PADDING)

        try:
            with Image.open(image_path) as image:
                flag = load_on_white(image)
                flag.thumbnail((CELL_WIDTH, CELL_HEIGHT))
                flag_x = x + (CELL_WIDTH - flag.width) // 2
                flag_y = y + (CELL_HEIGHT - flag.height) // 2

                sheet.paste(flag, (flag_x, flag_y))
        except Exception:
            # If a file cannot be opened, write UNREADABLE in red
            # (255 is max red; the colors are (R, G, B) = (200, 0, 0)).
            draw.text(
                (x, y),
                "UNREADABLE",
                fill=(200, 0, 0)
            )
        draw.text(
            (x, y + CELL_HEIGHT + 4),
            image_path.name,
            fill=(0, 0, 0)
        )

    sheet.save(CONTACT_SHEET_FILE)


def validate_all_flags():
    if not FLAGS_FOLDER.exists(): #Stop early if the flags folder does not exist.
        print("ERROR: Flags folder was not found.")
        print(FLAGS_FOLDER)
        return

    # Accept PNG, JPG and JPEG in any letter case then sort the list
    flag_images = sorted(
        path for path in FLAGS_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    )

    #An empty list counts as False, so "not flag_images" means "no images were found".
    if not flag_images:
        print("No flag images were found.")
        print(
            f"Looked for: {sorted(ALLOWED_EXTENSIONS)}"
        )
        return

    print("\nStarting flag validation...\n")
    results = [] #Will hold one result per image.
    thumbnails = {}
    hashes = {}

    #Go through every image, one at a time
    for image_path in flag_images:
        result, thumbnail = validate_flag(image_path) # Run all single-image checks.

        # Check exact duplicate files safely.
        # (Skipped for files that could not be read.)
        if result["status"] != "ERROR":

            # try/except so one unreadable file cannot crash the run.
            try:
                image_hash = calculate_hash(image_path)

                # Have we seen this exact fingerprint before?
                if image_hash in hashes:
                    # Yes: this file is a byte-for-byte copy.
                    result["duplicate_of"] = hashes[image_hash]
                    result["warnings"].append(
                        "EXACT DUPLICATE IMAGE"
                    )
                    result["status"] = "REVIEW"
                else:
                    #No: remember this fingerprint and file name.
                    hashes[image_hash] = image_path.name

            # OSError covers file-reading problems (permissions, etc.).
            except OSError as error:

                result["warnings"].append(
                    f"HASH CHECK FAILED: {error}"
                )
                result["status"] = "REVIEW"

        #Store thumbnail if the image was read successfully.
        if thumbnail is not None:
            thumbnails[image_path.name] = thumbnail
        #Add this image's result to the list.
        results.append(result)

    # Run checks that compare flags with each other.
    check_duplicate_class_names(results)
    check_near_duplicates(results, thumbnails)

    # Print results after all checks are complete,
    # so each file shows all of its warnings.
    for result in results:

        print(
            f"{result['filename']}: "
            f"{result['status']}"
        )

        for warning in result["warnings"]:

            print(f"  - {warning}")

    REPORT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # The CSV column names, in the order they will appear.
    fieldnames = ["filename","class_name","width","height","format","mode","aspect_ratio","file_size_kb","transparency","transparent_pixels_percent","partial_transparency_percent",
    "border_check","content_check","duplicate_of","similar_to","status","warnings"]

    with open(REPORT_FILE,"w",newline="",encoding="utf-8-sig") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        for result in results:

            # Convert the warning list into CSV-friendly text.
            # .copy() makes a separate copy of the row, so the original
            # result (with its warnings still a list) is left unchanged.
            # DESIGN NOTE: harmless and correct. Nothing in this script
            # uses the warnings after this point, so it only matters if
            # you add more code later that needs the list.
            row = result.copy()
            row["warnings"] = "; ".join(
                result["warnings"]
            )

            writer.writerow(row)

    # Save contact sheet.
    save_contact_sheet(flag_images)
    passed = sum(
        result["status"] == "PASS"
        for result in results
    )
    review = sum( 
        result["status"] == "REVIEW"
        for result in results
    )
    errors = sum(
        result["status"] == "ERROR"
        for result in results
    )
    print("\n" + "=" * 40)
    print("FLAG VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Total images: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Need review: {review}")
    print(f"Errors: {errors}")
    print("\nReport saved to:")
    print(REPORT_FILE)
    print("\nContact sheet saved to:")
    print(CONTACT_SHEET_FILE)

# This line is True only when you run THIS file directly
# (python validate_flags.py). If another file imports this one,
# the validation does not start automatically.
if __name__ == "__main__":

    validate_all_flags()