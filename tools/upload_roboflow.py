"""Upload a generated batch to Roboflow, keeping its train / valid / test split.

In a PowerShell terminal (the key stays in that terminal only):

    $env:ROBOFLOW_API_KEY = "your private API key"
    python tools/upload_roboflow.py dataset/batch_demo/roboflow_detect --project flags-batch-demo

The project is created in your default workspace if it does not exist yet.
Uses `roboflow auth login` credentials instead when ROBOFLOW_API_KEY is not set.
Note: on Roboflow's free plan, projects are public.
"""
import argparse
import os
from pathlib import Path

import roboflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="a roboflow_detect folder (train/valid/test + data.yaml)")
    ap.add_argument("--project", required=True, help="project name in Roboflow")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    folder = Path(a.folder)
    counts = {sp: len(list((folder / sp / "images").glob("*.jpg"))) for sp in ("train", "valid", "test")}
    if not (folder / "data.yaml").exists() or not sum(counts.values()):
        raise SystemExit(f"{folder} is not a roboflow_detect folder (no data.yaml or no images)")
    print(f"Uploading {sum(counts.values())} images {counts} from {folder} to project '{a.project}'")

    key = (os.environ.get("ROBOFLOW_API_KEY") or "").strip().strip("\"'").strip()
    if key:
        if key.upper() in ("YOUR_KEY", "YOUR KEY") or " " in key:
            raise SystemExit("ROBOFLOW_API_KEY still holds the placeholder text or spaces - "
                             "paste the real Private API Key between the quotes.")
        print(f"Using ROBOFLOW_API_KEY from this terminal ({len(key)} characters)")
    try:
        rf = roboflow.Roboflow(api_key=key) if key else roboflow.Roboflow()
    except RuntimeError as e:
        if "does not exist" in str(e) or "401" in str(e):
            raise SystemExit("Roboflow rejected the key. Copy the PRIVATE API Key (not the "
                             "Publishable one) from app.roboflow.com -> Settings -> API Keys - the new "
                             "one if you regenerated it - and set ROBOFLOW_API_KEY again.") from None
        raise
    ws = rf.workspace()
    print(f"Workspace: {ws.url}")
    ws.upload_dataset(str(folder), a.project, num_workers=a.workers,
                      project_type="object-detection", batch_name=folder.parent.name, num_retries=2)
    print(f"\nDone. Open https://app.roboflow.com/{ws.url} -> {a.project} -> Dataset to check the images.")


if __name__ == "__main__":
    main()
