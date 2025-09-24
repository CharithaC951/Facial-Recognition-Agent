import io, os, tempfile, shutil, subprocess
from typing import List, Optional, Tuple
from flask import Flask, request, jsonify, Response
import requests
from supabase import create_client, Client

# ------------ Config via env ------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # server-side key only
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "reference-assets")

# If your images live under a predictable prefix per patient, set this template:
# Example outcome: reference_faces/patient_001/
IMAGE_PREFIX_TEMPLATE = os.environ.get("IMAGE_PREFIX_TEMPLATE", "reference_faces/{patient_id}/")

# Optional: if you already store explicit image paths/urls in DB, you can ignore the template
# and send them in the request (see endpoint docs below).

# Where your script expects to read/write (hard-coded in your script):
SCRIPT_IN_DIR = "reference_faces/patient_001"
SCRIPT_OUT_REL = "reference_faces/patient_001_m_arcface.npy"

# Path to your script inside the container/source tree:
SCRIPT_PATH = os.environ.get("SCRIPT_PATH", "scripts/enroll_multi_avg.py")

# ---------------------------------------
app = Flask(__name__)
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def _list_storage_by_prefix(prefix: str) -> List[str]:
    """List files under a prefix in the default bucket."""
    items = sb.storage.from_(STORAGE_BUCKET).list(prefix, {"limit": 1000})
    # Only files (not subfolders); extend to recurse if you use subfolders
    return [f"{prefix.rstrip('/')}/{it['name']}" for it in (items or []) if it.get("name")]

def _download_from_storage_to(path_in_bucket: str, dest_file: str):
    blob = sb.storage.from_(STORAGE_BUCKET).download(path_in_bucket)
    with open(dest_file, "wb") as f:
        f.write(blob)

def _download_from_url_to(url: str, dest_file: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest_file, "wb") as f:
        f.write(r.content)

def _stage_images(temp_root: str, storage_paths: List[str], urls: List[str]) -> int:
    """Create reference_faces/patient_001/ under temp_root and fill it with images."""
    in_dir = os.path.join(temp_root, SCRIPT_IN_DIR)
    os.makedirs(in_dir, exist_ok=True)
    count = 0

    for p in storage_paths:
        try:
            out_path = os.path.join(in_dir, f"sp_{count}.jpg")
            _download_from_storage_to(p, out_path)
            count += 1
        except Exception:
            continue

    for u in urls:
        try:
            out_path = os.path.join(in_dir, f"url_{count}.jpg")
            _download_from_url_to(u, out_path)
            count += 1
        except Exception:
            continue

    return count

def _run_script_in(temp_root: str) -> str:
    """Run your script with cwd=temp_root so relative paths match your IN_GLOB/OUT_NPY."""
    result = subprocess.run(
        ["python", SCRIPT_PATH],
        cwd=temp_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result.stdout

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/enroll")
def enroll():
    """
    JSON body options:

    (A) Simplest: just patient_id (we infer storage prefix)
    {
      "patient_id": "patient_001"
    }

    (B) Or provide explicit lists:
    {
      "patient_id": "patient_001",
      "storage_paths": ["reference_faces/patient_001/1.jpg", ...],
      "image_urls": ["https://.../public/reference-assets/reference_faces/patient_001/2.jpg", ...]
    }
    """
    if not request.is_json:
        return jsonify(error="send JSON"), 400
    data = request.get_json()
    patient_id = data.get("patient_id")
    if not patient_id:
        return jsonify(error="patient_id is required"), 400

    storage_paths = data.get("storage_paths") or []
    image_urls = data.get("image_urls") or []

    # If caller didn't pass images, build a prefix and list files from storage
    if not storage_paths and not image_urls:
        prefix = IMAGE_PREFIX_TEMPLATE.format(patient_id=patient_id)
        storage_paths = _list_storage_by_prefix(prefix)

    if not storage_paths and not image_urls:
        return jsonify(error="no images found for this patient"), 404

    # Work in a temp folder (Cloud Run’s filesystem is ephemeral)
    temp_root = tempfile.mkdtemp(prefix=f"enroll_{patient_id}_")  # under /tmp by default
    try:
        n = _stage_images(temp_root, storage_paths, image_urls)
        if n == 0:
            return jsonify(error="no usable images staged"), 422

        _ = _run_script_in(temp_root)

        npy_path = os.path.join(temp_root, SCRIPT_OUT_REL)
        if not os.path.exists(npy_path):
            return jsonify(error="output .npy not found after script run"), 500

        with open(npy_path, "rb") as f:
            npy_bytes = f.read()

        filename = f"{patient_id}_m_arcface.npy"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(npy_bytes, mimetype="application/octet-stream", headers=headers)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

if __name__ == "__main__":
    # For local testing:
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
