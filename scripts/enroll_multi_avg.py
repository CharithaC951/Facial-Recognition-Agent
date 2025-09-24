# scripts/enroll_multi_avg.py
import cv2, numpy as np, glob
from pathlib import Path
from insightface.app import FaceAnalysis

IN_GLOB = "reference_faces/patient_001/*.jpg"
OUT_NPY = "reference_faces/patient_001_m_arcface.npy"

app = FaceAnalysis(name="buffalo_l")      # RetinaFace + ArcFace
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode for Cloud Run

embs = []
for path in glob.glob(IN_GLOB):
    img = cv2.imread(path)
    faces = app.get(img)
    if not faces:
        print(f"[skip] no face in {path}")
        continue
    face = max(faces, key=lambda f: f.det_score)
    embs.append(face.normed_embedding.astype("float32"))

if not embs:
    raise RuntimeError("No faces found in the enrollment images.")

# Average then re-normalize
ref = np.mean(np.stack(embs, axis=0), axis=0)
ref /= (np.linalg.norm(ref) + 1e-12)
Path(OUT_NPY).parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_NPY, ref.astype("float32"))
print("Saved averaged reference embedding to:", OUT_NPY)
