"""Download the biomedical classification datasets used by OGKD.

Datasets are pulled from the publicly released BiomedCoOp benchmark on the
HuggingFace Hub (prior work). 

Output directory can be overridden with the DATA_ROOT environment variable
(default: /workspace/dataset_raw).
"""
import os
import pathlib
import zipfile

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "TahaKoleilat/BiomedCoOp"  # public dataset release from the BiomedCoOp baseline

# Full benchmark
FILES = [
    "BTMRI.zip", "BUSI.zip", "CHMNIST.zip", "COVID_19.zip",
    "CTKidney.zip", "DermaMNIST.zip", "KneeXray.zip",
    "Kvasir.zip", "LungColon.zip", "OCTMNIST.zip", "RETINA.zip",
]


target_root = pathlib.Path(os.environ.get("DATA_ROOT", "/workspace/dataset_raw"))
cache_dir = os.environ.get("HF_CACHE_DIR", "hf_cache")
target_root.mkdir(parents=True, exist_ok=True)

for fname in tqdm(FILES):
    cached_path = hf_hub_download(
        repo_id=REPO,
        filename=fname,
        repo_type="dataset",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    with zipfile.ZipFile(cached_path) as zf:
        zf.extractall(target_root / fname.replace(".zip", ""))
    print("✓", fname, "→", target_root / fname.replace(".zip", ""))
