"""
Cloud asset loader — downloads images.zip from Google Drive once per session.
Used automatically when running on Streamlit Cloud (DATA_DIR env var set).
"""
import os, io, zipfile, tempfile
import streamlit as st

# Set IMAGES_GDRIVE_ID in Streamlit Cloud secrets
_CACHE_DIR = tempfile.gettempdir()
_ZIP_PATH  = os.path.join(_CACHE_DIR, "smv_images.zip")
_EXTRACTED: dict[int, bytes] | None = None


@st.cache_resource(show_spinner="Downloading image assets...")
def _download_zip() -> str:
    """Download images.zip from Google Drive if not already cached."""
    gdrive_id = st.secrets.get("IMAGES_GDRIVE_ID", "")
    if not gdrive_id:
        return ""
    if os.path.exists(_ZIP_PATH):
        return _ZIP_PATH
    import gdown
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, _ZIP_PATH, quiet=False)
    return _ZIP_PATH


@st.cache_resource(show_spinner="Loading image index...")
def load_cloud_images() -> dict[int, bytes]:
    """Returns {df_idx: jpeg_bytes} for all thumbnails."""
    zip_path = _download_zip()
    if not zip_path or not os.path.exists(zip_path):
        return {}
    images = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            try:
                idx = int(name.replace(".jpg", ""))
                images[idx] = z.read(name)
            except ValueError:
                pass
    return images


def get_cloud_image(orig_idx: int) -> bytes | None:
    imgs = load_cloud_images()
    return imgs.get(orig_idx)
