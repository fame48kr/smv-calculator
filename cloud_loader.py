"""
Cloud asset loader — downloads images.zip from Google Drive once per session.
Used automatically when running on Streamlit Cloud (DATA_DIR env var set).

To force re-download after uploading a new images.zip to Google Drive:
  → Update IMAGES_VERSION in Streamlit Cloud Secrets (e.g. "1" → "2")
"""
import os, io, zipfile, tempfile
import streamlit as st

_CACHE_DIR = tempfile.gettempdir()


def _zip_path(version: str) -> str:
    """Return a version-stamped path so each version gets its own file."""
    return os.path.join(_CACHE_DIR, f"smv_images_v{version}.zip")


@st.cache_resource(show_spinner="Downloading image assets...")
def _download_zip(gdrive_id: str, version: str) -> str:
    """Download images.zip from Google Drive.
    gdrive_id + version are cache-key args — changing either forces re-download.
    """
    path = _zip_path(version)
    if not os.path.exists(path):
        import gdown
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, path, quiet=False)
    return path


@st.cache_resource(show_spinner="Loading image index...")
def load_cloud_images(version: str = "1") -> dict[int, bytes]:
    """Returns {df_idx: jpeg_bytes} for all thumbnails."""
    try:
        gdrive_id = st.secrets.get("IMAGES_GDRIVE_ID", "")
        if not gdrive_id:
            return {}
        zip_path = _download_zip(gdrive_id, version)
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
    except Exception:
        return {}  # images unavailable — app still works without them


def get_cloud_image(orig_idx: int) -> bytes | None:
    version = st.secrets.get("IMAGES_VERSION", "1")
    imgs = load_cloud_images(version=version)
    return imgs.get(orig_idx)
