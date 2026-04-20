"""Image loader — local (Excel ZIP) or cloud (Google Drive ZIP), auto-detected."""
import os, zipfile, io
import xml.etree.ElementTree as ET
import streamlit as st
from PIL import Image

# Local path (ignored on Streamlit Cloud)
EXCEL_PATH = r"C:\Users\Admin\Desktop\2.품셈표 등록된 스타일 V2 (2026.02.13)- 수정본.xlsx"
THUMB_SIZE = (180, 180)

NS = {
    'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
    'a':   'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':   'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}

IS_CLOUD = not os.path.exists(EXCEL_PATH)


# ── Cloud mode ────────────────────────────────────────────────────
def _get_cloud_image(orig_idx: int) -> bytes | None:
    from cloud_loader import get_cloud_image
    return get_cloud_image(orig_idx)


def load_image_index() -> dict:
    """On cloud: no-op (images served directly by index). Local: build XML index."""
    if IS_CLOUD:
        try:
            from cloud_loader import load_cloud_images
            load_cloud_images()  # trigger download
        except Exception:
            pass  # images unavailable — app still works without them
        return {}
    return _load_image_index_local()


# ── Local mode ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading image index...")
def _load_image_index_local() -> dict:
    row_to_fname = {}
    with zipfile.ZipFile(EXCEL_PATH) as z:
        rels_data = z.read('xl/drawings/_rels/drawing1.xml.rels')
        rels_root = ET.fromstring(rels_data)
        rid_to_file = {
            rel.get('Id'): rel.get('Target').replace('../', 'xl/')
            for rel in rels_root
        }
        xml_data = z.read('xl/drawings/drawing1.xml')
        root = ET.fromstring(xml_data)
        all_files = set(z.namelist())
        for tag in ('xdr:twoCellAnchor', 'xdr:oneCellAnchor'):
            for anchor in root.findall(tag, NS):
                from_el = anchor.find('xdr:from', NS)
                if from_el is None: continue
                row_from_el = from_el.find('xdr:row', NS)
                if row_from_el is None: continue
                row_from = int(row_from_el.text)
                # Use center row (avg of from+to) to handle images that straddle row boundaries
                to_el = anchor.find('xdr:to', NS)
                row_to_el = to_el.find('xdr:row', NS) if to_el is not None else None
                row_to = int(row_to_el.text) if row_to_el is not None else row_from
                df_index = round((row_from + row_to) / 2) - 2
                blip = anchor.find('.//a:blip', NS)
                if blip is None: continue
                rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                fname = rid_to_file.get(rid)
                if fname and fname in all_files:
                    row_to_fname[df_index] = fname
    return row_to_fname


@st.cache_data(show_spinner=False, max_entries=300)
def _load_thumb_local(fname: str) -> bytes:
    with zipfile.ZipFile(EXCEL_PATH) as z:
        raw = z.read(fname)
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70, optimize=True)
    return buf.getvalue()


_LOCAL_INDEX: dict | None = None

def _get_local_index() -> dict:
    global _LOCAL_INDEX
    if _LOCAL_INDEX is None:
        _LOCAL_INDEX = _load_image_index_local()
    return _LOCAL_INDEX


# ── Public API ────────────────────────────────────────────────────
def get_image(orig_idx: int) -> bytes | None:
    if IS_CLOUD:
        return _get_cloud_image(orig_idx)
    fname = _get_local_index().get(orig_idx)
    if not fname:
        return None
    try:
        return _load_thumb_local(fname)
    except Exception:
        return None


def get_image_by_style(style_str: str, df_list) -> bytes | None:
    s = str(style_str).strip()
    match = df_list[df_list['STYLE'].astype(str).str.strip() == s]
    if match.empty:
        match = df_list[df_list['STYLE'].astype(str).str.contains(s[:8], na=False, regex=False)]
    if not match.empty:
        orig_idx = match.iloc[0].get('ORIG_IDX', -1)
        return get_image(orig_idx)
    return None
