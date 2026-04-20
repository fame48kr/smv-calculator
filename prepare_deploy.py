"""
Run this script ONCE locally to prepare deployment assets.
Output:
  data/df_list.parquet, df_smv.parquet, df_proc.parquet, df_cat.parquet
  data/images.zip  (all thumbnails, ~8MB)
"""
import os, io, zipfile
import pandas as pd
import numpy as np
from PIL import Image


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-type object columns to string so pyarrow can save them."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace('nan', '')
    return df

EXCEL_PATH = r"D:\업무 효율화 develop 관련\smv_calculator\2.품셈표 등록된 스타일 V2 (2026.02.13)- 수정본.xlsx"
THUMB_SIZE = (180, 180)
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Export data sheets to parquet ─────────────────────────────
print("Exporting data sheets...")

df_list = pd.read_excel(EXCEL_PATH, sheet_name='LIST ', header=1)
df_list.columns = [str(c).strip() for c in df_list.columns]
df_list = df_list.rename(columns={
    '번호':'NO','이미지':'IMAGE','시즌':'SEASON','브랜드':'BRAND',
    '디비젼':'DIVISION','공장명':'FACTORY','AGE / TYPE':'GENDER',
    'CATEGORY#1':'CAT1','CATEGORY#2':'CAT2','CATEGORY#3':'CAT3',
    'CATEGORY#4':'CAT4','원단유형':'FABRIC_TYPE','YDS 당 중량':'YDS_WEIGHT',
})
df_list = df_list[df_list['STYLE'].notna()].reset_index(drop=False)
df_list = df_list.rename(columns={'index':'ORIG_IDX'})
# Normalize GENDER to Title Case so duplicates like "1. womens" / "1. Womens" are unified
df_list['GENDER'] = df_list['GENDER'].astype(str).str.strip().str.title()
_clean_df(df_list).to_parquet(f"{OUT_DIR}/df_list.parquet", index=False)
print(f"  df_list: {len(df_list)} rows")

df_smv = pd.read_excel(EXCEL_PATH, sheet_name='SMV_요약')
df_smv.columns = [str(c).strip() for c in df_smv.columns]
df_smv = df_smv.rename(columns={
    'Style#':'STYLE','Gender':'GENDER','Category#1':'CAT1','Category#2':'CAT2',
    'Category#3':'CAT3','Category#4':'CAT4',
    'Total SMV(분)':'TOTAL_SMV','공정수':'PROC_COUNT','사용기계':'MACHINES'
})
_clean_df(df_smv).to_parquet(f"{OUT_DIR}/df_smv.parquet", index=False)
print(f"  df_smv: {len(df_smv)} rows")

df_proc = pd.read_excel(EXCEL_PATH, sheet_name='yakjin_smv_style_process_popup')
df_proc.columns = [str(c).strip() for c in df_proc.columns]
df_proc = df_proc.rename(columns={
    'No.':'NO','Style#':'STYLE','Gender':'GENDER',
    '기본공정':'PROCESS','기계명':'MACHINE',
    '가로스펙(W_SPEC)':'W_SPEC','세로스펙(H_SPEC)':'H_SPEC',
    '중간사이즈(M_SIZE)':'M_SIZE',
    'Category #1':'CAT1','Category #2':'CAT2',
    'Category #3':'CAT3','Category #4':'CAT4','GSD SMV':'SMV'
})
_clean_df(df_proc).to_parquet(f"{OUT_DIR}/df_proc.parquet", index=False)
print(f"  df_proc: {len(df_proc)} rows")

df_cat_raw = pd.read_excel(EXCEL_PATH, sheet_name='카테고리', header=None)
df_cat = df_cat_raw.iloc[1:].copy()
df_cat.columns = ['NO','TYPE','SEQ','CAT1','CAT2','CAT3','CAT4']
df_cat = df_cat[df_cat['CAT1'].notna()].reset_index(drop=True)
_clean_df(df_cat).to_parquet(f"{OUT_DIR}/df_cat.parquet", index=False)
print(f"  df_cat: {len(df_cat)} rows")

# ── 2. Extract & compress all thumbnails → images.zip ─────────────
print("\nExtracting thumbnails...")
import xml.etree.ElementTree as ET

NS = {
    'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
    'a':   'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':   'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}

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

    # Build index: df_index → fname
    idx_map = {}
    for tag in ('xdr:twoCellAnchor', 'xdr:oneCellAnchor'):
        for anchor in root.findall(tag, NS):
            from_el = anchor.find('xdr:from', NS)
            if from_el is None: continue
            row_from_el = from_el.find('xdr:row', NS)
            if row_from_el is None: continue
            row_from = int(row_from_el.text)
            # Use center row (avg of from+to) to handle images straddling row boundaries
            to_el = anchor.find('xdr:to', NS)
            row_to_el = to_el.find('xdr:row', NS) if to_el is not None else None
            row_to = int(row_to_el.text) if row_to_el is not None else row_from
            df_index = round((row_from + row_to) / 2) - 2
            blip = anchor.find('.//a:blip', NS)
            if blip is None: continue
            rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            fname = rid_to_file.get(rid)
            if fname and fname in all_files:
                idx_map[df_index] = fname

    print(f"  Found {len(idx_map)} images in Excel")

    zip_path = f"{OUT_DIR}/images.zip"
    count = 0
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
        for df_idx, fname in idx_map.items():
            try:
                raw = z.read(fname)
                img = Image.open(io.BytesIO(raw)).convert('RGB')
                img.thumbnail(THUMB_SIZE, Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=70, optimize=True)
                out_zip.writestr(f"{df_idx}.jpg", buf.getvalue())
                count += 1
                if count % 200 == 0:
                    print(f"  {count}/{len(idx_map)} done...")
            except Exception as e:
                print(f"  Skip {df_idx}: {e}")

zip_size = os.path.getsize(zip_path) / 1024 / 1024
print(f"\nDone! {count} thumbnails → {zip_path} ({zip_size:.1f} MB)")
print("\nNext steps:")
print("  1. Upload data/images.zip to Google Drive (share → anyone with link)")
print("  2. Push data/*.parquet + all .py files to GitHub")
print("  3. Deploy on Streamlit Cloud")
