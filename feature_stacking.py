#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
概要:
高次元衛星データ(PlanetScope)および補助データのスタッキング処理。
計算精度維持のため、すべての出力は float32 に統一し、NoData値を -9999.0 で定義する。
"""

import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Path Configurations ---
# 外部ストレージ(HDD)およびプロジェクトディレクトリの定義
EXTERNAL_MOUNT = ""
RAW_DATA_DIR = os.path.join(EXTERNAL_MOUNT, "data/PlanetScope_Tokyo_2023")

# 処理済みデータおよび中間出力
PREPROCESSED_DIR = "../../data/02_preprocessed"
AUX_DIR = "../../data/05_aux_rasters"
OUTPUT_DIR = "../../data/03_feature_stack"

# 具体的なファイルパス
MASTER_REF = os.path.join(PREPROCESSED_DIR, "tokyo_planet_mosaic_FINAL_MEDIAN.tif")
AUX_DEM = os.path.join(AUX_DIR, "DEM.tif")
AUX_NL = os.path.join(AUX_DIR, "Nightlights.tif")
FINAL_STACK_OUT = os.path.join(OUTPUT_DIR, "tokyo_feature_stack.tif")

# 一時処理用ディレクトリ
TEMP_ALIGN = os.path.join(PREPROCESSED_DIR, "temp_aligned_seasonal/")
TEMP_STACK = os.path.join(PREPROCESSED_DIR, "temp_stacking/")

# --- Technical Specs ---
DEFAULT_NODATA = 0
OUTPUT_NODATA = -9999.0
INTERNAL_DTYPE = 'float32'  # 演算精度確保のための統一型
MAX_WORKERS = mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1

# --- Indices & Band Mapping ---
WINTER_MONTHS = [12, 1, 2]
WINTER_COMPOSITE_OUT = os.path.join(TEMP_STACK, "aligned_WINTER_composite_7band.tif")
SELECTED_BANDS = [1, 3, 4, 5, 6, 7, 8] # 出力対象波段

# チャンネルインデックス定義 (1-based)
RED_IDX = 6
NIR_IDX = 8
GREEN_IDX_7B = 3
RED_IDX_7B = 5
NIR_IDX_7B = 7

# Utility Functions


def find_udm2_for_composite(composite_path):
    """対応するUDM2(品質マスク)ファイルの検索"""
    udm2_path = composite_path.replace("_composite.tif", "_composite_udm2.tif")
    return udm2_path if os.path.exists(udm2_path) else None

def find_tifs_in_subfolder(subfolder_path):
    """ディレクトリ内の有効なGeoTIFFファイルの抽出"""
    search_pattern = os.path.join(subfolder_path, "*_composite.tif")
    return [p for p in glob.glob(search_pattern) if not p.lower().endswith("_udm2.tif")]

def align_rasterio(input_raster, output_raster, master_crs, master_transform, master_width, master_height, resampling_method, nodata_val):
    """マスターデータへの空間リファレンス統合"""
    try:
        with rasterio.open(input_raster) as src:
            src_crs, src_transform = src.crs, src.transform
            src_nodata_val = src.nodata if src.nodata is not None else nodata_val

            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': master_crs, 'transform': master_transform,
                'width': master_width, 'height': master_height,
                'nodata': nodata_val, 'compress': 'lzw'
            })
            with rasterio.open(output_raster, 'w', **dst_profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        src_nodata=src_nodata_val, 
                        dst_nodata=nodata_val,
                        dst_transform=master_transform,
                        dst_crs=master_crs,
                        resampling=resampling_method,
                        num_threads=4 
                    )
        return True
    except Exception as e:
        print(f"Error [Aligning]: {os.path.basename(input_raster)} - {e}")
        return False

# Processing Kernels

def process_block_ndvi_only(ji, window, aligned_files, udm2_files):
    """ブロック単位でのNDVIメディアン合成処理"""
    nodata_block = np.full((window.height, window.width), OUTPUT_NODATA, dtype=np.float32)
    stack_buffer = []
    
    for idx, f in enumerate(aligned_files):
        try:
            with rasterio.open(f) as src:
                red = src.read(RED_IDX, window=window).astype(np.float32)
                nir = src.read(NIR_IDX, window=window).astype(np.float32)
                mask = (red == DEFAULT_NODATA) | (nir == DEFAULT_NODATA)
            
            udm2_f = udm2_files[idx]
            if udm2_f:
                with rasterio.open(udm2_f) as u_src:
                    mask |= (u_src.read(1, window=window) == 0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red + 1e-9)
            ndvi[mask] = np.nan
            stack_buffer.append(ndvi)
        except: continue 
        
    if not stack_buffer: return ji, nodata_block
    
    try:
        combined = np.stack(stack_buffer, axis=0)
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            res = np.nanmedian(combined, axis=0)
    except: res = nodata_block
    
    return ji, np.nan_to_num(res, nan=OUTPUT_NODATA).astype(np.float32)

def process_block_max_ndvi_7band(ji, window, aligned_files, udm2_files):
    """Max-NDVIクライテリアに基づく7バンド合成処理"""
    out_7b = np.full((len(SELECTED_BANDS), window.height, window.width), OUTPUT_NODATA, dtype=np.float32)
    max_ndvi = np.full((window.height, window.width), -np.inf, dtype=np.float32)
    
    for idx, f in enumerate(aligned_files):
        try:
            with rasterio.open(f) as src:
                r_arr = src.read(RED_IDX, window=window).astype(np.float32)
                n_arr = src.read(NIR_IDX, window=window).astype(np.float32)
                bands = src.read(SELECTED_BANDS, window=window).astype(np.float32)
                mask = (r_arr == DEFAULT_NODATA) | (n_arr == DEFAULT_NODATA)
            
            udm2_f = udm2_files[idx]
            if udm2_f:
                with rasterio.open(udm2_f) as u_src:
                    mask |= (u_src.read(1, window=window) == 0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                val = (n_arr - r_arr) / (n_arr + r_arr + 1e-9)
            val[mask] = -np.inf
            
            update = val > max_ndvi
            if np.any(update):
                out_7b[:, update] = bands[:, update]
                max_ndvi[update] = val[update]
        except: continue
    return ji, out_7b

# High-level Compositing Functions

def composite_aligned_batch_ndvi_only(aligned_folder, output_path, master_profile):
    """NDVI単層合成の実行管理"""
    tifs = find_tifs_in_subfolder(aligned_folder)
    if not tifs: return False
    udms = [find_udm2_for_composite(p) for p in tifs]
    
    with rasterio.open(tifs[0]) as src:
        windows = list(src.block_windows(1))
        meta = src.profile.copy()
    
    meta.update(count=1, dtype=INTERNAL_DTYPE, nodata=OUTPUT_NODATA, compress='lzw', bigtiff='YES')
    
    try:
        with rasterio.open(output_path, 'w', **meta) as dst:
            tasks = [(w[0], w[1], tifs, udms) for w in windows]
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
                for future in tqdm(as_completed([pool.submit(process_block_ndvi_only, *t) for t in tasks]), 
                                  total=len(tasks), desc="NDVI Comp", leave=False):
                    ji, res = future.result()
                    target_win = next(w[1] for w in windows if w[0] == ji)
                    dst.write(res, 1, window=target_win)
        return True
    except: return False

def composite_aligned_batch_max_ndvi_7band(aligned_folder, output_path, master_profile):
    """多バンド(7B)合成の実行管理"""
    tifs = find_tifs_in_subfolder(aligned_folder)
    if not tifs: return False
    udms = [find_udm2_for_composite(p) for p in tifs]
    
    with rasterio.open(tifs[0]) as src:
        windows = list(src.block_windows(1))
        meta = src.profile.copy()
    
    meta.update(count=len(SELECTED_BANDS), dtype=INTERNAL_DTYPE, nodata=OUTPUT_NODATA, compress='lzw', bigtiff='YES')
    
    try:
        with rasterio.open(output_path, 'w', **meta) as dst:
            tasks = [(w[0], w[1], tifs, udms) for w in windows]
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
                for future in tqdm(as_completed([pool.submit(process_block_max_ndvi_7band, *t) for t in tasks]), 
                                  total=len(tasks), desc="7B MaxNDVI Comp", leave=False):
                    ji, res = future.result()
                    target_win = next(w[1] for w in windows if w[0] == ji)
                    dst.write(res, window=target_win)
        return True
    except: return False

def stack_rasters_rasterio(files_to_stack, output_path, master_profile):
    """最終フィーチャースタックの生成 (Grid I/O方式)"""
    print(f"\n[Process 5/5] Final Stacking (Force Float32)...")
    
    readers = [rasterio.open(f) for f in files_to_stack]
    total_bands = sum(r.count for r in readers)

    master_profile.update(
        count=total_bands, driver='GTiff', compress='lzw', bigtiff='YES',
        tiled=True, blockxsize=1024, blockysize=1024,
        dtype=INTERNAL_DTYPE, nodata=OUTPUT_NODATA
    )
    
    w, h = master_profile['width'], master_profile['height']
    tile_size = 1024
    windows = [Window(c, r, min(tile_size, w - c), min(tile_size, h - r)) 
               for r in range(0, h, tile_size) for c in range(0, w, tile_size)]
            
    print(f"Total tiles: {len(windows)} ({tile_size}x{tile_size})")

    with rasterio.open(output_path, 'w', **master_profile) as dst:
        cursor = 1
        for src in tqdm(readers, desc="Stacking layers"):
            n_bands = src.count
            for win in windows:
                try:
                    data = src.read(window=win).astype(INTERNAL_DTYPE)
                    dst.write(data, window=win, indexes=list(range(cursor, cursor + n_bands)))
                except Exception as e:
                    print(f"Window Error: {win} - {e}")
            cursor += n_bands

    for r in readers: r.close()

def calculate_indices_from_7band(composite_tif, temp_folder, master_profile, season="SUMMER"):
    """正規化指数(NDVI/NDWI)の算出"""
    idx_step = 2 if season == "SUMMER" else 4
    print(f"[Process {idx_step}/5] Calculating Indices ({season})...")
    
    out_ndvi = os.path.join(temp_folder, f"aligned_{season}_ndvi.tif")
    out_ndwi = os.path.join(temp_folder, f"aligned_{season}_ndwi.tif")
    
    if os.path.exists(out_ndvi) and os.path.exists(out_ndwi):
        return out_ndvi, out_ndwi

    with rasterio.open(composite_tif) as src:
        g, r, n = [src.read(i).astype(np.float32) for i in [GREEN_IDX_7B, RED_IDX_7B, NIR_IDX_7B]]
    
    meta = master_profile.copy()
    meta.update(count=1, dtype=INTERNAL_DTYPE, nodata=OUTPUT_NODATA, compress='lzw')
    
    # NDVI
    ndvi = np.nan_to_num((n - r) / (n + r + 1e-9), nan=OUTPUT_NODATA)
    with rasterio.open(out_ndvi, 'w', **meta) as d: d.write(ndvi, 1)
    
    # NDWI
    ndwi = np.nan_to_num((g - n) / (g + n + 1e-9), nan=OUTPUT_NODATA)
    with rasterio.open(out_ndwi, 'w', **meta) as d: d.write(ndwi, 1)
    
    return out_ndvi, out_ndwi

def group_raw_files_by_month(root_dir):
    """月別のソースファイルグルーピング"""
    groups = {m: [] for m in range(1, 13)}
    pattern = re.compile(r'Tokyo_(\d{4})(\d{2})\d{2}')
    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path): continue
        match = pattern.match(folder)
        if match:
            m = int(match.groups()[1])
            tifs = find_tifs_in_subfolder(path)
            if tifs: groups[m].extend(tifs)
    return groups


# Main Pipeline


def create_full_feature_stack_automated():
    print("Workflow Execution: Feature Stack Creation (v1.1)")
    
    if not all(os.path.exists(f) for f in [MASTER_REF, AUX_DEM, AUX_NL]):
        print("Required base files missing.")
        return
        
    os.makedirs(TEMP_ALIGN, exist_ok=True)
    os.makedirs(TEMP_STACK, exist_ok=True)
    
    with rasterio.open(MASTER_REF) as src:
        m_crs, m_trans = src.crs, src.transform
        m_w, m_h = src.width, src.height
        m_profile = src.profile.copy()
        
    final_list = [MASTER_REF]
    
    # 1. Auxiliary Data Alignment
    dem_aligned = os.path.join(TEMP_STACK, "aligned_dem.tif")
    nl_aligned = os.path.join(TEMP_STACK, "aligned_nightlights.tif")
    if not os.path.exists(dem_aligned):
        align_rasterio(AUX_DEM, dem_aligned, m_crs, m_trans, m_w, m_h, Resampling.bilinear, OUTPUT_NODATA)
    final_list.extend([dem_aligned, nl_aligned])
    
    # 2. Summer Indices
    ndvi_s, ndwi_s = calculate_indices_from_7band(MASTER_REF, TEMP_STACK, m_profile, "SUMMER")
    final_list.extend([ndvi_s, ndwi_s])
    
    # 3. Monthly NDVI Compositions
    monthly_jobs = group_raw_files_by_month(RAW_DATA_DIR)
    winter_raws = []
    
    for m in range(1, 13):
        files = monthly_jobs[m]
        if m in WINTER_MONTHS: winter_raws.extend(files)
        if not files: continue
        
        m_ndvi_out = os.path.join(TEMP_STACK, f"aligned_NDVI_{m:02d}.tif")
        if os.path.exists(m_ndvi_out):
            final_list.append(m_ndvi_out)
            continue
            
        for f in files:
            target = os.path.join(TEMP_ALIGN, os.path.basename(f))
            if not os.path.exists(target):
                align_rasterio(f, target, m_crs, m_trans, m_w, m_h, Resampling.cubic, DEFAULT_NODATA)
                u = find_udm2_for_composite(f)
                if u: align_rasterio(u, os.path.join(TEMP_ALIGN, os.path.basename(u)), m_crs, m_trans, m_w, m_h, Resampling.nearest, 255)
        
        if composite_aligned_batch_ndvi_only(TEMP_ALIGN, m_ndvi_out, m_profile):
            final_list.append(m_ndvi_out)

    # 4. Winter Season Composite
    if winter_raws:
        if not os.path.exists(WINTER_COMPOSITE_OUT):
            composite_aligned_batch_max_ndvi_7band(TEMP_ALIGN, WINTER_COMPOSITE_OUT, m_profile)
        final_list.append(WINTER_COMPOSITE_OUT)
        ndvi_w, ndwi_w = calculate_indices_from_7band(WINTER_COMPOSITE_OUT, TEMP_STACK, m_profile, "WINTER")
        final_list.extend([ndvi_w, ndwi_w])

    # 5. Final Assembly
    if os.path.exists(FINAL_STACK_OUT): os.remove(FINAL_STACK_OUT)
    stack_rasters_rasterio(final_list, FINAL_STACK_OUT, m_profile)
    print(f"\nProcessing Complete. Output: {FINAL_STACK_OUT}")

if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except: pass
    create_full_feature_stack_automated()