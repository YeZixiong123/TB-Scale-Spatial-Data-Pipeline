#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- RFモデルのメモリ効率を考慮したバッチ処理の実装
- 大規模ラスターに対するTileベースのストリーミング書き込み (r+ mode)
- シャドウ・ソーラーパネルのヒューリスティック補正
"""

import geopandas as gpd
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from tqdm import tqdm
from rasterio.windows import Window
import os
import sys
import pickle
import joblib
import hashlib
import gc

# --- Environment Settings ---
# Input Resources
VECTOR_DATA = "../../data/04_vector_inputs/training_samples.shp"
INPUT_STACK = "../../data/03_feature_stack/tokyo_feature_stack.tif"
OUTPUT_MAP = "../../3_output/1_pixel_based/pixel_classification_result_v2.5.tif"

# Cache & Model Storage
CACHE_DIR = "../../3_output/1_pixel_based/cache"
MODEL_REPO = os.path.join(CACHE_DIR, "models_v2.2")
COMBO_LIST = os.path.join(CACHE_DIR, "combinations_cache.pkl")

#  Hyperparameters 
CLASS_URBAN = 2
CLASS_SOLAR = 12
CLASS_SHADOW_TMP = 16
SOLAR_THRESHOLD = 0.60

# Computing Resources
MODEL_BATCH_SIZE = 15
TILE_H = 256  # I/O負荷とメモリのバランスを考慮したタイルサイズ

def get_model_filename(combo):
    """
    モデルコンビネーションに対応するファイルパスの生成
    100文字を超える場合はMD5ハッシュで衝突回避
    """
    c_str = "_".join(combo)
    if len(c_str) > 100:
        safe_name = hashlib.md5(c_str.encode()).hexdigest()
    else:
        safe_name = c_str
    return os.path.join(MODEL_REPO, f"rf_model_{safe_name}.pkl")

print(f"[INIT] Classification v2.5 - Streaming Mode Enabled")

# 1. Validation
if not os.path.exists(COMBO_LIST):
    sys.exit(f"[FATAL ERROR] Missing required cache: {COMBO_LIST}")

with open(COMBO_LIST, 'rb') as f:
    pixel_combos = list(pickle.load(f))

n_combos = len(pixel_combos)
print(f"[INFO] Loaded unique pixel combinations: {n_combos}")

# 2. Output Buffer Initialization
print(f"[FILE] Initializing output GeoTIFF: {OUTPUT_MAP}")
with rasterio.open(INPUT_STACK) as src:
    meta = src.profile.copy()
    H, W = src.height, src.width
    bands = src.count
    
    meta.update(
        dtype='int32',
        count=1,
        nodata=-1,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    )
    
    with rasterio.open(OUTPUT_MAP, 'w', **meta) as dst:
        pass # Header only

# 3. Batch Inference Execution
chunks = [pixel_combos[i:i + MODEL_BATCH_SIZE] for i in range(0, n_combos, MODEL_BATCH_SIZE)]
print(f"[EXEC] Total batches to process: {len(chunks)}")

# I/O効率のためファイルハンドルを保持
with rasterio.open(OUTPUT_MAP, 'r+') as dst:
    
    for b_idx, current_batch in enumerate(chunks):
        print(f"\n[BATCH {b_idx + 1}/{len(chunks)}] Loading model subsets...")
        
        #  A. Model Loading 
        active_models = {}
        for c in tqdm(current_batch, desc="Deserializing", leave=False):
            m_path = get_model_filename(c)
            if os.path.exists(m_path):
                try:
                    active_models[c] = joblib.load(m_path)
                except: continue
        
        if not active_models:
            continue

        #  B. Spatial Scanning 
        with rasterio.open(INPUT_STACK) as src:
            for r_start in tqdm(range(0, H, TILE_H), desc=f"Scanline {b_idx+1}"):
                curr_h = min(TILE_H, H - r_start)
                win = Window(0, r_start, W, curr_h)
                
                # Input read
                data = src.read(window=win).astype(np.float32)
                
                # Null handle
                data[data == src.nodata] = np.nan
                data[np.isclose(data, -9999, atol=1e-2)] = np.nan
                
                flat_data = np.moveaxis(data, 0, -1).reshape(-1, bands)
                
                # C. Combinatorial Mapping
                # 該当バッチのモデルが存在するピクセルのみを抽出
                px_map = defaultdict(list)
                
                for idx, px in enumerate(flat_data):
                    if np.all(np.isnan(px)): continue
                    v_mask = ~np.isnan(px)
                    combo = tuple([f"B{i+1}" for i in range(bands) if v_mask[i]])
                    
                    if combo in active_models:
                        px_map[combo].append((idx, px[v_mask]))

                if not px_map:
                    continue
                
                # D. Local Inference & Rules
                patch_results = {} 
                
                for combo, items in px_map.items():
                    clf = active_models[combo]
                    X = np.array([v for _, v in items])
                    indices = [idx for idx, _ in items]
                    
                    probs = clf.predict_proba(X)
                    max_idx = np.argmax(probs, axis=1)
                    labels = clf.classes_[max_idx]
                    confs = np.max(probs, axis=1)
                    
                    # Rule: Solar panel confidence filter
                    if CLASS_SOLAR in clf.classes_:
                        m_solar = (labels == CLASS_SOLAR) & (confs < SOLAR_THRESHOLD)
                        labels[m_solar] = CLASS_URBAN
                    
                    # Rule: Shadow class remapping
                    if CLASS_SHADOW_TMP in clf.classes_:
                        labels[labels == CLASS_SHADOW_TMP] = CLASS_URBAN
                    
                    for l_idx, lbl in zip(indices, labels):
                        patch_results[l_idx] = lbl

                # E. In-place Disk Update
                if patch_results:
                    # 現状の値を読み出し、新規ラベルで上書き
                    buffer = dst.read(1, window=win)
                    flat_buf = buffer.flatten()
                    
                    for l_idx, lbl in patch_results.items():
                        flat_buf[l_idx] = lbl
                        
                    dst.write(flat_buf.reshape(curr_h, W), 1, window=win)

        # Resource management
        del active_models
        gc.collect()

print(f"\n[FINISH] Classification map exported: {OUTPUT_MAP}")
print(f"[STATUS] Resource cleanup completed. Task idle.")