#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimization Logic:
 Fiona Streaming I/O による skip 処理のボトルネック回避
ジオメトリの解析をスキップする Zero-copy 伝填による CPU 負荷の最小化
バッチ処理（Vectorized Mapping）による推論スループットの向上
"""

import os
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pickle
import gc
import time
import fiona
import joblib

# 警告制御: 大規模演算時の冗長な出力を抑制
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# --- 1. Path Configuration ---
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# データストレージと計算キャッシュの定義
DATA_ROOT = os.path.abspath(os.path.join(BASE_PATH, "../../3_output/2_object_based_EXP"))
CACHE_DIR = os.path.join(DATA_ROOT, "02_classified/V_GRID_cache_v2")
SRC_SEGMENTS = os.path.join(DATA_ROOT, "01_segments/segments_CLEANED_v2.gpkg")
OUT_VECTOR = os.path.join(DATA_ROOT, "02_classified/object_classification_result_v1.1.gpkg")

# モデル永続化パス
MODEL_REPO = os.path.join(DATA_ROOT, "02_classified/models")
PATH_RF_MAIN = os.path.join(MODEL_REPO, "rf_main.joblib")
PATH_RF_BACKUP = os.path.join(MODEL_REPO, "rf_backup.joblib")

# 特徴量セットの定義
SHAPE_FEATS = ["rectangularity", "elongation", "compactness", "shape_index", "num_vertices", "vertex_density"]

#  2. Operational Parameters 
BATCH_SIZE = 100000  # メモリ使用量とI/Oスループットのトレードオフ設定


def build_feature_matrix(result_chunk, band_cols):
    """キャッシュデータからの特徴量マトリックス再構成"""
    feats = result_chunk.get("features", [])
    if not feats: return pd.DataFrame()
    
    seg_ids = result_chunk.get("orig_idx_list", []) or result_chunk.get("mapped_orig_idx", [])
    shapes = result_chunk.get("shapes", {})
    
    # 幾何学的特徴量の結合
    shape_data = np.column_stack([shapes.get(f, [np.nan] * len(feats)) for f in SHAPE_FEATS])
    band_data = np.array(feats)
    
    if band_data.shape[0] != shape_data.shape[0]: 
        return pd.DataFrame()

    X = np.hstack([band_data, shape_data])
    df = pd.DataFrame(X, columns=band_cols + SHAPE_FEATS)
    df['label'] = result_chunk.get("labels", [])
    df['seg_id'] = seg_ids
    return df

def initialize_models():
    """推论モデルの初期化および予測ルックアップテーブルの構築"""
    if not os.path.exists(MODEL_REPO):
        os.makedirs(MODEL_REPO, exist_ok=True)
        
    print("[INFO] Initializing Inference Models...")
    
    rf_main, rf_backup = None, None
    
    # モデルの永続化チェック
    if os.path.exists(PATH_RF_MAIN) and os.path.exists(PATH_RF_BACKUP):
        print("[INFO] Loading pre-trained models from disk.")
        try:
            rf_main = joblib.load(PATH_RF_MAIN)
            rf_backup = joblib.load(PATH_RF_BACKUP)
        except Exception as e:
            print(f"[WARN] Failed to load models: {e}. Re-training required.")

    # 有効なキャッシュファイルのフィルタリング
    cache_files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) 
                   if f.endswith(".pkl") and os.path.getsize(os.path.join(CACHE_DIR, f)) > 1024]
    
    if not cache_files: raise FileNotFoundError("Target cache directory is empty.")

    with open(cache_files[0], 'rb') as f:
        meta = pickle.load(f)
        band_cols = [f"Band_{i+1}" for i in range(len(meta["features"][0]))]
        all_cols = band_cols + SHAPE_FEATS

    # モデル再学習プロセス (必要時のみ)
    if rf_main is None:
        print("[PROCESS] Starting model re-training phase.")
        x_list, y_list = [], []
        for p in tqdm(cache_files, desc="Training Set Construction"):
            try:
                with open(p, 'rb') as f: data = pickle.load(f)
                df = build_feature_matrix(data, band_cols)
                if df.empty: continue
                
                valid = (df['label'] > 0) & (df['label'] <= 255)
                if valid.any():
                    x_list.append(df.loc[valid, all_cols].fillna(0))
                    y_list.append(df.loc[valid, 'label'])
            except: continue
            
        if not x_list: raise ValueError("Insufficient training samples.")
        
        X_all = pd.concat(x_list)
        y_all = pd.concat(y_list)
        print(f"[INFO] Sample size: {len(X_all):,}")
        
        rf_main = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        rf_main.fit(X_all, y_all)
        
        rf_backup = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        rf_backup.fit(X_all[SHAPE_FEATS], y_all)
        
        joblib.dump(rf_main, PATH_RF_MAIN)
        joblib.dump(rf_backup, PATH_RF_BACKUP)
        
        del X_all, y_all
        gc.collect()
    
    # メイン予測用のルックアップテーブル構築 (計算負荷軽減のため)
    print("[PROCESS] Building Prediction Lookup Table...")
    preds_map = {}
    for p in tqdm(cache_files, desc="Map Mapping"):
        try:
            with open(p, 'rb') as f: data = pickle.load(f)
            df = build_feature_matrix(data, band_cols)
            if df.empty: continue
            
            preds = rf_main.predict(df[all_cols].fillna(0))
            preds_map.update(dict(zip(df['seg_id'].values.astype(int), preds.astype(int))))
        except: continue
        
    print(f"[SUCCESS] Lookup table indexed: {len(preds_map):,} records.")
    return rf_backup, preds_map

def execute_batch_inference(batch, rf_backup, preds_map):
    """
    バッチ単位でのクラス予測および属性の同期。
    ルックアップテーブルに不在のIDに対してはバックアップモデルを適用。
    """
    ids = []
    props_list = []
    
    for feat in batch:
        sid = feat['properties'].get('seg_id')
        ids.append(sid if sid is not None else -1)
        props_list.append(feat['properties'])
    
    ids = np.array(ids, dtype=int)
    
    # ベクトル化された ID マッピング
    pred_series = pd.Series(ids).map(preds_map)
    
    # ルックアップ失敗(NaN)に対するバックアップ予測
    missing_mask = pred_series.isna()
    if missing_mask.any():
        m_indices = np.where(missing_mask)[0]
        shape_X = np.array([[float(props_list[idx].get(f, 0) or 0) for f in SHAPE_FEATS] for idx in m_indices])
        
        backup_preds = rf_backup.predict(np.nan_to_num(shape_X, nan=0.0))
        pred_series.iloc[m_indices] = backup_preds

    # プロパティへの書き戻し
    final_preds = pred_series.fillna(0).astype(int).values
    for i, feat in enumerate(batch):
        feat['properties']['PredClass'] = int(final_preds[i])
        
    return batch

def run_main_pipeline():
    """メイン推論管線の実行: ストリーミングバッチ処理"""
    start_time = time.time()
    print("[INFO] Starting V-GRID Stage 02 - Dual-Model Streaming Pipeline")
    
    # 1. Model Preparation
    rf_backup, preds_map = initialize_models()
    
    # 2. Streaming I/O
    print("[INFO] Opening data streams for read/write operations.")
    with fiona.open(SRC_SEGMENTS, 'r') as src:
        meta = src.meta.copy()
        meta['driver'] = 'GPKG'
        if 'PredClass' not in meta['schema']['properties']:
            meta['schema']['properties']['PredClass'] = 'int'
            
        total = len(src)
        print(f"[INFO] Total target objects: {total:,}")
        
        with fiona.open(OUT_VECTOR, 'w', **meta) as dst:
            buffer = []
            for feat in tqdm(src, total=total, desc="Streaming Inference"):
                buffer.append(feat)
                
                if len(buffer) >= BATCH_SIZE:
                    dst.writerecords(execute_batch_inference(buffer, rf_backup, preds_map))
                    buffer = []
            
            # 残余データの処理
            if buffer:
                dst.writerecords(execute_batch_inference(buffer, rf_backup, preds_map))

    elapsed = (time.time() - start_time) / 60.0
    print(f"\n[SUCCESS] Workflow completed in {elapsed:.1f} minutes.")
    print(f"[INFO] Classification result saved: {OUT_VECTOR}")

if __name__ == "__main__":
    run_main_pipeline()
