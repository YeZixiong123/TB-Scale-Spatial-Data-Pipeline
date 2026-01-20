#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- PlanetScope(7-Band)影像に対するFelzenszwalb分割アルゴリズムの適用
- 幾何学的特徴（PCA）および分光インデックス（NDVI/NDWI/NDRE）の統合
- タイル間の不連続性を排除するためのバッファ(Padding)処理の実装
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from skimage.segmentation import felzenszwalb
from skimage.filters import gaussian 
from sklearn.decomposition import PCA
import fiona
from tqdm import tqdm
import gc

#  1. Path Definition 
INPUT_MOSAIC = '../../data/02_preprocessed/tokyo_planet_mosaic_FINAL_MEDIAN.tif'
OUTPUT_DIR = "../../3_output/2_object_based_EXP/01_segments_v1.1/"

#  2. Segmentation Hyperparameters 
# セグメンテーションの細分化度合いの制御 (Scale/Min_size)
FELZ_SCALE = 35        
FELZ_MIN_SIZE = 15     
GAUSSIAN_SIGMA = 0.5  # エッジの毛羽立ち抑制のための平滑化強度

# 分割統治(Tiling)設定: メモリ消費量とI/O効率の最適化
TILE_SIZE = 4096       
BUFFER_PADDING = 256  # 隣接タイルとの整合性確保のためのオーバーラップ幅

# 読み込み対象波段 (1-based index)
BANDS_TO_PROCESS = [1, 2, 3, 4, 5, 6, 7]


# Image Processing Kernels


def normalize_to_8bit(arr):
    """0-1の範囲を255スケールに正規化 (演算精度の維持を前提)"""
    arr = np.nan_to_num(arr, nan=0.0)
    # 2-98%パーセンタイルによるコントラスト調整
    p_low, p_high = np.percentile(arr, 2), np.percentile(arr, 98)
    if p_high - p_low == 0: 
        return np.zeros_like(arr)
    res = np.clip((arr - p_low) / (p_high - p_low), 0, 1)
    return (res * 255.0).astype(np.float32)

def extract_feature_stack(window_data):
    """
    分光・テクスチャ・インデックスの統合特徴スタックの生成
    Format: (H, W, Channels) for skimage processing
    """
    _, h, w = window_data.shape
    
    # バンドマッピング (0-based)
    green  = window_data[2].astype(np.float32)
    yellow = window_data[3].astype(np.float32)
    red    = window_data[4].astype(np.float32)
    red_e  = window_data[5].astype(np.float32) 
    nir    = window_data[6].astype(np.float32)
    
    # スペクトルインデックス計算 (微小値1e-5によるゼロ除算回避)
    ndvi = (nir - red) / (nir + red + 1e-5)
    ndwi = (green - nir) / (green + nir + 1e-5)
    ndre = (nir - red_e) / (nir + red_e + 1e-5) 
    
    # 次元圧縮(PCA)によるテクスチャ情報の抽出
    try:
        raw_reshaped = window_data.reshape(window_data.shape[0], -1).T
        pca_result = PCA(n_components=1).fit_transform(np.nan_to_num(raw_reshaped, nan=0.0))
        pca_band = pca_result.reshape(h, w)
    except:
        pca_band = np.zeros((h, w), dtype=np.float32)

    # 特徴量のスタッキングとガウスフィルタによるノイズ抑制
    stack = np.stack([
        normalize_to_8bit(green),
        normalize_to_8bit(red),
        normalize_to_8bit(yellow),
        normalize_to_8bit(red_e),
        normalize_to_8bit(ndvi),
        normalize_to_8bit(ndwi),
        normalize_to_8bit(ndre),
        normalize_to_8bit(pca_band)
    ], axis=-1)
    
    # 互換性を考慮したキーワード引数の処理
    try:
        return gaussian(stack, sigma=GAUSSIAN_SIGMA, channel_axis=-1, preserve_range=True)
    except TypeError:
        return gaussian(stack, sigma=GAUSSIAN_SIGMA, multichannel=True, preserve_range=True)


# Main Task Execution
def run_tiled_segmentation():
    """バッファ付き分塊分割およびベクトル化処理の実行"""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[PROCESS] Initializing Segmentation Task - Mode: Tiled-Buffer (Sigma: {GAUSSIAN_SIGMA})")
    
    try:
        with rasterio.open(INPUT_MOSAIC) as src:
            src_crs = src.crs
            src_h, src_w = src.height, src.width
            
            # グリッド分割計画の立案
            grid_windows = []
            for y in range(0, src_h, TILE_SIZE):
                for x in range(0, src_w, TILE_SIZE):
                    grid_windows.append(Window(x, y, min(TILE_SIZE, src_w - x), min(TILE_SIZE, src_h - y)))
            
            print(f"[INFO] Source Dimensions: {src_w}x{src_h}, Total Tiles: {len(grid_windows)}")
            global_id_offset = 0 

            for base_win in tqdm(grid_windows, desc="Processing Tiles"):
                # バッファ(Padding)領域の計算
                c_off, r_off = base_win.col_off, base_win.row_off
                w, h = base_win.width, base_win.height
                
                # 有効範囲境界の取得
                p_l = min(c_off, BUFFER_PADDING)
                p_t = min(r_off, BUFFER_PADDING)
                p_r = min(src_w - (c_off + w), BUFFER_PADDING)
                p_b = min(src_h - (r_off + h), BUFFER_PADDING)
                
                # 拡張読み込みウィンドウの定義
                read_win = Window(c_off - p_l, r_off - p_t, w + p_l + p_r, h + p_t + p_b)
                
                try:
                    raw_block = src.read(BANDS_TO_PROCESS, window=read_win)
                    if np.all(raw_block == 0): continue # 空白域のスキップ
                        
                    # 特徴抽出およびセグメンテーションの実行
                    smooth_feats = extract_feature_stack(raw_block)
                    buffered_labels = felzenszwalb(
                        smooth_feats, 
                        scale=FELZ_SCALE, sigma=0, min_size=FELZ_MIN_SIZE,
                        channel_axis=-1
                    )
                    
                    # バッファ除去による中心領域の抽出
                    core_labels = buffered_labels[int(p_t):int(p_t + h), int(p_l):int(p_l + w)]
                    core_labels_global = (core_labels + global_id_offset).astype(np.int32)
                    
                    # タイル別ESRI Shapefileへのエクスポート
                    tile_shp_path = os.path.join(OUTPUT_DIR, f"tile_{int(c_off)}_{int(r_off)}.shp")
                    transform = src.window_transform(base_win)
                    
                    shp_schema = {'geometry': 'Polygon', 'properties': {'seg_id': 'int'}}
                    with fiona.open(tile_shp_path, 'w', driver='ESRI Shapefile', 
                                    crs=src_crs.to_dict(), schema=shp_schema) as shp:
                        
                        gen_shapes = shapes(core_labels_global, mask=(core_labels != -1), transform=transform)
                        for geom, val in gen_shapes:
                            shp.write({'geometry': geom, 'properties': {'seg_id': int(val)}})
                    
                    global_id_offset = core_labels_global.max() + 1

                except Exception as tile_err:
                    # ログ形式でのエラーレポート
                    print(f"[WARN] Failed to process tile at {c_off}, {r_off}: {tile_err}")
                    continue
                finally:
                    gc.collect() # 累積メモリの解放

    except Exception as fatal_err:
        print(f"[FATAL] Critical failure in pipeline: {fatal_err}")
        return

    print(f"\n[COMPLETE] Segmentation workflow finished. Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # マルチプロセス環境下でのStart Method固定
    run_tiled_segmentation()
