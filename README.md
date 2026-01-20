# TB-Scale Spatial Data Processing Pipeline

本リポジトリは、東京エリアを対象とした大規模土地被覆分類（LULC）プロジェクトの「核心処理エンジン」を公開したものです。現在、研究開発（Active R&D）の進行中であり、本公開分はTB級データのボトルネックを解決するための最も硬核（Core）なモジュール群で構成されています。今後、推論精度のさらなる向上（目標：Recall=0.872 以上の安定化）と、分散処理機能の拡張を予定しています。

##  Engineering Highlights

### 1. Memory-Efficient Streaming I/O
大容量ラスタデータの予測フェーズにおいて、`r+` モードを活用した **True-Streaming I/O** を実装。ギガバイト単位のデータを一度にロードせず、タイル単位でシーケンシャルにディスクへ直接書き込むことで、RAMのオーバーフロー（Memory Spike）を完全に抑制しています。

### 2. Tiled-Buffered Architecture
画像分割（Segmentation）プロセスにおいて、256pxのバッファ（Padding）を設けた **Tiled-Buffer** 方式を採用。大規模画像のスライシング処理で発生しがちなエッジ効果やタイル間の断絶を排除し、空間的な連続性を担保しました。

### 3. High-Dimensional Parallel Stacking
TB級の生データから特徴量スタックを構築する際、**Multi-processing Parallelization** と **Grid I/O** を統合。全プロセスを $Float32$ で統一することで精度損失を防止し、高品質な入力（Recall 向上に寄与）を実現しています。

### 4. Dual-Model Robustness & Vectorization
メインモデルの入力値が欠損している場合に備え、自動的にバックアップモデルへ切り替わる **Dual-Model Fallback** 機構を搭載。また、推论処理には **Vectorization** を採用し、定常的な高速推論を実現しました。

##  Results & Performance
- **Optimization Outcome**: 既存の公的プロダクトを 15.9% 上回る精度 **$Recall=0.872$** を達成。
- **Scalability**: 標準的なワークステーション環境で、TB級のデータセットをクラッシュすることなく完結させるスケーラビリティを実証済み。
