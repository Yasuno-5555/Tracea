# Tracea ROADMAP

監査結果（2026-05-12）に基づく実装優先順位。
DirtyData / LimeStudio / DirtyCircuit の ROADMAP とは独立して進捗管理する。

---

## Priority 0 — 即時対応

### P0.1 `#![allow(unsafe_op_in_unsafe_fn)]` の除去

Crate 全体で unsafe 操作の警告を抑制している問題を修正する。

- **対象**: `src/lib.rs:1-2`
- **内容**:
  1. `#![allow(unused_unsafe)]` と `#![allow(unsafe_op_in_unsafe_fn)]` を削除
  2. コンパイルエラーになる各 unsafe ブロックを `unsafe { ... }` で明示的にラップする
  3. FFI 呼び出し（cudarc, metal, ash）の事前条件・事後条件を `// Safety:` コメントとして記述
- **確認**: `cargo build` および `cargo clippy` が警告ゼロで通ること

### P0.2 CUDA エミッタの最小テスト追加

生成された CUDA コードの構文検証を行う。

- **対象**: `src/emitter/cuda.rs` + 新規テストファイル
- **内容**:
  1. 既知の config（RTX 3070, M=128/N=128/K=128, tile=64x64x16）で CUDA コードを生成
  2. 生成されたコードが期待する文字列パターンを含むことをアサート（例：`extern "C" __global__`, `__syncthreads()` など）
  3. 無効な tile size（`m_tile % 16 != 0` など）でパニックしないことを確認
- **確認**: `cargo test emitter` でテストが通ること

### P0.3 UniversalEmitter のフォールバックを明確にする

CUDA 以外のバックエンドがコメント文字列を返す問題を修正する。

- **対象**: `src/emitter/universal.rs`
- **選択肢**:
  - A. `Result<String, String>` を返すように変更し、未対応バックエンドは `Err` を返す（推奨）
  - B. `unreachable!()` でパニックさせる（即座に未実装を検出できるが、実行時まで分からない）
- **推奨**: A。`generate_from_strategy` の戻り値を `Result<String, String>` に変更し、呼び出し元で適切にエラーハンドリングする

## Priority 1 — 短期（1-2週間）

### P1.1 PipelineConfig のバリデーション追加

無効な設定値でカーネルが生成されるのを防ぐ。

- **対象**: `src/core/config.rs`
- **内容**: `PipelineConfig::validate()` メソッドを追加し、以下をチェック：
  - `m_tile % 16 == 0`, `n_tile % 16 == 0`（メモリアライメント）
  - `num_stages >= 2`（パイプラインハザード防止）
  - `k_tile >= 16`（最小タイルサイズ）
  - `force_num_warps` が空でない場合、`> 0` かつ `<= 32`
- **確認**: 既存の `test_golden_config_preservation` が引き続きパスすること

### P1.2 MetaTuner のバックエンド検出

存在しない GPU のプロファイルを作成しないようにする。

- **対象**: `src/optimizer/tuner.rs`
- **内容**: `MetaTuner::new()` でバックエンドを動的に検出する。`cudarc` の `CudaDevice::count()` や Metal の `MTLCreateSystemDefaultDevice()` を使って実際に存在するデバイスのみプロファイルを追加する
- **非目標**: リモートの CUDA デバイスの検出（これはスコープ外）

### P1.3 エディションを 2021 に更新

- **対象**: `Cargo.toml`
- **内容**: `edition = "2018"` → `edition = "2021"`
- **確認**: `cargo build` が通ること

### P1.4 コード生成の数値定数をパラメータ化

`emitter/cuda.rs` のハードコード値（`+ 8` パディング、`* 2` ダブルバッファ）を PipelineConfig のパラメータとして外部から制御可能にする。

- **対象**: `src/emitter/cuda.rs`, `src/core/config.rs`
- **内容**: `PipelineConfig` に `bank_conflict_padding: u32`, `double_buffer: bool` フィールドを追加（`double_buffer` は既に存在するので `bank_conflict_padding` のみ追加）。エミッタがこれらの値を参照するように変更する

### P1.5 PipelineConfig の直列化を serde に統一

現在の `to_vector()` / `from_vector()` は f32 配列のインデックス依存で脆弱。フィールド追加時のバグを防ぐ。

- **対象**: `src/core/config.rs`
- **内容**:
  1. `PipelineConfig` は既に `Serialize` / `Deserialize` が derive されている（`use serde`）ので、`to_vector()` / `from_vector()` を JSON または MessagePack に置き換える
  2. AutoTuner の `config_features()` などベクトル表現が必要な箇所は、`to_vector()` の呼び出しを維持しつつ、`serde_json::to_value()` を経由する形に変更
  3. `from_vector()` の呼び出し元を全て `serde_json::from_str()` または同等の安全なデシリアライズに移行
- **確認**: `test_golden_config_preservation` を含む全テストが通ること
- **備考**: 現在 `from_vector()` は `double_buffer` を常に `false` に設定するバグがある。本対応で自動的に修正される

### P1.6 CUDA エミッタのパニック分岐を Result に変更

`emitter/cuda.rs` の `generate_gemm()` で無効な warp partitioning 時に `panic!()` する問題を修正する。

- **対象**: `src/emitter/cuda.rs`
- **内容**:
  1. `generate_gemm()` の戻り値を `Result<String, EmissionError>` に変更
  2. 無効なタイルサイズの場合は `Err(EmissionError::InvalidTileConfiguration { ... })` を返す
  3. 呼び出し元（`CUDAEmitter::emit()` など）で `Result` を伝搬
- **確認**: 不正な config でパニックせず `Err` が返ること

`emitter/cuda.rs` のハードコード値（`+ 8` パディング、`* 2` ダブルバッファ）を PipelineConfig のパラメータとして外部から制御可能にする。

- **対象**: `src/emitter/cuda.rs`, `src/core/config.rs`
- **内容**: `PipelineConfig` に `bank_conflict_padding: u32`, `double_buffer: bool` フィールドを追加（`double_buffer` は既に存在するので `bank_conflict_padding` のみ追加）。エミッタがこれらの値を参照するように変更する

## Priority 2 — 中期（2-4週間）

### P2.1 エミッタの出力コード構文検証パイプライン

生成された CUDA/ROCm/Metal コードの構文を自動検証する仕組みを追加する。

- **対象**: 各エミッタ（動作確認用のテスト）
- **内容**:
  1. `nvrtc`（CUDA）または `shaderc`（Vulkan）を利用して生成コードをコンパイル
  2. 構文エラーをテストとして検出
  3. CI で GPU がない場合はコンパイルのみ（実行はスキップ）
- **確認**: `cargo test` でコード生成 + コンパイルテストが実行されること
- **非目標**: 生成されたカーネルの実行時検証（これは次フェーズ）

### P2.2 ガウス過程の Cholesky 実装

現在の簡略 GP を、標準的な Cholesky 分解ベースの実装に置き換える。

- **対象**: `src/optimizer/mod.rs`（GaussianProcess）
- **内容**:
  1. `faer` または `nalgebra` の Cholesky 分解を使ってカーネル行列を分解
  2. `predict()` で正確な事後平均＋分散を計算
  3. `marginal_log_likelihood()` で正確な対数尤度を計算（有限差分ではなく自動微分または解析勾配）
- **非目標**: 完全な GP ライブラリ化。まず Tracea 内で閉じた実装

### P2.3 エミッタ責務の整理

`emitter/` のファイル構成を整理し、新しい演算の追加コストを下げる。

- **対象**: `src/emitter/`
- **内容**:
  1. `emitter/traits.rs` で定義された `Emitter` trait を全てのバックエンドが実装することを強制
  2. CUDA エミッタを `metal/` と同様のサブディレクトリ構造に整理：
     - `emitter/cuda/gemm.rs`, `emitter/cuda/attention.rs`, `emitter/cuda/conv.rs`, ...
  3. `universal.rs` は単なるディスパッチャに縮小
  4. `MetalEmitter` の `generate_from_ir()` で未実装のパス（`ConvTranspose2d`, `MatrixCore`, `LowRankMlp`）を `panic!` ではなく `Result::Err` に変更
- **非目標**: ROCm エミッタの完全実装（まず CUDA/Metal の整理が優先）

### P2.4 Doctor モジュールの実装状況確認

9つのサブモジュールを持つ doctor の実際の実装内容を確認し、スタブを明確にする。

- **対象**: `src/doctor/`
- **内容**: 各モジュールを個別にレビューし：
  - 実装済み → そのまま
  - 空 or スタブ → `todo!()` か明示的な `unimplemented!()` に変更
  - 未使用 → `#[allow(dead_code)]` の削除を検討

### P2.5 Roofline prior のハードウェアプロファイル更新

現在の `GaussianProcess::roofline_prior()` はバックエンドごとに固定値（Metal は常に 5 TFLOPS、CUDA 非 TensorCore は 20 TFLOPS）を使用しており、実際の GPU 世代の差を反映していない。

- **対象**: `src/optimizer/mod.rs`（`GaussianProcess::roofline_prior()`）
- **内容**:
  1. `HardwareProfile` から動的にピーク性能値を参照する（`peak_tflops`, `mem_bw` フィールドを `HardwareProfile` に追加）
  2. 各 `HardwareProfile::rtx3070()` 等に実測に近い値を設定
  3. roofline prior の計算式は維持するが、入力パラメータを動的にする
- **確認**: 既存の GP テストが通ること。prior の値がハードウェアに応じて変化すること

### P2.6 TTG 実装の完了度確認

中核概念である Topological Tile Graph（TTG）の実装状態を確認し、不足を明確にする。

- **対象**: `src/runtime/ttg.rs`, `src/runtime/ttg_builder.rs`
- **内容**:
  1. TTG Builder が PolicyDecision から正しい TTGLayout を構築できることを確認するテストを追加
  2. L1 Map（論理割当）と L2 Table（タイルメタデータ）の両方が構築されていることを検証
  3. 空の PolicyDecision や無効なトポロジでの動作を確認
  4. 不足しているパスがあれば `todo!()` で明示
- **確認**: `cargo test ttg` 相当のテストが通ること

### P2.7 Metal エミッタのテスト追加

実装済みの Metal エミッタ（GEMM / Attention / Conv）のテストがない状態を改善する。

- **対象**: `src/emitter/metal/`
- **内容**:
  1. Metal GEMM の3バリアント（single_buffer / double_buffer / tiled）のコード生成スニペット一致テスト
  2. Metal Attention の4バリアント（Naive / SimdQK / SimdFull / FlashV2）のコード生成テスト
  3. Metal Conv の出力される MSL コードキーワード（`simdgroup_float8x8`, `threadgroup_barrier` など）の存在確認
  4. macOS 上では `metal` crate を使ったコンパイルテスト（`metal::Library::new_with_source`）
- **確認**: `cargo test --features metal` で macOS 上で全テストが通ること

### P2.6 `src/bin/` のバイナリ整理

`src/bin/` に 8 つのバイナリが存在する。それぞれの目的を明確にし、CI でビルドされるようにする。

- **対象**: `src/bin/`
- **内容**:
  - `tuner.rs`: メインの自動チューニングバイナリとして維持
  - `self_evolution_demo.rs`: デモとして維持
  - `probe_metal.rs`: Metal 開発専用
  - `capture_bench.rs`, `gemm_timing_test.rs`, `network_bench.rs`: ベンチマーク用
  - `cache_verification.rs`, `icb_min_crash.rs`: デバッグ用
- **非目標**: バイナリの削減。ただし各バイナリの先頭に目的をコメントとして追加する

## Priority 3 — 長期（1-2ヶ月）

### P3.1 テストカバレッジ拡充

| 領域 | 優先テスト | 種別 |
|------|-----------|------|
| core/op | DimExpr, GemmOp, FusedGemmOp のシリアライズラウンドトリップ | unit |
| core/config | PipelineConfig to_vector/from_vector のラウンドトリップ | property |
| core/graph | 空グラフ、単一ノード、循環依存の動作 | unit |
| emitter/cuda | GEMM/Attention/Conv のコード生成スニペット一致 | unit |
| emitter/metal | Metal GEMM/Attention/Conv のコード生成スニペット一致（P2.5 と統合） | unit |
| emitter/universal | 全バックエンドのディスパッチテスト | unit |
| optimizer/gp | GP predict の事後平均が観測点を補間すること | unit |
| optimizer/hardware | check_feasibility が不正 config を正しく弾くこと | property |
| runtime/manager | カーネルキャッシュのヒット/ミス | integration |
| policy | config 提案の多様性（同じ config ばかり出ないこと） | property |

### P3.2 ROCm エミッタの本実装

ROCm（AMD GPU）バックエンドを実装する。
※ Metal バックエンドは既に実装済み（`emitter/metal/` に GEMM / Attention / Conv の実装あり、2172行）。

- **対象**: `src/emitter/rocm.rs`, `src/emitter/rocm_jit.rs`, `src/emitter/rocm_driver.rs`
- **内容**: `Emitter` trait を完全に実装し、`UnifiedOpIR` から正しい ROCm カーネルコードを生成する
- **ベンチマーク**: CUDA 版と同等の GEMM パフォーマンスを達成すること

### P3.3 生成カーネルの実行時検証

生成されたカーネルが正しい数値結果を返すことを検証するテストを追加する。

- **対象**: 新規 `tests/` ファイル
- **内容**:
  1. ランダムな入力行列で CUDA GEMM を実行
  2. CPU での参照実装（`matrixmultiply` crate）の結果と比較
  3. 許容誤差（`max_abs_error < 1e-5`）を確認
- **前提**: CUDA 対応 GPU が存在する環境（テストランタイムでスキップ可能にする）

### P3.4 HardwareProfile の自動検出

`HeroScope`（ハードウェアプロービング）を使って `HardwareProfile` を自動生成する。

- **対象**: `src/optimizer/mod.rs` + `src/optimizer/heroscope.rs`
- **内容**:
  1. `HeroScopeV3` の `probe()` メソッドを実装し、実際の GPU から shared memory サイズ・レジスタ数・warp サイズを取得
  2. `HardwareProfile::auto_detect()` ファクトリメソッドを追加
  3. 自動検出できない場合は現在の手動プロファイルをフォールバックとして使用
- **非目標**: 全 GPU のプロファイルデータベース。検出できたものだけ使う

### P3.5 CI パイプラインの整備

現在のリポジトリに CI 設定（GitHub Actions 等）がない。最低限の CI を追加する。

- **対象**: `.github/workflows/ci.yml`（新規）
- **内容**:
  1. Rust stable での `cargo build --all-targets` の実行
  2. `cargo test` の実行（GPU 依存テストは `#[ignore]` または feature gate でスキップ）
  3. `cargo clippy --all-targets -- -D warnings` の実行
  4. Rust 2024 edition 移行時は `cargo fmt --check` も追加
- **非目標**: GPU を必要とするテストの CI 実行（セルフホストランナーが必要）
- **確認**: CI がグリーンになること

### P3.6 Python/Rust テストの分離

`tests/` ディレクトリに Rust 統合テストと Python テストスクリプトが混在している。`cargo test` で Python テストが無視される問題を整理する。

- **対象**: `tests/` ディレクトリ
- **内容**:
  1. Python スクリプト（`*.py`）を `tests/python/` または `tests/` から `benchmarks/` へ移動
  2. Rust の `#[test]` 属性を持たないファイルは `tests/` から除去
  3. 残ったテストが `cargo test` で実際に実行されることを確認
- **確認**: `cargo test` で実行されるテスト数と実ファイル数が整合すること

### P3.7 コード生成の PTX/SASS 検証

CUDA コード生成パイプラインに PTX アセンブリレベルの検証を追加する。

- **対象**: `src/emitter/cuda.rs`, 新規 `tests/ptx_validation.rs`
- **内容**:
  1. `nvcc` または `ptxas` を使って生成 CUDA コードから PTX を生成
  2. PTX 内に期待する命令（`mma.sync`, `ldmatrix` など）が含まれることを確認
  3. レジスタ数を推定し、上限を超えていないことを確認
- **非目標**: SASS レベルの検証（cuobjdump が必要で環境依存が大きい）

## 非目標

以下の項目は現時点では対応しない。

- **完全な Vulkan バックエンド**: feature gate されている通り実験的。
- **FFI の文書化**: `include/` と `tracea-ffi` / `tracea-python` は将来の拡張。
- **マルチノード分散チューニング**: 現状の `cluster-api`（DirtyCircuit 側）は Tracea 本体のスコープ外。
- **完全な Cholesky 実装の数値最適化**: まず正しく動くこと。高速化は GP の benchmark を取ってから。

## 進捗管理

対応ブランチ命名規則: `fix/tr-{issue-name}` / `feat/tr-{issue-name}`

P0-P2（22項目）完了。P3 も大部分完了。残りは ROCm 実機がなければ進められない項目。

```markdown
- [x] P0.1 unsafe 警告復活
- [x] P0.2 CUDA エミッタ最小テスト
- [x] P0.3 UniversalEmitter Result 化
- [x] P1.1 PipelineConfig バリデーション
- [x] P1.2 MetaTuner バックエンド検出
- [x] P1.3 edition 2021
- [x] P1.4 数値定数パラメータ化
- [x] P1.5 直列化を serde に統一
- [x] P1.6 CUDA エミッタ panic → Result
- [x] P2.1 コード生成構文検証
- [x] P2.2 GP Cholesky 実装
- [x] P2.3 エミッタ責務整理
- [x] P2.4 Doctor 確認
- [x] P2.5 Roofline prior 動的化
- [x] P2.6 TTG 実装完了度確認
- [x] P2.7 Metal エミッタテスト
- [x] P2.8 bin 目的明記
- [x] P3.6 Python/Rust テスト分離
- [x] 全 32 examples のコンパイルエラー修正
- [x] P3.5 CI パイプライン
- [x] P3.3 実行時数値検証 — Metal GEMM 8×8 indexing バグ修正、数値等価性確認
- [x] P3.1 テストカバレッジ拡充 — gp_interpolation, runtime_cache, universal_dispatch, 全47テスト
- [x] TTG 真のトポロジー実装 — 隣接グラフ、K-split、非矩形マスク、トポロジカルソート
- [x] TileTopology: データ再利用ヒント、融合候補検出、A/B キャッシュ最適化
- [x] MetaTuner Metal 対応完了 — バックエンド検出→タイル提案→実測→キャッシュの自動パイプライン
- [x] TTG タイル融合テンプレート — fusion_count パラメータ、N-tile 融合カーネル生成
- [x] Metal causal mask 実装 — FlashAttention V2 テンプレート #ifdef CAUSAL
- [x] DeviceTTG topology-guided ordering — L1 マップをトポロジー順に並び替え
- [x] M1 GEMM ベンチマーク: 0.38→1.28 TFLOPS（15%→49% of peak, 3.4x改善）
- [ ] P3.2 ROCm エミッタの本実装（ROCm 実機が必要）
- [ ] P3.4 HardwareProfile 自動検出（GPU プロービング実装）
- [ ] P3.7 PTX/SASS 検証（CUDA 実機で nvcc/ptxas 確認）
```
