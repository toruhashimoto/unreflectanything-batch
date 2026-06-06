# UnReflect Batch（日本語）

[![CI](https://github.com/toruhashimoto/unreflectanything-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/toruhashimoto/unreflectanything-batch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream: UnReflectAnything](https://img.shields.io/badge/upstream-UnReflectAnything-blue)](https://github.com/alberto-rota/UnReflectAnything)
[![3DGS: LichtFeld Studio](https://img.shields.io/badge/3DGS-LichtFeld%20Studio-orange)](https://github.com/MrNeRF/LichtFeld-Studio)

[English](README.md) · **日本語**

> **独立ラッパーです。** 本プロジェクトは
> [UnReflectAnything](https://github.com/alberto-rota/UnReflectAnything) の
> **独立したバッチ用ラッパー**であり、本家の作者とは**無関係（非公認）**です。本家のコードは
> **一切同梱していません**（モデルは PyPI から導入し、公開 API/CLI 経由で呼び出します）。

入力写真から **鏡面反射・白飛びハイライト**を一括除去し、**3D Gaussian Splatting（3DGS）/
フォトグラメトリ**（RealityScan・Postshot・Nerfstudio・COLMAP など）の**前処理**として使うための
Windows 向けローカルアプリです。

[**UnReflectAnything**](https://alberto-rota.github.io/UnReflectAnything/) を呼び出すだけの
**薄い・安全なラッパー**で、研究コードは改変しません。**元画像は決して変更せず**、クリーン化画像・
before/after プレビュー・差分ヒートマップ・各画像のログを**別フォルダ**に出力します。**ファイル名と
サブフォルダ構成は維持**するので、既存の SfM/3DGS パイプラインへそのまま渡せます。

> ⚠️ **評価用途限定。** 単一画像の反射除去には**マルチビュー整合の保証がありません**。視点ごとに
> 異なる内容を補完してしまうと SfM の特徴照合を**悪化**させ得ます。出力は「**可視化・品質改善の評価**」
> 目的として扱い、測定の正データとはみなさないでください。必ず再構成を**除去あり/なしで A/B 比較**
> してから採用を判断してください（→ [推奨ワークフロー](#推奨ワークフロー3dgs--フォトグラメトリ)）。

---

## デモ

![before / after / diff](examples/demo_before_after.jpg)

*合成シーン（鏡面グレアあり）→ `--mask-composite` → 差分ヒートマップ。元画像は変更されず、白飛び領域
だけが変化し、ヒートマップが変化箇所を正確に示します。詳細・再現方法は [`examples/`](examples/) を参照。*

---

## 1. 動作要件

| | |
|---|---|
| OS | Windows 10/11（Windows 11 で検証） |
| Python | **3.11+**（3.11 で検証） |
| GPU（任意） | NVIDIA CUDA GPU。**RTX 50 系（Blackwell/sm_120）で検証**（RTX 5070 Ti）。CPU でも動作可（低速）。 |
| ディスク | 約 3 GB（PyTorch）＋ **約 5.9 GB（モデル重み）**＋ 画像 |

GPU が無くても自動で CPU にフォールバックします（低速）。

---

## 2. セットアップ

### 最も簡単：起動スクリプトに任せる
**`run_app.bat`**（GUI）または **`run_batch_example.bat`**（CLI）をダブルクリック。初回は仮想環境作成・
依存導入・重みダウンロードまで自動で行います。

### セットアップスクリプトを直接実行
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gui
```
`-Gui` で Streamlit も導入。`-SkipWeights` で 5.9 GB のダウンロードを後回し、
`-CudaIndex https://download.pytorch.org/whl/cu130` で新しいドライバ向けに変更可。

### 手動（2 つの落とし穴を理解する）
```powershell
# 1) 仮想環境を作成
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# 2) 落とし穴#1 — CUDA 版 PyTorch を「先に」PyTorch インデックスから入れる。
#    Windows の既定 PyPI torch は CPU 専用で、Blackwell(sm_120) には cu128 ビルドが必要。
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# 3) 落とし穴#2 — requirements.txt は transformers を「特定コミット」に固定。
#    素の `pip install unreflectanything` は最新 transformers を入れてしまい、DINOv3 のキーが
#    合わず推論が「Error(s) in loading state_dict ... Missing key(s) ... dinov3」で失敗する。
pip install -r requirements.txt
```

### GPU 確認（任意）
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), 'sm_120' in torch.cuda.get_arch_list())"
# 期待:  2.9.1+cu128 True True
```

---

## 3. モデル重みのダウンロード（初回のみ・必須）

**自動ダウンロードはありません**。一度だけ取得してください（約 5.9 GB）：
```powershell
.\.venv\Scripts\unreflectanything.exe download --weights
.\.venv\Scripts\unreflectanything.exe verify --weights   # 任意：読み込み確認
```
重みは `%LOCALAPPDATA%\unreflectanything\weights` にキャッシュされ、毎回再利用されます。

ツールから取得させることもできます：`main.py` に **`--download-weights`** を付ける、または GUI サイドバーの
**「Download model weights」**ボタン。重み未取得時は、**実行途中で落ちず**に、コマンドとキャッシュ場所を
示す分かりやすいメッセージで即座に停止します。

---

## 4. 使い方 — GUI
```powershell
run_app.bat
```
入力・出力フォルダを選び、サイドバーで設定して **Run batch**。進捗・サマリ・before/after サンプルが
表示されます。GUI は CLI と同じエンジンを呼びます。

## 5. 使い方 — CLI

### バッチ（メイン機能）
```powershell
python main.py --input "D:\photo_input" --output "D:\photo_unreflect" --recursive --make-preview --device cuda
```

主なオプション：

| フラグ | 既定 | 説明 |
|---|---|---|
| `--input, -i` | — | 入力フォルダ（**必須**） |
| `--output, -o` | — | 出力フォルダ（**入力の外**であること。**必須**） |
| `--recursive, -r` | off | サブフォルダを再帰処理（構成を出力に反映） |
| `--device, -d` | `auto` | `auto` / `cuda` / `cpu`（auto は動作可能な GPU があれば使用） |
| `--extensions` | `.jpg,.jpeg,.png,.tif,.tiff` | 対象拡張子（カンマ区切り） |
| `--make-preview` | off | before/after 横並びを `preview_compare/` に保存 |
| `--heatmap` | off | 輝度差ヒートマップを `heatmap/` に保存 |
| `--emit-mask` | off | 変更領域マスクを `masks/` に保存（COLMAP の除外マスク用） |
| `--composite` | off | モデル内部 composite（約 448px で合成、全体はソフト化されたまま） |
| `--mask-composite` | off | **ラッパーのフル解像度 composite**：白飛び以外は原寸のまま保持（**高解像度の SfM/3DGS 入力に最適**） |
| `--mask-level` | `248` | mask-composite：この輝度(0-255)より明るい画素のみ置換（高いほどタイト＝ボケにくい） |
| `--mask-dilation` | `0` | mask-composite：置換領域を N px 膨張（小さく保つ。大きいと主役がボケる） |
| `--exiftool` | off | exiftool があれば全メタデータを複写（メーカーノート/GPS/XMP・全形式・低速） |
| `--verbose` | off | エンジン自身の出力を表示 |
| `--overwrite` | off | 既存出力を上書き（既定はスキップ） |
| `--jpeg-quality` | `95` | JPEG 品質（95 以上を強制、4:4:4） |
| `--threshold` | `0.3` | ハイライト検出閾値（モデル） |
| `--dilation` | `40` | ハイライトマスクの膨張(px)（モデル） |
| `--limit N` | — | **テストモード**：先頭 N 枚のみ処理 |
| `--max-size PX` | — | **簡易モード**：長辺を縮小して処理（⚠ 出力寸法が変わる＝COLMAP 入力には不可） |
| `--download-weights` | off | 重みが無ければ約 5.9 GB を先に取得してから実行 |
| `--dry-run` | off | 処理内容を表示のみ（実行しない） |
| `--no-progress` | off | 進捗バーを無効化 |

### 単体処理（エンジンを直接）
```powershell
.\.venv\Scripts\unreflectanything.exe inference "in.jpg" -o "out.jpg" -d cuda
```

---

## 6. 出力構成
```
<output>/
├── <元のツリー・元のファイル名>            # クリーン化画像（形式/寸法/EXIF 維持）
├── preview_compare/                      # [Original | UnReflect | (Diff)] 横並び  (--make-preview)
├── heatmap/                              # 輝度差ヒートマップ                       (--heatmap)
├── masks/                                # 変更領域マスク                          (--emit-mask)
└── logs/
    ├── process_log.jsonl                 # 画像ごとの詳細 JSON（1 行 1 件）
    ├── process_log.csv                   # フラットな集計表
    ├── errors.csv                        # 失敗画像のみ
    └── run_summary.json                  # 実行全体の集計＋設定
```
各レコードに `processed_by: "UnReflectAnything"`、元画像参照・日時・モデル名/版・デバイス・入出力サイズ・
処理時間・使用パラメータ・評価指標・エラー内容を記録します。

---

## 7. 評価機能
- **平均輝度差**（前→後）
- **ハイライト画素率**（前/後）— 白飛び除去の度合いの指標
- **差分ヒートマップ**（`--heatmap`）— どこをどれだけ変えたか
- **変更マスク**（`--emit-mask`）— COLMAP の特徴**除外マスク**として渡せる（補完画素を幾何に使うより安全）
- **テストモード**（`--limit N`）・**簡易モード**（`--max-size PX`）

---

## 8. COLMAP / 3DGS 互換性の保ち方
- **元画像は不可侵**。出力は別ツリー、同名は既定でスキップ。
- **寸法を維持**（COLMAP は EXIF と画像サイズから焦点距離(px)を導出するため、リサイズは intrinsics を壊す）。
- **EXIF を維持**（特に `FocalLengthIn35mmFilm`/`FocalLength`。JPEG→JPEG は完全移植、TIFF/PNG はベストエフォート）＋ICC。
- **形式を維持**（JPEG→JPEG は品質≥95・4:4:4・単一エンコード、PNG/TIFF はロスレス）。二重 JPEG なし。
- **全集合を同一設定で均一処理**（視点間の輝度不連続を回避）。

---

## 9. 推奨ワークフロー（3DGS / フォトグラメトリ）
1. **撮影時の対策が最優先**：クロス偏光（ライト＋レンズに偏光）、つや消しスプレー、ソフト拡散光、露出固定。
   AI 除去は**撮り直せない素材の救済**に使う。
2. `--make-preview --heatmap` で実行し、**プレビューを目視**：ハイライトを綺麗に除去できているか、
   テクスチャを捏造していないか。視点ごとに異なる捏造は SfM を悪化させる。
3. **`--composite`**（ハイライト領域のみ変更）や **`--emit-mask`**（COLMAP 除外マスク）を検討。
4. **A/B 比較**：元画像とクリーン化の両方で再構成し、登録率・アーティファクトの良い方を採用。3DGS の
   一部手法は反射を**除去せずモデル化**する点にも留意。

---

## 9b. A/B 評価パイプライン（任意）

`tools/` 配下の 2 つのハーネスで、クリーニングが実際に再構成を良くするかを**測定**できます。外部ツールは
フラグ・環境変数・PATH のいずれからも解決（ハードコードなし）。

**外部ツール（非同梱）：**
- [COLMAP](https://github.com/colmap/colmap/releases) — `--colmap` または `$COLMAP_EXE`
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)（3DGS 用）— `--lichtfeld` または `$LICHTFELD_EXE`

### SfM の A/B — `tools/ab_colmap.py`
各画像セットで COLMAP スパース再構成を行い、登録枚数・3D 点数・トラック長・再投影誤差を比較。
```powershell
python tools\ab_colmap.py --work ab_work --matcher sequential --max-image-size 2000 ^
    --set original "D:\photo_input" --set cleaned "D:\photo_unreflect"
```

**マスク除外 — アライメントでは「除去」より良いことが多い。** SfMでは反射領域を*除去*する代わりに、**特徴抽出から“除外”**できます。画像を改変しないので、周囲の特徴点は記述子・幾何精度をそのまま保ち、視点依存で不安定な反射画素だけを無視できます。`--set-masked` で生成・比較：
```powershell
python tools\make_colmap_masks.py -i "D:\photo_input" -o "D:\refl_masks" --level 240 --dilation 2
python tools\ab_colmap.py --work ab_work --matcher sequential ^
    --set original "D:\photo_input" ^
    --set-masked masked "D:\photo_input" "D:\refl_masks"
```
マスクは**タイト**に — 実際の反射だけを除外し、空や白い面のような**明るい拡散面**まで消さないこと（除外し過ぎると特徴の宝庫を潰し、復元が分裂します）。実測では、タイトなマスク除外は除去（インペイント）より**3D点数も再投影精度も良好**でした。

### 3DGS の A/B — `tools/ab_3dgs.py`
セットごとに **COLMAP → LichtFeld Studio ヘッドレス学習 → eval レンダー → 同一視点の比較図＋レポート**
（PSNR/SSIM/ガウシアン数）。
```powershell
$env:LICHTFELD_EXE = "C:\path\to\LichtFeld-Studio.exe"
python tools\ab_3dgs.py --work ab3dgs_out ^
    --set original "D:\photo_input" --set cleaned "D:\photo_unreflect" ^
    --shared-poses original --steps-scaler 0.5 --resize-factor 2
```
- `--shared-poses NAME`：全セットを NAME のポーズで学習→フレーム単位で直接比較可能（同名・整列画像が前提）。
  省略すると各セット独立パイプライン。
- COLMAP の SIMPLE_RADIAL カメラには歪みがあるため、LichtFeld へ `--undistort` を自動付与（`--no-undistort` で無効）。
- 出力：`<work>/compare/*.jpg`（`GT | 各セットの 3DGS レンダー`）と `<work>/report.md`。

> **数値の正しい読み方。** PSNR/SSIM は**各セット自身の GT** に対する値です。GT が異なる（グレア有 vs クリーン）
> 場合、それは「絶対画質」ではなく**再構成のしやすさ**を表します。必ず比較図と併せて読んでください。実測では、
> 視点依存のグレアを除去すると PSNR/SSIM が上がりガウシアン数が**減る**傾向で、`--mask-composite` が SfM を
> 害さず有効なことが多いです（既定除去は高解像度をソフト化し SfM を悪化させ得る）。

---

## 10. トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| `Error(s) in loading state_dict … Missing key(s) … dinov3` | transformers が不一致。固定コミットを導入：`pip install "transformers @ https://github.com/huggingface/transformers/archive/2fe43376cdde02b7ffcf117e6eb9aa4375fb2dd1.zip"` |
| `torch.cuda.is_available()` が `False` / `no kernel image is available` | CPU 専用/不一致 torch。再導入：`pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall` |
| `Pretrained weights … not downloaded` | 重み未取得 → [3 章](#3-モデル重みのダウンロード初回のみ必須)。`--download-weights` でも可。 |
| `UnicodeEncodeError: 'cp932' …`（`unreflectanything --help` 実行時） | 非 UTF-8 コンソール。`PYTHONUTF8=1` を設定（本アプリと .bat は設定済み）。 |
| 大画像で CUDA メモリ不足 | `--max-size` で簡易確認、または `--device cpu`、または枚数を分割。 |
| 1 枚失敗 | `logs/errors.csv` に記録して処理は継続（設計どおり）。 |

---

## 11. プロジェクト構成
```
unreflectanything-batch/
├── main.py                 # CLI エントリ
├── app.py                  # Streamlit GUI（第2段階）
├── requirements.txt / pyproject.toml
├── run_app.bat / run_batch_example.bat
├── scripts/setup_env.ps1   # 環境インストーラ（venv + cu128 torch + 依存 + 重み）
├── src/
│   ├── image_io.py         # 探索 + EXIF/形式保持 I/O
│   ├── metrics.py          # 輝度/ハイライト指標・ヒートマップ・変更マスク・full-res composite
│   ├── preview.py          # before/after 比較生成
│   ├── logger.py           # JSONL / CSV / errors / summary
│   └── unreflect_batch.py  # エンジン（device 選択・モデル読込・1枚処理）
├── tools/
│   ├── ab_colmap.py        # COLMAP スパース再構成 A/B
│   └── ab_3dgs.py          # 3DGS A/B（COLMAP→LichtFeld→比較図）
├── examples/               # 合成デモ（make_demo.py で再現）
└── tests/                  # 高速ユニットテスト（torch 不要）
```

## 12. テスト
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
ユーティリティ（探索・I/O・指標・プレビュー・ログ）を torch 非依存で検証。CI は Ubuntu+Windows ×
Python 3.11/3.12 で実行します。

## 13. 注意・ライセンス・謝辞
- 本ラッパーは現状有姿で提供。**UnReflectAnything** は MIT ですが、内部の凍結バックボーン **DINOv3** は
  **Meta の DINOv3 ライセンス**（非オープンソース・「Built with DINOv3」表記と利用制限あり）に従います。
  再配布・商用前に必ず確認してください。
- **高解像度入力はソフト化されます。** モデルは内部で約 448px に縮小→元寸へ拡大するため、例えば 4K では
  画像**全体**の高周波が失われ、SfM の特徴照合を**悪化**させ得ます。高解像度のフォトグラメトリでは
  **`--mask-composite`** を使ってください（白飛び以外は原寸を保持）。実測：4K で既定除去は鮮鋭度
  （ラプラシアン分散）を約 94% 低下、`--mask-composite` は大部分を維持。
- 内部リサイズのため、変更領域の微細テクスチャは「再構成」であり「測定」ではありません。だから出力は
  **評価用**であり計測用ではありません。

### 謝辞 / 第三者

本プロジェクトは以下を**同梱・再配布しません**（導入済みの依存・外部ツールとして呼び出すのみ）。各ライセンスを
順守してください：

- [UnReflectAnything](https://github.com/alberto-rota/UnReflectAnything)（MIT）— 反射除去モデル。凍結
  バックボーン **DINOv3** は **Meta の DINOv3 ライセンス**。
- [COLMAP](https://github.com/colmap/colmap)（BSD）— Structure-from-Motion（A/B ハーネスで使用）。
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)（GPL-3.0）— 3DGS トレーナ。
  `tools/ab_3dgs.py` が**外部プロセスとして呼び出すのみ**（本プロジェクトにリンク/同梱はしません）。
- [PyTorch](https://pytorch.org)・[Pillow](https://python-pillow.org)・
  [piexif](https://github.com/hMatoba/Piexif)・[Streamlit](https://streamlit.io)。

本プロジェクトは **MIT ライセンス**（[`LICENSE`](LICENSE)）。第三者ツール/モデルの条項は
[`NOTICE.md`](NOTICE.md)（DINOv3・LichtFeld の GPL-3.0 等）を参照。
