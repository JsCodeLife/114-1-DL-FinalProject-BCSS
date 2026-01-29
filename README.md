# 114-1-DL-FinalProject-BCSS
CE6146 Introduction to Deep Learning

## Analyzing Preprocessing Techniques and Various Models on the BCSS Dataset

端到端的組織病理影像分割框架，基於 MMSegmentation，支援多模型、多解析度訓練與評估。

## 📌 概述

此專案針對 **BCSS（Breast Cancer Segmentation）** 資料集進行語意分割，使用 MMSegmentation 框架整合三種深度學習模型：
- **UNet** - 經典的編碼-解碼架構
- **DeepLabV3+** - 基於 Atrous Convolution 的多尺度模型
- **SegFormer** - Vision Transformer 型分割器

支援 **224×224** 與 **512×512** 兩種解析度的訓練與測試。

## 📂 專案模組

*   **[preprocessing/](./preprocessing/README.md)**: 影像前處理模組 (Macenko Stain Normalization + CLAHE)

## 🚀 快速開始

### 環境設置

```bash
# 1. 克隆專案
git clone <repo_url>
cd DLFinal

# 2. 創建 conda 環境
conda create -n segmentation python=3.10
conda activate segmentation

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 安裝 MMSegmentation （已在 requirements.txt 中，或手動安裝最新版本）
pip install mmsegmentation
cd mmsegmentation && pip install -e . && cd ..
```

### 準備資料

資料須按照 MMSegmentation 標準格式組織：

```
data/
├── BCSS_MMSEG_FINAL/               # 224×224 原始資料
│   ├── img_dir/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── ann_dir/
│       ├── train/
│       ├── val/
│       └── test/
└── BCSS_512_MMSEG_FINAL/           # 512×512 原始資料
    ├── img_dir/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── ann_dir/
        ├── train/
        ├── val/
        └── test/
```

若使用預處理資料（推薦），資料在 `data_V2/preprocess_data/` 目錄。

### 訓練

#### 方式 1：互動式批次訓練（推薦）

```bash
python Bcss_auto_run_V2.1.py
```

互動式選單會引導你：
1. 選擇訓練模式：
   - `v2`：使用原始資料 (`data/BCSS_MMSEG_FINAL` 等)
   - `v2.1`：使用預處理資料 (`data_V2/preprocess_data`)
2. 選擇模型配置：
   - 單個：`1`
   - 多個：`1,3,5`
   - 範圍：`1-4`
   - 全部：`all`

輸出結果：
- 訓練日誌：`work_dirs/<model_name>/`
- 錯誤日誌：`training_errors/` 或 `training_errors_p/`

#### 方式 2：直接執行 mmseg 命令

```bash
cd mmsegmentation

# 單卡訓練
python tools/train.py \
    configs/configs_comparison/unet_224_config.py \
    --work-dir ../work_dirs/unet_224

# 8 卡分散式訓練
bash tools/dist_train.sh \
    configs/configs_comparison/unet_224_config.py \
    8 \
    --work-dir ../work_dirs/unet_224
```

### 評估與測試

```bash
python run_test_evaluation.py
```

互動式選單：
- 選擇模式（v2 / v2.1 / 兩者）
- 自動掃描訓練結果
- 執行測試評估並保存 JSON 結果

### 可視化

```bash
python visualize_results.py
```

生成：
- 訓練曲線圖表（`metrics_comparison.png`）
- 指標摘要（`metrics_summary.csv`）
- 模型推論對比圖（`prediction_results_<image>.png`）

## 📁 專案結構

```
DLFinal/
├── mmsegmentation/                 # MMSegmentation 框架
│   ├── configs/
│   │   └── configs_comparison/     # 訓練配置（224/512 三模型各 2 個）
│   ├── tools/
│   │   ├── train.py
│   │   ├── dist_train.sh
│   │   ├── test.py
│   └── mmseg/                      # MMSegmentation 源代碼
├── preprocessing/                   # ⭐ 影像前處理模組 (本機新增)
├── data/                            # 原始資料（224/512）
├── data_V2/preprocess_data/         # 預處理資料（推薦使用）
├── work_dirs/                       # v2 訓練結果
├── work_dirs_p/                     # v2.1 訓練結果
├── final_results/                   # v2 評估與可視化結果
├── final_result_p/                  # v2.1 評估與可視化結果
├── Bcss_auto_run_V2.1.py           # ⭐ 主訓練腳本（雙模式）
├── run_test_evaluation.py           # 測試評估腳本
├── visualize_results.py             # 可視化腳本
├── requirements.txt                 # Python 依賴
└── README.md                        # 本檔案
```

## 📊 資料集統計

| 解析度 | 資料集 | 訓練 | 驗證 | 測試 | 檔案大小 |
|--------|--------|------|------|------|---------|
| 224×224 | BCSS | 30,760 | 5,150 | 5,150 | ~4.8GB |
| 512×512 | BCSS_512 | 6,000 | 1,500 | 1,500 | ~5.1GB |

- 查看資料集可透過 `inspect_data.py` 檔案確認，會執行後會呈現原圖、遮罩與疊圖呈現
- 須將原始資料集放置於該架構：
  ```
    ├── data/
    │   ├── BCSS/                   # 224x224 原始資料
    │   │   ├── train/
    │   │   ├── train_mask/
    │   │   ├── val/
    │   │   └── val_mask/
    │   │
    │   └── BCSS_512/               # 512x512 高解析度資料
    │       ├── train_512/
    │       ├── train_mask_512/
    │       ├── val_512/
    │       └── val_mask_512/
  ```

## 🔧 核心腳本說明

### `Bcss_auto_run_V2.1.py`
- 批次訓練管理器
- 支援 v2（原始）與 v2.1（預處理）雙模式
- 自動路徑覆蓋，無需手動編輯配置
- 完整日誌與錯誤追蹤

### `run_test_evaluation.py`
- 自動掃描訓練結果
- 執行測試評估（支援 8 卡並行）
- 解析 mIoU / mDice / aAcc 指標
- 輸出 JSON 結果與摘要表

### `visualize_results.py`
- 繪製訓練曲線
- 導出指標 CSV
- 智慧樣本篩選（基於 mask 標籤豐富度與熵）
- 生成推論對比圖

## 📈 訓練指標

訓練過程中監控的指標：
- **訓練指標**：Cross Entropy Loss、Dice Loss
- **驗證指標**：mIoU（平均 Intersection over Union）、mDice、aAcc（整體準確率）

典型訓練曲線保存在 `metrics_comparison.png`；詳細數據在 `metrics_summary.csv`。

## 🎯 模型配置

所有模型配置在 `mmsegmentation/configs/configs_comparison/`：

| 配置檔案 | 模型 | 解析度 | 類別數 |
|---------|------|--------|--------|
| `unet_224_config.py` | UNet | 224×224 | 3 (background, tumor, other) |
| `unet_512_config.py` | UNet | 512×512 | 22 (BCSS 完整) |
| `deeplabv3plus_224_config.py` | DeepLabV3+ | 224×224 | 3 |
| `deeplabv3plus_512_config.py` | DeepLabV3+ | 512×512 | 22 |
| `segformer_224_config.py` | SegFormer | 224×224 | 3 |
| `segformer_512_config.py` | SegFormer | 512×512 | 22 |

基礎配置：
- `base_224_ds.py` - 224 共用配置
- `base_512_ds.py` - 512 共用配置

## 📝 資料正規化

使用 BCSS 資料集計算的通道統計值進行正規化。詳見 `data_V2/preprocess_data/normalization_example`。

## 🐛 常見問題

### GPU 不足
若機器 < 8 卡，修改：
- `Bcss_auto_run_V2.1.py` 中的 `dist_train.sh` 參數
- 或改用 `python tools/train.py` 進行單卡訓練

### 資料路徑錯誤
確認：
- `data/BCSS_MMSEG_FINAL` 與 `data/BCSS_512_MMSEG_FINAL` 存在
- 或預處理資料在 `data_V2/preprocess_data/` 下

### 訓練中斷恢復
從最新的 checkpoint 繼續：
```bash
python tools/train.py <config> --resume-from <checkpoint_path>
```

### 如有問題，請檢查：
1. `training_errors/` 或 `training_errors_p/` 的錯誤日誌
2. 各模型目錄的 `*_train.log`
3. 本 README 的常見問題章節

<!--
## 📚 相關文檔

- [QUICK_START_TRAINER_V2P.md](QUICK_START_TRAINER_V2P.md) - 訓練快速指南
- [VISUALIZATION_GUIDE_V2P.md](VISUALIZATION_GUIDE_V2P.md) - 可視化詳細說明
- [TRAINER_MODIFICATION_SUMMARY.md](TRAINER_MODIFICATION_SUMMARY.md) - 技術細節
-->

## 📜 許可與致謝

- **MMSegmentation**: OpenMMLab https://github.com/open-mmlab/mmsegmentation
- **BCSS 資料集**: Breast Cancer Semantic Segmentation | link: https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss

## 📚 Paper
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
- Enhancing U-Net Segmentation Accuracy Through Comprehensive Data Preprocessing

## 👤 作者

- 國立中央大學 114學年度 第一學期 深度學習介紹-課程小組-第十六組-專案團隊 (2025) 
- 成員: 康祐典, 蔡善祥, 吳秉宸, 洪翊婕
- 課程簡報介紹：https://www.canva.com/design/DAG67ojVSF0/5dj06vKaDra6ereud10Ggg/view?utm_content=DAG67ojVSF0&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=ha36cdeb3be
