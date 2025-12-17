import os
import glob
import json
import random
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmseg.apis import init_model, inference_model, show_result_pyplot

# 設定圖表風格
plt.style.use('seaborn-v0_8-whitegrid')


class Config:
    """可視化配置，支援 v2 / v2.1 兩種模式。"""

    MODELS = ["unet", "deeplabv3plus", "segformer"]
    RESOLUTIONS = [224, 512]
    NUM_SAMPLE_IMAGES_PER_RES = 4  # 每個解析度取 4 張

    def __init__(self, mode: str = "v2"):
        self.mode = mode
        if mode == "v2.1":
            self.WORK_DIR_ROOT = "/home/henrywu123/DLFinal/work_dirs_p"
            self.RESULT_DIR = "/home/henrywu123/DLFinal/final_result_p"
            # 使用預處理後的 mmseg 結構測試集
            self.TEST_IMG_PATH_224 = "/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_MMSEG_FINAL/img_dir/test"
            self.TEST_ANN_PATH_224 = "/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_MMSEG_FINAL/ann_dir/test"
            self.TEST_IMG_PATH_512 = "/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_512_MMSEG_FINAL/img_dir/test"
            self.TEST_ANN_PATH_512 = "/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_512_MMSEG_FINAL/ann_dir/test"
        else:
            self.WORK_DIR_ROOT = "/home/henrywu123/DLFinal/mmsegmentation/work_dirs"
            self.RESULT_DIR = "/home/henrywu123/DLFinal/final_results"
            self.TEST_IMG_PATH_224 = "/home/henrywu123/DLFinal/data/BCSS_MMSEG_FINAL/img_dir/test"
            self.TEST_ANN_PATH_224 = "/home/henrywu123/DLFinal/data/BCSS_MMSEG_FINAL/ann_dir/test"
            self.TEST_IMG_PATH_512 = "/home/henrywu123/DLFinal/data/BCSS_512_MMSEG_FINAL/img_dir/test"
            self.TEST_ANN_PATH_512 = "/home/henrywu123/DLFinal/data/BCSS_512_MMSEG_FINAL/ann_dir/test"

        self.CONFIG_DIR = "/home/henrywu123/DLFinal/mmsegmentation/configs/configs_comparison"


def select_mode() -> str:
    """互動式選擇可視化模式: v2（原始）或 v2.1（預處理）。"""
    print("\n可視化模式選擇:")
    print("=" * 80)
    print("  1. v2 (原始資料集) - 可視化 work_dirs 結果")
    print("  2. v2.1 (預處理資料集) - 可視化 work_dirs_p 結果")
    print("=" * 80)

    while True:
        choice = input("\n請選擇可視化模式 (1 或 2): ").strip()
        if choice == "1":
            print("已選擇: v2 模式 (原始資料集)")
            return "v2"
        if choice == "2":
            print("已選擇: v2.1 模式 (預處理資料集)")
            return "v2.1"
        print("錯誤: 請輸入 1 或 2")


def parse_mmseg_log(work_dir: str):
    """讀取 work_dir 中最新的 vis_data json log，回傳統計結果。"""
    json_logs = sorted(
        glob.glob(os.path.join(work_dir, "**/vis_data/*.json"), recursive=True)
    )
    json_logs = [p for p in json_logs if "scalars.json" not in p]
    if not json_logs:
        print(f"No logs found in {work_dir}")
        return None

    latest_log = json_logs[-1]
    train_loss_map = {}
    train_dice_loss_map = {}
    val_metrics = {}

    with open(latest_log, "r") as f:
        for line in f:
            try:
                log_dict = json.loads(line.strip())
            except Exception:
                continue

            if "loss" in log_dict and "epoch" in log_dict:
                ep = log_dict["epoch"]
                train_loss_map.setdefault(ep, []).append(log_dict["loss"])
                if "decode.loss_dice" in log_dict:
                    train_dice_loss_map.setdefault(ep, []).append(
                        log_dict["decode.loss_dice"]
                    )

            if "mIoU" in log_dict and "step" in log_dict:
                ep = log_dict["step"]
                val_metrics[ep] = {
                    "mIoU": log_dict["mIoU"],
                    "mDice": log_dict.get("mDice", 0),
                    "aAcc": log_dict.get("aAcc", 0),
                }

    epochs = sorted(val_metrics.keys())
    mious = [val_metrics[ep]["mIoU"] for ep in epochs]
    mdices = [val_metrics[ep]["mDice"] for ep in epochs]
    vaccs = [val_metrics[ep]["aAcc"] for ep in epochs]

    avg_losses = []
    avg_dice_losses = []
    for ep in epochs:
        if ep in train_loss_map:
            avg_losses.append(np.mean(train_loss_map[ep]))
            avg_dice_losses.append(np.mean(train_dice_loss_map.get(ep, [0])))
        else:
            avg_losses.append(avg_losses[-1] if avg_losses else 0)
            avg_dice_losses.append(avg_dice_losses[-1] if avg_dice_losses else 0)

    return {
        "epochs": epochs,
        "train_loss": avg_losses,
        "val_loss": avg_dice_losses,
        "val_iou": mious,
        "val_dice": mdices,
        "val_acc": vaccs,
    }


def export_metrics_csv(config: Config):
    """匯出所有模型/解析度的訓練指標。"""
    Path(config.RESULT_DIR).mkdir(parents=True, exist_ok=True)
    csv_path = Path(config.RESULT_DIR) / "metrics_summary.csv"
    rows = []

    for model in config.MODELS:
        for res in config.RESOLUTIONS:
            folder = f"{model}_{res}"
            path = Path(config.WORK_DIR_ROOT) / folder
            data = parse_mmseg_log(str(path))
            if not data:
                continue
            for i, ep in enumerate(data["epochs"]):
                rows.append(
                    {
                        "model": model,
                        "resolution": res,
                        "epoch": ep,
                        "train_loss": data["train_loss"][i],
                        "val_loss": data["val_loss"][i],
                        "val_iou": data["val_iou"][i],
                        "val_dice": data["val_dice"][i],
                        "val_acc": data["val_acc"][i],
                    }
                )

    fieldnames = [
        "model",
        "resolution",
        "epoch",
        "train_loss",
        "val_loss",
        "val_iou",
        "val_dice",
        "val_acc",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics CSV to {csv_path}")


def plot_metrics(config: Config):
    """繪製訓練/驗證指標曲線。"""
    Path(config.RESULT_DIR).mkdir(parents=True, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "unet_224": "#2E86AB",
        "unet_512": "#06A77D",
        "deeplabv3plus_224": "#F18F01",
        "deeplabv3plus_512": "#C73E1D",
        "segformer_224": "#6A994E",
        "segformer_512": "#BC4B51",
    }

    labels = {
        "unet_224": "Unet (224)",
        "unet_512": "Unet (512)",
        "deeplabv3plus_224": "DeepLabV3Plus (224)",
        "deeplabv3plus_512": "DeepLabV3Plus (512)",
        "segformer_224": "Segformer (224)",
        "segformer_512": "Segformer (512)",
    }

    for model in config.MODELS:
        for res in config.RESOLUTIONS:
            key = f"{model}_{res}"
            path = Path(config.WORK_DIR_ROOT) / key
            data = parse_mmseg_log(str(path))
            if not data or not data["epochs"]:
                continue

            label = labels.get(key, key)
            color = colors.get(key, "gray")

            ax1.plot(data["epochs"], data["train_loss"], label=label, color=color, marker="o", markersize=4)
            ax2.plot(data["epochs"], data["val_loss"], label=label, color=color, marker="o", markersize=4)
            ax3.plot(data["epochs"], data["val_iou"], label=label, color=color, marker="o", markersize=4)
            ax4.plot(data["epochs"], data["val_dice"], label=label, color=color, marker="o", markersize=4)

    ax1.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Validation IoU", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("IoU")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4.set_title("Validation Dice", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Dice")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    mode_label = "v2.1 (Preprocessed)" if config.mode == "v2.1" else "v2 (Original)"
    plt.suptitle(f"114 DL FINAL - {mode_label}", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_path = Path(config.RESULT_DIR) / "metrics_comparison.png"
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved metrics plot to {out_path}")


def _score_mask(mask_path: Path) -> float:
    """為 mask 計分，選出更有辨識度的樣本。

    計分策略：
    - 唯一標籤數量（越多越好）
    - 標籤分布熵（越高代表內容更豐富）
    """

    if not mask_path.exists():
        return -1
    mask = mmcv.imread(str(mask_path), flag="grayscale")
    if mask is None:
        return -1
    vals, counts = np.unique(mask, return_counts=True)
    # 排除全空或無效情況
    if len(vals) == 0:
        return -1
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return len(vals) + entropy


def _collect_samples(config: Config):
    """選擇較有辨識度的樣本：224 和 512 各選 4 張。
    
    Returns:
        dict: {'224': [(img, ann, res), ...], '512': [(img, ann, res), ...]}
    """

    def gather(img_dir: Path, ann_dir: Path, res: int):
        """收集單一解析度的候選樣本"""
        candidates = []
        if not img_dir.exists() or not ann_dir.exists():
            return candidates
        for img_path in img_dir.glob("*.png"):
            ann_path = ann_dir / img_path.name
            if not ann_path.exists():
                continue
            score = _score_mask(ann_path)
            if score < 0:
                continue
            candidates.append((score, img_path, ann_path, res))
        return candidates

    # 分別收集 224、512 候選
    candidates_224 = gather(Path(config.TEST_IMG_PATH_224), Path(config.TEST_ANN_PATH_224), 224)
    candidates_512 = gather(Path(config.TEST_IMG_PATH_512), Path(config.TEST_ANN_PATH_512), 512)

    # 各自排序並取前 NUM_SAMPLE_IMAGES_PER_RES 張
    samples_224 = []
    samples_512 = []
    
    if candidates_224:
        candidates_224.sort(key=lambda x: x[0], reverse=True)
        picks_224 = candidates_224[: config.NUM_SAMPLE_IMAGES_PER_RES]
        samples_224 = [(img, ann, res) for _, img, ann, res in picks_224]
    
    if candidates_512:
        candidates_512.sort(key=lambda x: x[0], reverse=True)
        picks_512 = candidates_512[: config.NUM_SAMPLE_IMAGES_PER_RES]
        samples_512 = [(img, ann, res) for _, img, ann, res in picks_512]
    
    return {'224': samples_224, '512': samples_512}


def visualize_predictions(config: Config):
    """可視化預測：224 圖只跑 224 模型，512 圖只跑 512 模型。"""
    Path(config.RESULT_DIR).mkdir(parents=True, exist_ok=True)

    samples_dict = _collect_samples(config)
    samples_224 = samples_dict.get('224', [])
    samples_512 = samples_dict.get('512', [])
    
    if not samples_224 and not samples_512:
        print("No test images found.")
        return

    print(f"\n找到測試樣本: {len(samples_224)} 張 224×224, {len(samples_512)} 張 512×512")

    # 處理所有樣本：224 和 512
    all_samples = samples_224 + samples_512
    
    for img_path, ann_path, img_res in all_samples:
        img_path = Path(img_path)
        ann_path = Path(ann_path)
        if not ann_path.exists():
            print(f"Annotation not found for {img_path.name}, skip.")
            continue

        print(f"Visualizing inference on: {img_path.name} (resolution: {img_res})")
        img = mmcv.imread(str(img_path))
        img_rgb = mmcv.bgr2rgb(img)
        gt_mask = mmcv.imread(str(ann_path), flag="grayscale")

        # 只跑對應解析度的模型（3 個模型）
        model_configs = [(m, img_res) for m in config.MODELS]
        
        fig = plt.figure(figsize=(18, 12))  # 3 個模型，高度減少
        row_idx = 0

        for model_name, model_res in model_configs:
            work_dir = Path(config.WORK_DIR_ROOT) / f"{model_name}_{model_res}"
            cfg_file = Path(config.CONFIG_DIR) / f"{model_name}_{model_res}_config.py"
            ckpt_pattern = work_dir / "best_mIoU_epoch_*.pth"
            ckpts = glob.glob(str(ckpt_pattern))
            if not ckpts:
                print(f"Checkpoint not found for {model_name}_{model_res}")
                row_idx += 1
                continue
            checkpoint = ckpts[0]

            try:
                model = init_model(str(cfg_file), checkpoint, device="cuda:0")
                result = inference_model(model, str(img_path))
            except Exception as e:  # pragma: no cover - runtime safeguard
                print(f"✗ Error processing {model_name}_{model_res}: {e}")
                row_idx += 1
                continue

            ax = plt.subplot(len(model_configs), 4, row_idx * 4 + 1)
            ax.imshow(img_rgb)
            ax.set_title("Original", fontsize=10, fontweight="bold")
            ax.axis("off")

            ax = plt.subplot(len(model_configs), 4, row_idx * 4 + 2)
            ax.imshow(gt_mask, cmap="tab20")
            ax.set_title("Ground Truth", fontsize=10, fontweight="bold")
            ax.axis("off")

            ax = plt.subplot(len(model_configs), 4, row_idx * 4 + 3)
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
            ax.imshow(pred_mask, cmap="tab20")
            ax.set_title("Prediction", fontsize=10, fontweight="bold")
            ax.axis("off")

            ax = plt.subplot(len(model_configs), 4, row_idx * 4 + 4)
            vis_img = show_result_pyplot(model, str(img_path), result, show=False, out_file=None)
            ax.imshow(mmcv.bgr2rgb(vis_img))
            ax.set_title(f"{model_name} ({model_res})", fontsize=10, fontweight="bold")
            ax.axis("off")

            row_idx += 1

        mode_label = "v2.1 (Preprocessed)" if config.mode == "v2.1" else "v2 (Original)"
        plt.suptitle(
            f"Segmentation Results - {mode_label} [{img_res}×{img_res}]\n{img_path.name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        out_name = img_path.stem
        out_path = Path(config.RESULT_DIR) / f"prediction_results_{img_res}_{out_name}.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved prediction results to {out_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("模型可視化工具 - 自適應模式版本")
    print("=" * 80)

    mode = select_mode()
    cfg = Config(mode=mode)

    print(f"\n開始可視化 {mode} 模式的訓練結果...")
    print(f"工作目錄: {cfg.WORK_DIR_ROOT}")
    print(f"輸出目錄: {cfg.RESULT_DIR}")
    print(f"測試圖片 (224): {cfg.TEST_IMG_PATH_224}")
    print(f"測試圖片 (512): {cfg.TEST_IMG_PATH_512}")
    print("=" * 80)

    plot_metrics(cfg)
    export_metrics_csv(cfg)
    visualize_predictions(cfg)