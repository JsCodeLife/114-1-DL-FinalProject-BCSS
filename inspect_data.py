import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ================= 設定區 =================
# 可以切換要檢查 224 還是 512
DATA_ROOT = './data/BCSS'       # 檢查 224
# DATA_ROOT = './data/BCSS_512'   # 檢查 512

# 設定輸出圖片的存檔資料夾
OUTPUT_DIR = './inspect_data'
# ==========================================

def get_paths(root_dir, split='train'):
    """根據路徑規則取得圖片與 Mask 資料夾"""
    # 判斷是否為 512 資料集
    suffix = '_512' if '512' in root_dir else ''
    
    img_dir = os.path.join(root_dir, f"{split}{suffix}")
    mask_dir = os.path.join(root_dir, f"{split}_mask{suffix}")
    
    return img_dir, mask_dir

def mask_to_rgb(mask):
    """將 Mask (0, 1, 2...) 轉換成 RGB 顏色以便視覺化"""
    colors = np.array([
        [0, 0, 0],       # Class 0: Background (黑)
        [255, 0, 0],     # Class 1: Tumor (紅)
        [0, 255, 0],     # Class 2: Stroma (綠)
        [0, 0, 255],     # Class 3: Inflammatory (藍)
        [255, 255, 0],   # Class 4: Necrosis (黃)
        [0, 255, 255],   # Class 5: Other (青)
    ])
    
    # 避免 mask 值超過顏色表範圍 (防呆)
    max_val = mask.max()
    if max_val >= len(colors):
        # 動態補齊顏色
        extra_colors = np.random.randint(0, 255, (max_val - len(colors) + 1, 3))
        colors = np.vstack([colors, extra_colors])

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(len(colors)):
        rgb[mask == i] = colors[i]
        
    return rgb

def inspect_dataset():
    print(f"🔍  正在檢查資料集路徑: {DATA_ROOT}")
    
    # 1. 檢查並建立輸出資料夾
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已建立輸出資料夾: {OUTPUT_DIR}")
    else:
        print(f"📁 輸出資料夾已存在: {OUTPUT_DIR}")

    splits = ['train', 'val']
    total_images = 0
    
    for split in splits:
        img_dir, mask_dir = get_paths(DATA_ROOT, split)
        
        if not os.path.exists(img_dir):
            print(f"❌ 找不到資料夾: {img_dir}")
            continue
            
        # 搜尋圖片
        images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        
        print(f"   📂 [{split.upper()}] 圖片: {len(images)} 張 | Mask: {len(masks)} 張")
        total_images += len(images)
        
        # 隨機抽樣 3 組並存檔
        if len(images) > 0:
            num_samples = 3
            samples = random.sample(images, num_samples)
            print(f"   🎨 正在生成 {split} 的 {num_samples} 張範例圖片...")
            
            for i, img_path in enumerate(samples):
                filename = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir, filename)
                
                if not os.path.exists(mask_path):
                    print(f"      ⚠️ Warning: 對應的 Mask 不存在 ({filename})")
                    continue

                # 讀取
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path)
                mask_np = np.array(mask)
                
                # 轉 RGB
                mask_rgb = mask_to_rgb(mask_np)
                
                # 繪圖 (1列3行: 原圖 | Mask | 疊合)
                plt.figure(figsize=(15, 5))
                
                # 原圖
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title(f"Original: {filename}")
                plt.axis('off')
                
                # Mask
                plt.subplot(1, 3, 2)
                plt.imshow(mask_rgb)
                plt.title(f"GT Mask (Max Class: {mask_np.max()})")
                plt.axis('off')
                
                # 疊合
                plt.subplot(1, 3, 3)
                plt.imshow(img)
                plt.imshow(mask_rgb, alpha=0.4)
                plt.title("Overlay")
                plt.axis('off')
                
                # 存檔
                save_name = f"{split}_sample_{i+1}_{filename}"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                plt.savefig(save_path)
                plt.close() # 關閉畫布釋放記憶體
                
            print(f"      已儲存範例圖片至 {OUTPUT_DIR}")

    if total_images == 0:
        print("⚠️ 警告: 未偵測到任何圖片，請檢查 DATA_ROOT 設定是否正確！")

if __name__ == '__main__':
    inspect_dataset()
