import os
import glob
import argparse
import yaml
import json
import numpy as np
import cv2
import shutil
from joblib import Parallel, delayed
import utils
import time
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        print(f"Warning: tqdm not installed. Install with 'pip install tqdm' for progress bar.")
        return iterable

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def process_batch_image(img_path, out_img_path, ref_vectors, ref_max_c, clahe_clip):
    try:
        # Load
        img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        
        # 1. Macenko
        norm_img = utils.macenko_normalize(img, ref_vectors, ref_max_c)
        
        # 2. CLAHE
        final_img = utils.apply_clahe_lab(norm_img, clip_limit=clahe_clip)
        
        # 3. Save
        ensure_dir(os.path.dirname(out_img_path))
        cv2.imwrite(out_img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        
        return out_img_path, "success"
    except Exception as e:
        return img_path, str(e)

def main():
    parser = argparse.ArgumentParser(description="BCSS Batch Preprocessing (Macenko + CLAHE)")
    parser.add_argument("--data_root", required=True, help="Root directory containing BCSS and BCSS_512 datasets.")
    parser.add_argument("--out_dir", default="preprocess_data", help="Output directory name (relative to data_root).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    base_root = args.data_root
    out_root = os.path.join(base_root, args.out_dir)
    
    datasets = [
        {
            "name": "BCSS",
            "splits": ["train", "val", "test"],
            "stats_split": "train" # Compute stats from this split only
        },
        {
            "name": "BCSS_512",
            "splits": ["train_512", "val_512"],
            "stats_split": "train_512"
        }
    ]
    
    # Common Config
    clahe_clip = 2.0
    n_workers = args.workers
    
    # Check if base root exists
    if not os.path.exists(base_root):
        print(f"Error: Data root directory not found: {base_root}")
        return

    # 1. Select Reference from BCSS/train (First image)
    ref_dir = os.path.join(base_root, "BCSS", "train")
    if not os.path.exists(ref_dir):
        # Fallback check or error if BCSS structure is missing
        print(f"Error: Reference directory not found at {ref_dir}. Ensure 'BCSS/train' exists.")
        return

    ref_candidates = glob.glob(os.path.join(ref_dir, "*.png"))
    
    if not ref_candidates:
        print("Error: No reference images found in BCSS/train")
        return
        
    ref_img_path = ref_candidates[0]
    print(f"Using Reference Image: {ref_img_path}")
    
    ref_img = np.array(cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB))
    ref_vectors, ref_max_c = utils.get_stain_vectors(ref_img)
    
    if ref_vectors is None:
        print("Error: Failed to extract stain vectors from reference.")
        return

    # Process Loop
    for ds in datasets:
        ds_name = ds["name"]
        print(f"\nProcessing Dataset: {ds_name}...")
        
        stats_paths = []
        
        for split in ds["splits"]:
            raw_split_dir = os.path.join(base_root, ds_name, split)
            out_split_dir = os.path.join(out_root, ds_name, split)
            
            # Gather files
            img_files = glob.glob(os.path.join(raw_split_dir, "*.png"))
            print(f"  Split '{split}': Found {len(img_files)} images.")
            
            if not img_files:
                continue
                
            # Process in parallel
            # Output path mapping
            tasks = []
            for f in img_files:
                fname = os.path.basename(f)
                out_p = os.path.join(out_split_dir, fname)
                tasks.append((f, out_p))
            
            results = Parallel(n_jobs=n_workers)(
                delayed(process_batch_image)(f, out, ref_vectors, ref_max_c, clahe_clip) 
                for f, out in tqdm(tasks, desc=f"Processing {split}")
            )
            
            # Check results and collect for stats
            success_count = 0
            for res_path, status in results:
                if status == "success":
                    success_count += 1
                    if split == ds["stats_split"]:
                        stats_paths.append(res_path)
                else:
                    print(f"    Failed: {res_path} - {status}")
            
            print(f"    Processed {success_count}/{len(img_files)} images.")

        # Compute Stats for this Dataset
        print(f"  Calculating stats for {ds_name} (using {len(stats_paths)} images from {ds['stats_split']})...")
        if stats_paths:
            mean, std = utils.compute_stats(stats_paths)
            stats_out = os.path.join(out_root, ds_name, f"{ds_name.lower()}_stats.json")
            
            with open(stats_out, "w") as f:
                json.dump({
                    "dataset": ds_name,
                    "reference_image": ref_img_path,
                    "description": f"Stats computed on {ds['stats_split']} only. Applied: Macenko+CLAHE.",
                    "mean_rgb": mean,
                    "std_rgb": std
                }, f, indent=4)
            print(f"  Stats saved to {stats_out}")
        else:
            print("  Warning: No images for stats calculation.")

    print("\nBatch Processing Complete.")

if __name__ == "__main__":
    main()
