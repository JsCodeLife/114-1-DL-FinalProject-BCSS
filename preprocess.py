import os
import glob
import argparse
import yaml
import csv
import json
import numpy as np
import cv2
import shutil
from joblib import Parallel, delayed
import utils
import time

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def process_image(img_path, mask_path, ref_vectors, ref_max_c, clahe_clip):
    try:
        img_name = os.path.basename(img_path)
        img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        
        # 1. Macenko Normalization
        norm_img = utils.macenko_normalize(img, ref_vectors, ref_max_c)
        
        # 2. CLAHE (Local Contrast)
        clahe_img = utils.apply_clahe_lab(norm_img, clip_limit=clahe_clip)
        
        # 3. Background Whitening (REMOVED)
        final_img = clahe_img

        return img_name, final_img, "success"
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return os.path.basename(img_path), None, str(e)

def main():
    parser = argparse.ArgumentParser(description="BCSS Preprocessing Pipeline (Visual Enhancement)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--raw_dir", help="Override raw directory")
    parser.add_argument("--out_dir", help="Override output directory")
    parser.add_argument("--workers", type=int, help="Number of workers")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = args.raw_dir or config["raw_dir"]
    out_dir = args.out_dir or config["out_dir"]
    n_workers = args.workers or config.get("workers", 1)
    
    img_dir = os.path.join(raw_dir, "images")
    mask_dir = os.path.join(raw_dir, "masks")
    
    processed_img_dir = os.path.join(out_dir, "images")
    processed_mask_dir = os.path.join(out_dir, "masks")
    
    ensure_dir(processed_img_dir)
    ensure_dir(processed_mask_dir)
    
    log_file = os.path.join(out_dir, "../logs/preprocess.log")
    ensure_dir(os.path.dirname(log_file))
    
    manifest_file = os.path.join(out_dir, "manifest.csv")
    stats_file = os.path.join(out_dir, "dataset_stats.json")

    # Get Files
    all_files = glob.glob(os.path.join(img_dir, "*.png"))
    print(f"Found {len(all_files)} images.")

    # Validation & Reference Selection
    valid_files = []
    for f in all_files:
        bn = os.path.basename(f)
        mask_path = os.path.join(mask_dir, bn)
        if os.path.exists(mask_path):
            valid_files.append((f, mask_path))
        else:
            with open(log_file, "a") as log:
                log.write(f"{bn},skipped,missing_mask\n")
    
    if not valid_files:
        print("No valid files found!")
        return

    # Select Reference
    ref_idx = config["macenko"].get("ref_idx") or 0
    ref_img_path = valid_files[ref_idx][0]
    print(f"Using reference image: {ref_img_path}")
    
    ref_img = np.array(cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB))
    ref_vectors, ref_max_c = utils.get_stain_vectors(ref_img)
    
    if ref_vectors is None:
        print("Error: Could not extract stain vectors from reference image.")
        return

    # Processing Pipeline
    clahe_clip = config["clahe"].get("clip_limit", 2.0)
    print("Starting processing pipeline (Macenko -> CLAHE)...")
    
    results = Parallel(n_jobs=n_workers)(
        delayed(process_image)(f, m, ref_vectors, ref_max_c, clahe_clip) for f, m in valid_files
    )
    
    manifest_rows = []
    processed_paths = []
    
    for (f_orig, m_orig), (res_name, res_img, status) in zip(valid_files, results):
        if status == "success":
            out_img_path = os.path.join(processed_img_dir, res_name)
            out_mask_path = os.path.join(processed_mask_dir, res_name)
            
            cv2.imwrite(out_img_path, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
            shutil.copy(m_orig, out_mask_path)
            
            processed_paths.append(out_img_path)
            manifest_rows.append([res_name, f_orig, out_img_path, "success", ""])
            
            # Generate visual comparison for first 5
            if len(processed_paths) <= 5:
                comp_dir = os.path.join(out_dir, "../comparison")
                ensure_dir(comp_dir)
                orig = cv2.imread(f_orig)
                final_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
                combined = np.hstack((orig, final_bgr))
                cv2.imwrite(os.path.join(comp_dir, f"comp_{res_name}"), combined)
        else:
            with open(log_file, "a") as log:
                log.write(f"{res_name},error,{status}\n")

    # Final Stats Calculation
    print("Calculating dataset statistics for runtime normalization...")
    mean, std = utils.compute_stats(processed_paths)
    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")
    
    stats_data = {
        "description": "Computed on processed (Macenko+CLAHE) images. Use these for transforms.Normalize.",
        "mean_rgb": mean,
        "std_rgb": std
    }
    
    with open(stats_file, "w") as f:
        json.dump(stats_data, f, indent=4)

    # Write Manifest
    with open(manifest_file, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "raw_path", "processed_path", "status", "notes"])
        writer.writerows(manifest_rows)

    print(f"Pipeline completed. Stats saved to {stats_file}")

if __name__ == "__main__":
    main()
