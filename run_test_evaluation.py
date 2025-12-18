#!/usr/bin/env python3
"""
批量執行模型測試評估
對已訓練完成的模型在 test set 上評估性能
"""
import os
import subprocess
import json
import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')


def find_best_checkpoint(work_dir):
    """尋找最佳 checkpoint"""
    work_dir = Path(work_dir)
    
    # 優先尋找 best_mIoU
    best_miou = list(work_dir.glob('best_mIoU_epoch_*.pth'))
    if best_miou:
        return best_miou[0]
    
    # 其次尋找 best_mDice
    best_dice = list(work_dir.glob('best_mDice_epoch_*.pth'))
    if best_dice:
        return best_dice[0]
    
    # 最後尋找 latest.pth
    latest = work_dir / 'latest.pth'
    if latest.exists():
        return latest
    
    return None


def run_test(config_path, checkpoint_path, work_dir, mode='v2'):
    """執行單個模型的測試評估
    
    Args:
        config_path: 配置檔案路徑
        checkpoint_path: 權重檔案路徑
        work_dir: 工作目錄
        mode: 'v2' 或 'v2.1'
    """
    config_name = Path(config_path).stem.replace('_config', '')
    
    print(f"\n{'='*80}")
    print(f"測試評估: {config_name}")
    print(f"模式: {mode}")
    print(f"配置: {config_path}")
    print(f"權重: {checkpoint_path}")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 準備測試命令
    cmd = [
        'bash',
        'tools/dist_test.sh',
        config_path,
        str(checkpoint_path),
        '8'
    ]
    
    # v2.1 模式：添加資料集覆蓋選項
    if mode == 'v2.1':
        cfg_options = build_cfg_options(config_name)
        if cfg_options:
            cmd.extend(['--cfg-options'] + cfg_options)
    
    # 執行測試
    try:
        result = subprocess.run(
            cmd,
            cwd='/home/henrywu123/DLFinal/mmsegmentation',
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ {config_name} 測試完成！")
            
            # 解析結果
            output = result.stdout
            metrics = extract_metrics(output)
            
            # 保存結果
            save_test_results(config_name, metrics, work_dir, mode)
            
            return True, metrics
        else:
            print(f"✗ {config_name} 測試失敗！")
            print(f"錯誤輸出:\n{result.stderr}")
            return False, None
            
    except Exception as e:
        print(f"✗ {config_name} 執行錯誤: {e}")
        return False, None


def build_cfg_options(config_name):
    """根據配置名稱為 v2.1 模式生成資料集覆蓋選項"""
    # 判斷是 224 還是 512 配置
    if '224' in config_name.lower():
        return [
            'test_dataloader.dataset.data_root=/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_MMSEG_FINAL',
            'test_dataloader.dataset.data_prefix.img_path=img_dir/test',
            'test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/test',
        ]
    elif '512' in config_name.lower():
        return [
            'test_dataloader.dataset.data_root=/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_512_MMSEG_FINAL',
            'test_dataloader.dataset.data_prefix.img_path=img_dir/test',
            'test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/test',
        ]
    return None


def extract_metrics(output):
    """從輸出中提取指標（支援多種格式）"""
    metrics = {}
    
    lines = output.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # 方法 1: 表格格式 | mIoU | 值 |
        if '|' in line:
            if 'miou' in line_lower:
                parts = line.split('|')
                for j, part in enumerate(parts):
                    if 'miou' in part.lower() and j + 1 < len(parts):
                        try:
                            metrics['test_mIoU'] = float(parts[j + 1].strip())
                        except:
                            pass
            
            if 'mdice' in line_lower:
                parts = line.split('|')
                for j, part in enumerate(parts):
                    if 'mdice' in part.lower() and j + 1 < len(parts):
                        try:
                            metrics['test_mDice'] = float(parts[j + 1].strip())
                        except:
                            pass
            
            if 'aacc' in line_lower:
                parts = line.split('|')
                for j, part in enumerate(parts):
                    if 'aacc' in part.lower() and j + 1 < len(parts):
                        try:
                            metrics['test_aAcc'] = float(parts[j + 1].strip())
                        except:
                            pass
        
        # 方法 2: 鍵值對格式 mIoU: 0.75
        if 'miou' in line_lower and ':' in line:
            try:
                value_str = line.split(':')[-1].strip().split()[0]
                metrics['test_mIoU'] = float(value_str)
            except:
                pass
        
        if 'mdice' in line_lower and ':' in line:
            try:
                value_str = line.split(':')[-1].strip().split()[0]
                metrics['test_mDice'] = float(value_str)
            except:
                pass
        
        if 'aacc' in line_lower and ':' in line:
            try:
                value_str = line.split(':')[-1].strip().split()[0]
                metrics['test_aAcc'] = float(value_str)
            except:
                pass
    
    return metrics


def save_test_results(config_name, metrics, work_dir, mode):
    """保存測試結果到 JSON 檔案"""
    work_dir = Path(work_dir)
    result_file = work_dir / f'test_results_{mode}.json'
    
    result = {
        'model': config_name,
        'mode': mode,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"測試結果已保存至: {result_file}")


def export_test_results_csv(all_results):
    """匯出測試結果到 CSV"""
    csv_path = Path('/home/henrywu123/DLFinal/final_results/test_metrics_summary.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for model_key, metrics in sorted(all_results.items()):
        mode, model_name = model_key.split('_', 1)
        rows.append({
            'mode': mode,
            'model': model_name,
            'test_mIoU': metrics.get('test_mIoU', 'N/A'),
            'test_mDice': metrics.get('test_mDice', 'N/A'),
            'test_aAcc': metrics.get('test_aAcc', 'N/A')
        })
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['mode', 'model', 'test_mIoU', 'test_mDice', 'test_aAcc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ 測試指標 CSV 已保存至: {csv_path}")


def visualize_test_results(all_results):
    """可視化測試結果（長條圖對比）"""
    output_dir = Path('/home/henrywu123/DLFinal/final_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分組資料
    v2_data = {}
    v2p_data = {}
    
    for model_key, metrics in all_results.items():
        if model_key.startswith('v2.1_'):
            model_name = model_key.replace('v2.1_', '')
            v2p_data[model_name] = metrics
        elif model_key.startswith('v2_'):
            model_name = model_key.replace('v2_', '')
            v2_data[model_name] = metrics
    
    # 整理模型名稱與指標
    models = sorted(set(list(v2_data.keys()) + list(v2p_data.keys())))
    
    if not models:
        print("無資料可視化。")
        return
    
    # 準備資料
    v2_miou = [v2_data.get(m, {}).get('test_mIoU', 0) for m in models]
    v2p_miou = [v2p_data.get(m, {}).get('test_mIoU', 0) for m in models]
    
    v2_mdice = [v2_data.get(m, {}).get('test_mDice', 0) for m in models]
    v2p_mdice = [v2p_data.get(m, {}).get('test_mDice', 0) for m in models]
    
    # 過濾非數值
    v2_miou = [x if isinstance(x, (int, float)) else 0 for x in v2_miou]
    v2p_miou = [x if isinstance(x, (int, float)) else 0 for x in v2p_miou]
    v2_mdice = [x if isinstance(x, (int, float)) else 0 for x in v2_mdice]
    v2p_mdice = [x if isinstance(x, (int, float)) else 0 for x in v2p_mdice]
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # mIoU Comparison
    bars1 = ax1.bar(x - width/2, v2_miou, width, label='v2 (Original)', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, v2p_miou, width, label='v2.1 (Preprocessed)', color='#F18F01', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test mIoU', fontsize=12, fontweight='bold')
    ax1.set_title('Test Set mIoU Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上標註數值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # mDice Comparison
    bars3 = ax2.bar(x - width/2, v2_mdice, width, label='v2 (Original)', color='#6A994E', alpha=0.8)
    bars4 = ax2.bar(x + width/2, v2p_mdice, width, label='v2.1 (Preprocessed)', color='#BC4B51', alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test mDice', fontsize=12, fontweight='bold')
    ax2.set_title('Test Set mDice Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model Test Evaluation Results Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = output_dir / 'test_metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 測試指標可視化已保存至: {output_path}")


def main():
    config_dir = Path('/home/henrywu123/DLFinal/mmsegmentation/configs/configs_comparison')
    
    print("=" * 80)
    print("模型測試評估工具")
    print("=" * 80)
    
    # 選擇模式
    print("\n測試模式選擇:")
    print("  1. v2 (原始資料集)")
    print("  2. v2.1 (預處理資料集)")
    print("  3. 兩者都測試")
    
    while True:
        choice = input("\n請選擇測試模式 (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("錯誤: 請輸入 1, 2 或 3")
    
    modes_to_test = []
    if choice == '1':
        modes_to_test = [('v2', '/home/henrywu123/DLFinal/mmsegmentation/work_dirs')]
    elif choice == '2':
        modes_to_test = [('v2.1', '/home/henrywu123/DLFinal/work_dirs_p')]
    else:
        modes_to_test = [
            ('v2', '/home/henrywu123/DLFinal/mmsegmentation/work_dirs'),
            ('v2.1', '/home/henrywu123/DLFinal/work_dirs_p')
        ]
    
    # 收集所有可測試的模型
    all_results = {}
    
    for mode, work_dir_base in modes_to_test:
        work_dir_base = Path(work_dir_base)
        
        if not work_dir_base.exists():
            print(f"\n⚠️  工作目錄不存在: {work_dir_base}")
            continue
        
        print(f"\n{'='*80}")
        print(f"掃描 {mode} 模式的訓練結果...")
        print(f"{'='*80}")
        
        # 尋找所有模型目錄
        model_dirs = [d for d in work_dir_base.iterdir() if d.is_dir()]
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            
            # 跳過空目錄（通常是備份目錄）
            has_checkpoint = find_best_checkpoint(model_dir) is not None
            if not has_checkpoint:
                print(f"⚠️  跳過空目錄: {model_name} (無 checkpoint)")
                continue
            
            # 移除可能已存在的 _config 後綴，避免重複
            if model_name.endswith('_config'):
                config_name = model_name
            else:
                config_name = f"{model_name}_config"
            config_path = config_dir / f"{config_name}.py"
            
            if not config_path.exists():
                print(f"⚠️  配置檔案不存在: {config_path}")
                continue
            
            # 尋找最佳 checkpoint
            checkpoint = find_best_checkpoint(model_dir)
            
            if checkpoint is None:
                print(f"⚠️  未找到 checkpoint: {model_dir}")
                continue
            
            print(f"\n找到模型: {model_name}")
            print(f"  Checkpoint: {checkpoint.name}")
            
            # 執行測試
            success, metrics = run_test(
                str(config_path),
                checkpoint,
                model_dir,
                mode
            )
            
            if success:
                all_results[f"{mode}_{model_name}"] = metrics
    
    # 顯示總結
    print(f"\n{'='*80}")
    print("測試總結:")
    print(f"{'='*80}\n")
    
    if all_results:
        print(f"{'模型':<30} {'Test mIoU':<12} {'Test mDice':<12} {'Test aAcc':<12}")
        print("-" * 80)
        
        for model_key, metrics in sorted(all_results.items()):
            miou = metrics.get('test_mIoU', 'N/A')
            mdice = metrics.get('test_mDice', 'N/A')
            aacc = metrics.get('test_aAcc', 'N/A')
            
            miou_str = f"{miou:.2f}" if isinstance(miou, (int, float)) else miou
            mdice_str = f"{mdice:.2f}" if isinstance(mdice, (int, float)) else mdice
            aacc_str = f"{aacc:.2f}" if isinstance(aacc, (int, float)) else aacc
            
            print(f"{model_key:<30} {miou_str:<12} {mdice_str:<12} {aacc_str:<12}")
        
        # 匯出 CSV 與可視化
        export_test_results_csv(all_results)
        visualize_test_results(all_results)
    else:
        print("未成功執行任何測試。")
    
    print(f"\n完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
