#!/usr/bin/env python3
import os
import subprocess
import glob
from collections import deque
from datetime import datetime
import re

def select_mode():
    """互動式選擇訓練模式: v2（原始資料）或 v2.1（預處理資料）"""
    print("\n訓練模式選擇:")
    print("=" * 80)
    print("  1. v2 (原始資料集) - 使用 BCSS_MMSEG_FINAL 資料")
    print("  2. v2.1 (預處理資料集) - 使用 data_V2/preprocess_data 資料")
    print("=" * 80)
    
    while True:
        choice = input("\n請選擇訓練模式 (1 或 2): ").strip()
        if choice == '1':
            print("已選擇: v2 模式 (原始資料集)")
            return 'v2'
        elif choice == '2':
            print("已選擇: v2.1 模式 (預處理資料集)")
            return 'v2.1'
        else:
            print("錯誤: 請輸入 1 或 2")

def find_config_files(config_dir):
    """自動偵測以 config.py 結尾的配置檔案"""
    pattern = os.path.join(config_dir, '*config.py')
    config_files = glob.glob(pattern)
    # 排序以確保訓練順序一致
    config_files.sort()
    return config_files

def select_configs(config_files):
    """互動式選擇要訓練的配置檔案"""
    print("\n可用的配置檔案:")
    print("=" * 80)
    for i, config in enumerate(config_files, 1):
        config_name = os.path.basename(config).replace('_config.py', '')
        print(f"  {i}. {config_name}")
    print("=" * 80)
    
    print("\n選擇方式:")
    print("  - 輸入數字選擇單個配置 (例如: 1)")
    print("  - 輸入多個數字用逗號分隔 (例如: 1,3,5)")
    print("  - 輸入範圍 (例如: 1-4)")
    print("  - 輸入 'all' 訓練全部")
    print("  - 輸入 'q' 退出")
    
    while True:
        choice = input("\n請選擇要訓練的配置: ").strip()
        
        if choice.lower() == 'q':
            print("已取消。")
            return []
        
        if choice.lower() == 'all':
            return config_files
        
        try:
            selected_indices = set()
            
            # 處理逗號分隔的選項
            parts = choice.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # 處理範圍 (例如: 1-4)
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start, end + 1))
                else:
                    # 處理單個數字
                    selected_indices.add(int(part))
            
            # 驗證選擇的索引
            if all(1 <= idx <= len(config_files) for idx in selected_indices):
                selected = [config_files[idx - 1] for idx in sorted(selected_indices)]
                
                print(f"\n已選擇 {len(selected)} 個配置:")
                for i, config in enumerate(selected, 1):
                    print(f"  {i}. {os.path.basename(config).replace('_config.py', '')}")
                
                confirm = input("\n確認開始訓練? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected
                else:
                    print("請重新選擇。")
            else:
                print(f"錯誤: 請輸入 1-{len(config_files)} 之間的數字")
        except ValueError:
            print("錯誤: 輸入格式不正確，請重試")

def train_model(config_path, mode='v2.1', work_dir_base='work_dirs', error_log_dir='training_errors'):
    """訓練單個模型，串流輸出並記錄錯誤
    
    Args:
        config_path: 配置檔案路徑
        mode: 訓練模式 ('v2' 或 'v2.1')
        work_dir_base: 工作目錄基礎路徑
        error_log_dir: 錯誤日誌目錄
    """
    config_name = os.path.basename(config_path).replace('_config.py', '')
    
    # 根據模式決定輸出目錄
    if mode == 'v2.1':
        work_dir = os.path.join(work_dir_base + '_p', config_name)
        error_log_dir = error_log_dir + '_p'
    else:
        work_dir = os.path.join(work_dir_base, config_name)
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(error_log_dir, exist_ok=True)

    log_file = os.path.join(work_dir, f"{config_name}_train.log")

    print(f"\n{'='*80}")
    print(f"開始訓練: {config_name}")
    print(f"訓練模式: {mode}")
    print(f"配置檔案: {config_path}")
    print(f"工作目錄: {work_dir}")
    print(f"訓練日誌: {log_file}")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 準備訓練命令
    cmd = [
        'bash',
        'tools/dist_train.sh',
        config_path,
        '8',
        '--work-dir', work_dir
    ]
    
    # v2.1 模式：添加資料集覆蓋選項
    if mode == 'v2.1':
        # 根據配置名稱決定使用 224 或 512 資料集
        cfg_options = build_cfg_options(config_name)
        if cfg_options:
            cmd.extend(['--cfg-options'] + cfg_options)
    
    # 串流輸出，同時保留最後 200 行以便錯誤報告
    last_lines = deque(maxlen=200)
    try:
        with open(log_file, 'w', encoding='utf-8') as lf:
            process = subprocess.Popen(
                cmd,
                cwd='/home/henrywu123/DLFinal/mmsegmentation',
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:  # type: ignore
                print(line, end='')           # 即時顯示在終端
                lf.write(line)               # 寫入模型專屬 log 檔
                last_lines.append(line.rstrip('\n'))

            process.wait()

        if process.returncode == 0:
            print(f"\n✓ {config_name} 訓練完成！")
            return True, None
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        tail_output = '\n'.join(list(last_lines)[-50:]) if last_lines else '(無輸出)'
        error_msg = f"""
訓練失敗: {config_name}
錯誤碼: {e.returncode}
時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

命令: {' '.join(cmd)}

最近輸出 (最後50行):
{('=' * 80)}
{tail_output}

完整訓練日誌: {log_file}
"""
        print(f"\n✗ {config_name} 訓練失敗！錯誤碼: {e.returncode}")
        print(f"詳細錯誤訊息已保存")
        
        # 保存錯誤日誌
        error_file = os.path.join(
            error_log_dir, 
            f"{config_name}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)
        print(f"錯誤日誌: {error_file}")
        
        return False, error_msg

def build_cfg_options(config_name):
    """根據配置名稱為 v2.1 模式生成資料集覆蓋選項"""
    # 判斷是 224 還是 512 配置
    if '224' in config_name.lower():
        # 使用新建的 mmseg 結構 (img_dir/ann_dir)
        base = '/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_MMSEG_FINAL'
        return [
            f'train_dataloader.dataset.data_root={base}',
            'train_dataloader.dataset.data_prefix.img_path=img_dir/train',
            'train_dataloader.dataset.data_prefix.seg_map_path=ann_dir/train',
            f'val_dataloader.dataset.data_root={base}',
            'val_dataloader.dataset.data_prefix.img_path=img_dir/val',
            'val_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val',
            f'test_dataloader.dataset.data_root={base}',
            'test_dataloader.dataset.data_prefix.img_path=img_dir/test',
            'test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/test',
        ]
    elif '512' in config_name.lower():
        # 512 版本改用已建立的 mmseg 結構 (img_dir/ann_dir)
        base512 = '/home/henrywu123/DLFinal/data_V2/preprocess_data/BCSS_512_MMSEG_FINAL'
        return [
            f'train_dataloader.dataset.data_root={base512}',
            'train_dataloader.dataset.data_prefix.img_path=img_dir/train',
            'train_dataloader.dataset.data_prefix.seg_map_path=ann_dir/train',
            f'val_dataloader.dataset.data_root={base512}',
            'val_dataloader.dataset.data_prefix.img_path=img_dir/val',
            'val_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val',
            f'test_dataloader.dataset.data_root={base512}',
            'test_dataloader.dataset.data_prefix.img_path=img_dir/test',
            'test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/test',
        ]
    return None


def main():
    config_dir = '/home/henrywu123/DLFinal/mmsegmentation/configs/configs_comparison'
    error_log_dir = '/home/henrywu123/DLFinal/training_errors'
    work_dir_base = '/home/henrywu123/DLFinal/work_dirs'
    
    print("=" * 80)
    print("模型訓練批次執行工具 - 自適應模式版本")
    print("=" * 80)
    
    # 讓用戶選擇訓練模式
    mode = select_mode()
    
    # 偵測所有配置檔案
    all_config_files = find_config_files(config_dir)
    
    if not all_config_files:
        print(f"在 {config_dir} 中沒有找到任何 config.py 檔案！")
        return
    
    # 讓用戶選擇要訓練的配置
    selected_configs = select_configs(all_config_files)
    
    if not selected_configs:
        print("未選擇任何配置，結束程式。")
        return
    
    # 記錄訓練結果
    results = {}
    errors = {}
    
    print(f"\n{'='*80}")
    print(f"開始批次訓練 - 共 {len(selected_configs)} 個配置")
    print(f"訓練模式: {mode}")
    if mode == 'v2.1':
        print(f"輸出目錄 (防止覆蓋): work_dirs_p, final_result_p, training_errors_p")
    print(f"{'='*80}\n")
    
    # 依序訓練每個配置
    for i, config_path in enumerate(selected_configs, 1):
        config_name = os.path.basename(config_path).replace('_config.py', '')
        print(f"\n[{i}/{len(selected_configs)}] 處理配置: {config_name}")
        
        success, error_msg = train_model(
            config_path, 
            mode=mode,
            work_dir_base=work_dir_base,
            error_log_dir=error_log_dir
        )
        results[config_name] = success
        if not success:
            errors[config_name] = error_msg
    
    # 顯示總結
    print(f"\n{'='*80}")
    print("訓練總結:")
    print(f"{'='*80}")
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    print("\n成功的模型:")
    for config_name, success in results.items():
        if success:
            print(f"  ✓ {config_name}")
    
    if failed > 0:
        print("\n失敗的模型:")
        for config_name, success in results.items():
            if not success:
                print(f"  ✗ {config_name}")
    
    print(f"\n總計: {successful} 成功, {failed} 失敗")
    print(f"訓練模式: {mode}")
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if errors:
        actual_error_dir = error_log_dir + '_p' if mode == 'v2.1' else error_log_dir
        print(f"\n錯誤日誌目錄: {actual_error_dir}")
        print("失敗模型的詳細錯誤訊息已保存至上述目錄")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
