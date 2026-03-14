"""
逐项核验报告中的所有数据是否来自实际训练文件，标记任何不一致之处。
复用 plot_losses.py 的清洗逻辑以确保一致。
"""
import os
import sys
import numpy as np

# 复用 plot_losses 的解析和清洗逻辑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_losses import parse_loss_log, epoch_means, LOSS_KEYS, PROJECT_DIR

issues = []

print("=" * 70)
print("核验报告：逐项对照源文件（使用清洗后数据）")
print("=" * 70)


# ==================== 工具函数 ====================

def parse_opt(path):
    opts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('---'):
                parts = line.split(':')
                key = parts[0].strip()
                val = ':'.join(parts[1:]).split('\t')[0].strip()
                opts[key] = val
    return opts


def epoch_range(raw, ep):
    """从 raw dict 中取指定 epoch 的所有值"""
    return {k: raw[k].get(ep, []) for k in LOSS_KEYS}


# ==================== 加载数据 ====================

maps_raw = parse_loss_log(
    os.path.join(PROJECT_DIR, 'checkpoints', 'maps_cyclegan', 'loss_log.txt'), 'Maps')
monet_raw = parse_loss_log(
    os.path.join(PROJECT_DIR, 'checkpoints', 'monet_cyclegan', 'loss_log.txt'), 'Monet')

maps_train_opt = parse_opt(
    os.path.join(PROJECT_DIR, 'checkpoints', 'maps_cyclegan', 'train_opt.txt'))
monet_train_opt = parse_opt(
    os.path.join(PROJECT_DIR, 'checkpoints', 'monet_cyclegan', 'train_opt.txt'))


# ==================== 第2节：复现实验配置 ====================
print("\n## 第2节：复现实验配置")
maps_checks = {
    'dataroot': ('./datasets/maps', maps_train_opt.get('dataroot', '')),
    'model': ('cycle_gan', maps_train_opt.get('model', '')),
    'netG': ('resnet_9blocks', maps_train_opt.get('netG', '')),
    'netD': ('basic', maps_train_opt.get('netD', '')),
    'ngf': ('64', maps_train_opt.get('ngf', '')),
    'ndf': ('64', maps_train_opt.get('ndf', '')),
    'n_layers_D': ('3', maps_train_opt.get('n_layers_D', '')),
    'norm': ('instance', maps_train_opt.get('norm', '')),
    'gan_mode': ('lsgan', maps_train_opt.get('gan_mode', '')),
    'lr': ('0.0002', maps_train_opt.get('lr', '')),
    'beta1': ('0.5', maps_train_opt.get('beta1', '')),
    'batch_size': ('1', maps_train_opt.get('batch_size', '')),
    'lambda_A': ('10.0', maps_train_opt.get('lambda_A', '')),
    'lambda_B': ('10.0', maps_train_opt.get('lambda_B', '')),
    'lambda_identity': ('0.5', maps_train_opt.get('lambda_identity', '')),
    'pool_size': ('50', maps_train_opt.get('pool_size', '')),
    'n_epochs': ('100', maps_train_opt.get('n_epochs', '')),
    'n_epochs_decay': ('100', maps_train_opt.get('n_epochs_decay', '')),
    'load_size': ('286', maps_train_opt.get('load_size', '')),
    'crop_size': ('256', maps_train_opt.get('crop_size', '')),
    'preprocess': ('resize_and_crop', maps_train_opt.get('preprocess', '')),
    'no_dropout': ('True', maps_train_opt.get('no_dropout', '')),
}
for param, (report_val, actual_val) in maps_checks.items():
    status = "✅" if report_val == actual_val else "❌"
    if report_val != actual_val:
        issues.append(f"Maps {param}: 报告={report_val}, 实际={actual_val}")
    print(f"  {status} Maps {param}: 报告={report_val}, 实际={actual_val}")


# ==================== 第3节：损失范围核验 ====================
print("\n## 第3节：损失范围核验")

def check_range(raw, epoch, claim_ranges, label):
    ep_data = epoch_range(raw, epoch)
    for key, (cmin, cmax) in claim_ranges.items():
        vals = ep_data[key]
        if not vals:
            issues.append(f"{label} {key}: epoch {epoch} 无数据")
            print(f"  ❌ {label} {key}: epoch {epoch} 无数据")
            continue
        actual_min, actual_max = min(vals), max(vals)
        ok = actual_min >= cmin - 0.05 and actual_max <= cmax + 0.05
        status = "✅" if ok else "❌"
        if not ok:
            issues.append(f"{label} {key}: 报告={cmin}~{cmax}, 实际=[{actual_min:.3f}, {actual_max:.3f}]")
        print(f"  {status} {label} {key}: 报告={cmin}~{cmax}, 实际=[{actual_min:.3f}, {actual_max:.3f}]")

print("\n  --- Maps Epoch 1 ---")
check_range(maps_raw, 1, {
    'cycle_A': (1.2, 3.1), 'cycle_B': (0.7, 2.0),
    'D_A': (0.12, 0.38), 'D_B': (0.17, 0.42),
    'G_A': (0.15, 0.55), 'G_B': (0.19, 0.45),
    'idt_A': (0.28, 0.96), 'idt_B': (0.58, 1.56),
}, "Maps Ep1")

print("\n  --- Maps Epoch 200 ---")
check_range(maps_raw, 200, {
    'cycle_A': (0.4, 0.8), 'cycle_B': (0.1, 0.4),
    'D_A': (0.03, 0.20), 'D_B': (0.14, 0.24),
    'G_A': (0.3, 1.1), 'G_B': (0.26, 0.38),
    'idt_A': (0.04, 0.11), 'idt_B': (0.08, 0.16),
}, "Maps Ep200")

print("\n  --- Monet Epoch 1 (清洗后) ---")
check_range(monet_raw, 1, {
    'cycle_A': (0.4, 4.3), 'cycle_B': (0.4, 5.0),
    'D_A': (0.03, 0.58), 'D_B': (0.03, 0.57),
    'G_A': (0.06, 1.12), 'G_B': (0.05, 1.25),
    'idt_A': (0.11, 2.50), 'idt_B': (0.16, 2.19),
}, "Monet Ep1")

print("\n  --- Monet Epoch 200 (清洗后) ---")
check_range(monet_raw, 200, {
    'cycle_A': (0.3, 0.9), 'cycle_B': (0.3, 0.9),
    'D_A': (0.04, 0.34), 'D_B': (0.03, 0.39),
    'G_A': (0.3, 0.8), 'G_B': (0.3, 1.2),
    'idt_A': (0.08, 0.34), 'idt_B': (0.09, 0.32),
}, "Monet Ep200")


# ==================== 图8 柱状图数值 ====================
print("\n## 图8 柱状图数值")
fig8_maps = {'D_A': 0.085, 'G_A': 0.726, 'cycle_A': 0.584, 'idt_A': 0.072,
             'D_B': 0.197, 'G_B': 0.313, 'cycle_B': 0.206, 'idt_B': 0.118}
fig8_monet = {'D_A': 0.143, 'G_A': 0.495, 'cycle_A': 0.511, 'idt_A': 0.164,
              'D_B': 0.085, 'G_B': 0.733, 'cycle_B': 0.504, 'idt_B': 0.158}
for key in LOSS_KEYS:
    maps_mean = np.mean(maps_raw[key][200])
    monet_mean = np.mean(monet_raw[key][200])
    ms = "✅" if abs(maps_mean - fig8_maps[key]) < 0.01 else "❌"
    mos = "✅" if abs(monet_mean - fig8_monet[key]) < 0.01 else "❌"
    print(f"  {ms} Maps {key}: 图={fig8_maps[key]:.3f}, 算={maps_mean:.3f}")
    print(f"  {mos} Monet {key}: 图={fig8_monet[key]:.3f}, 算={monet_mean:.3f}")
    if abs(maps_mean - fig8_maps[key]) >= 0.01:
        issues.append(f"图8 Maps {key}: 标注={fig8_maps[key]}, 计算={maps_mean:.3f}")
    if abs(monet_mean - fig8_monet[key]) >= 0.01:
        issues.append(f"图8 Monet {key}: 标注={fig8_monet[key]}, 计算={monet_mean:.3f}")


# ==================== 文件数量 ====================
print("\n## 文件数量")
maps_pth = len([f for f in os.listdir(os.path.join(PROJECT_DIR, 'checkpoints/maps_cyclegan')) if f.endswith('.pth')])
monet_pth = len([f for f in os.listdir(os.path.join(PROJECT_DIR, 'checkpoints/monet_cyclegan')) if f.endswith('.pth')])
maps_img = len(os.listdir(os.path.join(PROJECT_DIR, 'checkpoints/maps_cyclegan/web/images')))
monet_img = len(os.listdir(os.path.join(PROJECT_DIR, 'checkpoints/monet_cyclegan/web/images')))
for name, actual, expected in [
    ("Maps .pth", maps_pth, 164), ("Monet .pth", monet_pth, 164),
    ("Maps images", maps_img, 1600), ("Monet images", monet_img, 1600),
]:
    s = "✅" if actual == expected else "❌"
    print(f"  {s} {name}: 报告={expected}, 实际={actual}")
    if actual != expected:
        issues.append(f"{name}: 报告={expected}, 实际={actual}")


# ==================== 总结 ====================
print("\n" + "=" * 70)
if issues:
    print(f"⚠️  发现 {len(issues)} 处不一致：")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("✅ 所有数据项均与源文件一致，未发现编造或错误。")
print("=" * 70)
