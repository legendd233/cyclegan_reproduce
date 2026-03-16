"""
根据 loss_log.txt 生成复现报告所需的数据图。
输出保存在 figures/ 目录下。

日志清洗策略：
  loss_log.txt 可能包含多段 session（由 "Training Loss" header 分隔）。
  例如 Monet 日志包含三段：首次训练 epoch 1-108、一次误启动 epoch 1（仅21条）、
  正式续训 epoch 108-200。脚本按 header 拆分 session，根据每段覆盖的 epoch 范围
  拼接出一条不重叠的时间线，短 session 覆盖的 epoch 如果已被更长 session 覆盖则丢弃。
"""

import re
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------- 解析 loss_log.txt ----------

HEADER_RE = re.compile(r'=+ Training Loss \((.+?)\) =+')
LINE_RE = re.compile(
    r'\(epoch:\s*(\d+),\s*iters:\s*(\d+).*?\)'
    r'\s*,\s*D_A:\s*([\d.]+),\s*G_A:\s*([\d.]+),\s*cycle_A:\s*([\d.]+),\s*idt_A:\s*([\d.]+)'
    r',\s*D_B:\s*([\d.]+),\s*G_B:\s*([\d.]+),\s*cycle_B:\s*([\d.]+),\s*idt_B:\s*([\d.]+)'
)

LOSS_KEYS = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']


def _parse_sessions(path):
    """将日志文件按 header 拆分为多段 session，每段返回 (timestamp, [records])。"""
    sessions = []
    current_ts = None
    current_records = []

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            hm = HEADER_RE.search(line)
            if hm:
                if current_records:
                    sessions.append((current_ts, current_records))
                current_ts = hm.group(1)
                current_records = []
                continue

            m = LINE_RE.search(line)
            if m:
                current_records.append({
                    'epoch': int(m.group(1)),
                    'iters': int(m.group(2)),
                    **{k: float(m.group(i)) for i, k in enumerate(LOSS_KEYS, 3)},
                    '_lineno': lineno,
                })

    if current_records:
        sessions.append((current_ts, current_records))

    return sessions


def _merge_sessions(sessions, log_name=''):
    """
    从多段 session 中拼接出一条不重叠的 epoch 时间线。
    策略：按 session 顺序扫描，对于每段 session 覆盖的 epoch 范围，
    如果与已选中的 epoch 不重叠则直接纳入；如果重叠，选择记录数更多的那段。
    """
    if len(sessions) <= 1:
        records = sessions[0][1] if sessions else []
        if sessions:
            print(f"  [{log_name}] 1 段 session: {sessions[0][0]}, "
                  f"{len(records)} 条记录, epoch {records[0]['epoch']}-{records[-1]['epoch']}")
        return records

    # 按 session 统计 epoch 覆盖
    print(f"  [{log_name}] 检测到 {len(sessions)} 段 session:")
    session_info = []
    for i, (ts, recs) in enumerate(sessions):
        if not recs:
            print(f"    Session {i+1} ({ts}): 空 session, 丢弃")
            continue
        epochs = sorted(set(r['epoch'] for r in recs))
        ep_min, ep_max = epochs[0], epochs[-1]
        print(f"    Session {i+1} ({ts}): {len(recs)} 条, "
              f"epoch {ep_min}-{ep_max} ({len(epochs)} 个 epoch)")
        session_info.append((i, ts, recs, epochs, ep_min, ep_max))

    # 按 epoch 范围大小降序排列，优先保留跨度大的 session
    session_info.sort(key=lambda s: len(s[3]), reverse=True)
    claimed_epochs = set()
    selected_records = []

    for i, ts, recs, epochs, ep_min, ep_max in session_info:
        new_epochs = set(epochs) - claimed_epochs
        if not new_epochs:
            print(f"    -> Session {i+1} 的 epoch 全部已被覆盖, 丢弃")
            continue
        overlap = set(epochs) & claimed_epochs
        if overlap:
            # 只保留未被占用的 epoch
            recs = [r for r in recs if r['epoch'] in new_epochs]
            print(f"    -> Session {i+1}: 保留 {len(new_epochs)} 个新 epoch, "
                  f"丢弃 {len(overlap)} 个重叠 epoch")
        else:
            print(f"    -> Session {i+1}: 全部保留")
        claimed_epochs |= new_epochs
        selected_records.extend(recs)

    # 按 epoch + iters 排序确保时间线正确
    selected_records.sort(key=lambda r: (r['epoch'], r['iters']))

    # 完整性校验
    final_epochs = sorted(set(r['epoch'] for r in selected_records))
    expected = list(range(final_epochs[0], final_epochs[-1] + 1))
    missing = set(expected) - set(final_epochs)
    if missing:
        print(f"  ⚠️  [{log_name}] 缺失 epoch: {sorted(missing)}")
    else:
        print(f"  [{log_name}] 最终时间线: epoch {final_epochs[0]}-{final_epochs[-1]}, "
              f"共 {len(final_epochs)} 个 epoch, {len(selected_records)} 条记录, 无缺失")

    return selected_records


def parse_loss_log(path, log_name=''):
    """
    解析 loss_log.txt，处理多 session 拼接，
    返回 {key: {epoch: [values]}} 的字典。
    """
    if not os.path.isfile(path):
        print(f"  ❌ [{log_name}] 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    sessions = _parse_sessions(path)
    if not sessions:
        print(f"  ❌ [{log_name}] 未解析到任何记录: {path}", file=sys.stderr)
        sys.exit(1)

    records = _merge_sessions(sessions, log_name)

    epoch_losses = {k: defaultdict(list) for k in LOSS_KEYS}
    for r in records:
        for k in LOSS_KEYS:
            epoch_losses[k][r['epoch']].append(r[k])

    return epoch_losses


def epoch_means(epoch_losses):
    """将每 epoch 的多次记录取平均，返回 {key: (epochs, means)}"""
    result = {}
    for k, ep_dict in epoch_losses.items():
        epochs = sorted(ep_dict.keys())
        means = [np.mean(ep_dict[e]) for e in epochs]
        result[k] = (np.array(epochs), np.array(means))
    return result


# ---------- 加载两个实验的数据 ----------

print("=" * 60)
print("日志清洗与加载")
print("=" * 60)

maps_log = os.path.join(PROJECT_DIR, 'checkpoints', 'maps_cyclegan', 'loss_log.txt')
monet_log = os.path.join(PROJECT_DIR, 'checkpoints', 'monet_cyclegan', 'loss_log.txt')
cityscapes_log = os.path.join(PROJECT_DIR, 'checkpoints', 'checkpoints',
                              'cityscapes_cyclegan', 'loss_log.txt')

maps_raw = parse_loss_log(maps_log, 'Maps')
monet_raw = parse_loss_log(monet_log, 'Monet')
cityscapes_raw = parse_loss_log(cityscapes_log, 'Cityscapes')
maps = epoch_means(maps_raw)
monet = epoch_means(monet_raw)
cityscapes = epoch_means(cityscapes_raw)

print("=" * 60)
print()


# =====================================================================
# 图1: Maps — Cycle Consistency Loss 与 Identity Loss 随 epoch 变化
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(*maps['cycle_A'], label='cycle_A (map→aerial→map)', linewidth=1.5)
ax.plot(*maps['cycle_B'], label='cycle_B (aerial→map→aerial)', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Maps — Cycle Consistency Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='LR decay start')

ax = axes[1]
ax.plot(*maps['idt_A'], label='idt_A', linewidth=1.5)
ax.plot(*maps['idt_B'], label='idt_B', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Maps — Identity Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_maps_cycle_idt.png'), dpi=150)
plt.close()


# =====================================================================
# 图2: Maps — GAN Loss (生成器 + 判别器) 随 epoch 变化
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(*maps['G_A'], label='G_A', linewidth=1.5, alpha=0.85)
ax.plot(*maps['G_B'], label='G_B', linewidth=1.5, alpha=0.85)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Maps — Generator Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(*maps['D_A'], label='D_A', linewidth=1.5, alpha=0.85)
ax.plot(*maps['D_B'], label='D_B', linewidth=1.5, alpha=0.85)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Maps — Discriminator Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_maps_gan.png'), dpi=150)
plt.close()


# =====================================================================
# 图3: Monet — Cycle Consistency Loss 与 Identity Loss 随 epoch 变化
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(*monet['cycle_A'], label='cycle_A (photo→monet→photo)', linewidth=1.5)
ax.plot(*monet['cycle_B'], label='cycle_B (monet→photo→monet)', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Monet2Photo — Cycle Consistency Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(*monet['idt_A'], label='idt_A', linewidth=1.5)
ax.plot(*monet['idt_B'], label='idt_B', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Monet2Photo — Identity Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_monet_cycle_idt.png'), dpi=150)
plt.close()


# =====================================================================
# 图4: Monet — GAN Loss (生成器 + 判别器) 随 epoch 变化
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(*monet['G_A'], label='G_A', linewidth=1.5, alpha=0.85)
ax.plot(*monet['G_B'], label='G_B', linewidth=1.5, alpha=0.85)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Monet2Photo — Generator Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(*monet['D_A'], label='D_A', linewidth=1.5, alpha=0.85)
ax.plot(*monet['D_B'], label='D_B', linewidth=1.5, alpha=0.85)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Monet2Photo — Discriminator Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_monet_gan.png'), dpi=150)
plt.close()


# =====================================================================
# 图5: 三个数据集 Cycle Loss 对比
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(*maps['cycle_A'], label='Maps cycle_A', linewidth=1.5)
ax.plot(*monet['cycle_A'], label='Monet cycle_A', linewidth=1.5)
ax.plot(*cityscapes['cycle_A'], label='Cityscapes cycle_A', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('cycle_A 对比 (A→B→A 重建)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(*maps['cycle_B'], label='Maps cycle_B', linewidth=1.5)
ax.plot(*monet['cycle_B'], label='Monet cycle_B', linewidth=1.5)
ax.plot(*cityscapes['cycle_B'], label='Cityscapes cycle_B', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('cycle_B 对比 (B→A→B 重建)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_cycle_compare.png'), dpi=150)
plt.close()


# =====================================================================
# 图6: 三个数据集所有损失的最终值对比 (柱状图)
# =====================================================================
labels = LOSS_KEYS
maps_final = [np.mean(maps_raw[k][200]) if 200 in maps_raw[k] else np.mean(maps_raw[k][max(maps_raw[k].keys())]) for k in labels]
monet_final = [np.mean(monet_raw[k][200]) if 200 in monet_raw[k] else np.mean(monet_raw[k][max(monet_raw[k].keys())]) for k in labels]
cityscapes_final = [np.mean(cityscapes_raw[k][200]) if 200 in cityscapes_raw[k] else np.mean(cityscapes_raw[k][max(cityscapes_raw[k].keys())]) for k in labels]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 5.5))
bars1 = ax.bar(x - width, maps_final, width, label='Maps (epoch 200)', color='#4C72B0')
bars2 = ax.bar(x, monet_final, width, label='Monet2Photo (epoch 200)', color='#DD8452')
bars3 = ax.bar(x + width, cityscapes_final, width, label='Cityscapes (epoch 200)', color='#55A868')

ax.set_ylabel('Loss (epoch mean)')
ax.set_title('Epoch 200 各损失项最终均值对比')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig6_final_loss_compare.png'), dpi=150)
plt.close()


# =====================================================================
# 图7: Maps 全部 8 项损失汇总 (2×4 子图)
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Maps — 所有损失项训练曲线', fontsize=14, y=1.01)

for idx, key in enumerate(LOSS_KEYS):
    ax = axes[idx // 4][idx % 4]
    epochs, means = maps[key]
    ax.plot(epochs, means, linewidth=1.2, color='#4C72B0')
    ax.set_title(key, fontsize=12)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig7_maps_all_losses.png'), dpi=150)
plt.close()


# =====================================================================
# 图8: Monet 全部 8 项损失汇总 (2×4 子图)
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Monet2Photo — 所有损失项训练曲线', fontsize=14, y=1.01)

for idx, key in enumerate(LOSS_KEYS):
    ax = axes[idx // 4][idx % 4]
    epochs, means = monet[key]
    ax.plot(epochs, means, linewidth=1.2, color='#DD8452')
    ax.set_title(key, fontsize=12)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig8_monet_all_losses.png'), dpi=150)
plt.close()


# =====================================================================
# 图9: Cityscapes — Cycle Consistency Loss 与 Identity Loss 随 epoch 变化
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(*cityscapes['cycle_A'], label='cycle_A (photo→seg→photo)', linewidth=1.5)
ax.plot(*cityscapes['cycle_B'], label='cycle_B (seg→photo→seg)', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Cityscapes — Cycle Consistency Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(*cityscapes['idt_A'], label='idt_A', linewidth=1.5)
ax.plot(*cityscapes['idt_B'], label='idt_B', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Cityscapes — Identity Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig9_cityscapes_cycle_idt.png'), dpi=150)
plt.close()


# =====================================================================
# 图10: Cityscapes — GAN Loss (生成器 + 判别器) 随 epoch 变化
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(*cityscapes['G_A'], label='G_A', linewidth=1.5, alpha=0.85)
ax.plot(*cityscapes['G_B'], label='G_B', linewidth=1.5, alpha=0.85)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Cityscapes — Generator Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(*cityscapes['D_A'], label='D_A', linewidth=1.5, alpha=0.85)
ax.plot(*cityscapes['D_B'], label='D_B', linewidth=1.5, alpha=0.85)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Cityscapes — Discriminator Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig10_cityscapes_gan.png'), dpi=150)
plt.close()


# =====================================================================
# 图11: Cityscapes 全部 8 项损失汇总 (2×4 子图)
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Cityscapes — 所有损失项训练曲线', fontsize=14, y=1.01)

for idx, key in enumerate(LOSS_KEYS):
    ax = axes[idx // 4][idx % 4]
    epochs, means = cityscapes[key]
    ax.plot(epochs, means, linewidth=1.2, color='#55A868')
    ax.set_title(key, fontsize=12)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig11_cityscapes_all_losses.png'), dpi=150)
plt.close()


print("所有图表已保存至 figures/ 目录：")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith('.png'):
        print(f"  - {f}")
