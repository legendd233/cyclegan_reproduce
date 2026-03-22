"""
CycleGAN Cityscapes IoU 指标计算

根据 CycleGAN 论文 (Section 5.1.1) 的评估方法，计算：
1. Photo→Labels 方向 (Table 3): 直接比较生成的标签图(fake_B)与真实标签图(real_B)
2. Labels→Photo 方向 (Table 2, FCN score): 用预训练分割模型对生成的照片(fake_A)进行语义分割，
   再与真实标签图(real_B)比较

指标包括：Per-pixel accuracy, Per-class accuracy, Mean class IoU (mIoU)
"""


import os
import numpy as np
from PIL import Image

# ============================================================
# Step 1: 定义 Cityscapes 19 个评估类别及其标准调色板
# ============================================================

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# Cityscapes 标准调色板 (19 个评估类别的 RGB 颜色)
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],  # road
    [244,  35, 232],  # sidewalk
    [ 70,  70,  70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170,  30],  # traffic light
    [220, 220,   0],  # traffic sign
    [107, 142,  35],  # vegetation
    [152, 251, 152],  # terrain
    [ 70, 130, 180],  # sky
    [220,  20,  60],  # person
    [255,   0,   0],  # rider
    [  0,   0, 142],  # car
    [  0,   0,  70],  # truck
    [  0,  60, 100],  # bus
    [  0,  80, 100],  # train
    [  0,   0, 230],  # motorcycle
    [119,  11,  32],  # bicycle
], dtype=np.float32)

NUM_CLASSES = len(CITYSCAPES_CLASSES)  # 19


def rgb_to_class_id(image: np.ndarray) -> np.ndarray:
    """
    将 RGB 标签图映射为类别 ID (0~18)，使用最近邻颜色匹配。
    由于图像经过缩放，边界处存在插值产生的非标准颜色，需要最近邻匹配。
    无法匹配的像素（距离过大）标记为 255 (忽略)。

    Args:
        image: (H, W, 3) uint8 RGB 图像
    Returns:
        label: (H, W) uint8 类别 ID 图
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)  # (N, 3)

    # 计算每个像素到 19 个调色板颜色的欧氏距离
    # pixels: (N, 3), palette: (19, 3) -> dists: (N, 19)
    dists = np.linalg.norm(pixels[:, None, :] - CITYSCAPES_PALETTE[None, :, :], axis=2)

    min_dist = np.min(dists, axis=1)
    class_ids = np.argmin(dists, axis=1).astype(np.uint8)

    # 距离过大的像素标记为忽略 (255)，阈值设为 100
    class_ids[min_dist > 100] = 255

    return class_ids.reshape(h, w)


# ============================================================
# Step 2: 实现 IoU / Accuracy 指标计算（基于混淆矩阵）
# ============================================================

def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    计算混淆矩阵 (num_classes x num_classes)。
    忽略 gt 或 pred 中标记为 255 的像素。
    """
    mask = (gt != 255) & (pred != 255)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (gt[mask].astype(np.int64), pred[mask].astype(np.int64)), 1)
    return conf


def metrics_from_confusion_matrix(conf: np.ndarray):
    """
    从混淆矩阵计算三个指标：
    - Per-pixel accuracy: 正确分类的像素比例
    - Per-class accuracy: 每个类别 accuracy 的平均值
    - Mean class IoU (mIoU): 每个类别 IoU 的平均值
    """
    # Per-pixel accuracy
    per_pixel_acc = np.diag(conf).sum() / (conf.sum() + 1e-10)

    # Per-class accuracy = mean of (TP_i / (TP_i + FN_i))
    class_correct = np.diag(conf)  # TP for each class
    class_total = conf.sum(axis=1)  # TP + FN for each class (ground truth count)
    valid = class_total > 0
    class_acc = np.zeros(conf.shape[0])
    class_acc[valid] = class_correct[valid] / class_total[valid]
    per_class_acc = class_acc[valid].mean()

    # Mean class IoU = mean of (TP_i / (TP_i + FP_i + FN_i))
    intersection = np.diag(conf)
    union = conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf)
    valid_iou = union > 0
    iou = np.zeros(conf.shape[0])
    iou[valid_iou] = intersection[valid_iou] / union[valid_iou]
    mean_iou = iou[valid_iou].mean()

    return per_pixel_acc, per_class_acc, mean_iou, iou


# ============================================================
# Step 3: Photo→Labels 方向评估 (Table 3)
# ============================================================

def evaluate_photo_to_labels(image_dir: str):
    """
    评估 Photo→Labels 方向：直接比较 fake_B (生成的标签图) 与 real_B (真实标签图)。
    对应论文 Table 3。
    """
    print("=" * 60)
    print("Photo → Labels 评估 (对应论文 Table 3)")
    print("比较 fake_B (G(real_A)) vs real_B (ground truth)")
    print("=" * 60)

    # 获取所有测试图像索引
    indices = sorted(
        set(f.split("_A_")[0] for f in os.listdir(image_dir) if f.endswith("_A_real_B.png")),
        key=lambda x: int(x),
    )
    print(f"测试图像数量: {len(indices)}")

    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for i, idx in enumerate(indices):
        gt_path = os.path.join(image_dir, f"{idx}_A_real_B.png")
        pred_path = os.path.join(image_dir, f"{idx}_A_fake_B.png")

        gt_img = np.array(Image.open(gt_path).convert("RGB"))
        pred_img = np.array(Image.open(pred_path).convert("RGB"))

        gt_labels = rgb_to_class_id(gt_img)
        pred_labels = rgb_to_class_id(pred_img)

        conf_matrix += compute_confusion_matrix(pred_labels, gt_labels)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(indices)} 张图像...")

    per_pixel_acc, per_class_acc, mean_iou, per_class_iou = metrics_from_confusion_matrix(conf_matrix)

    print(f"\n{'指标':<20} {'数值':>10}")
    print("-" * 32)
    print(f"{'Per-pixel accuracy':<20} {per_pixel_acc:>10.4f}")
    print(f"{'Per-class accuracy':<20} {per_class_acc:>10.4f}")
    print(f"{'Mean class IoU':<20} {mean_iou:>10.4f}")

    print(f"\n各类别 IoU:")
    for cls_name, cls_iou in zip(CITYSCAPES_CLASSES, per_class_iou):
        print(f"  {cls_name:<20} {cls_iou:.4f}")

    return per_pixel_acc, per_class_acc, mean_iou


# ============================================================
# Step 4: Labels→Photo 方向评估 (FCN score, Table 2)
# ============================================================

def evaluate_labels_to_photo(image_dir: str):
    """
    评估 Labels→Photo 方向 (FCN score)：
    用 Cityscapes 预训练的语义分割模型 (SegFormer-B5) 对 fake_A (从标签生成的照片) 进行分割，
    将分割结果与 real_B (真实标签) 比较。对应论文 Table 2。

    论文原文使用 FCN-8s (Long et al.)，此处使用 SegFormer-B5 (Cityscapes 预训练)，
    同为 Cityscapes 19 类语义分割模型，评估逻辑一致。
    """
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError:
        print("\n" + "=" * 60)
        print("Labels → Photo 评估 (FCN score, 对应论文 Table 2)")
        print("=" * 60)
        print("需要安装 torch 和 transformers:")
        print("  pip install torch transformers")
        print("跳过此评估。")
        return None, None, None

    print("\n" + "=" * 60)
    print("Labels → Photo 评估 (FCN score, 对应论文 Table 2)")
    print("比较 SegFormer(fake_A) vs real_B (ground truth)")
    print("使用 SegFormer-B5 (Cityscapes 19类预训练)")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 Cityscapes 预训练的 SegFormer-B5，输出 19 个 Cityscapes 评估类别
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    model = model.to(device)
    model.eval()

    indices = sorted(
        set(f.split("_A_")[0] for f in os.listdir(image_dir) if f.endswith("_A_real_B.png")),
        key=lambda x: int(x),
    )
    print(f"测试图像数量: {len(indices)}")

    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            gt_path = os.path.join(image_dir, f"{idx}_A_real_B.png")
            fake_photo_path = os.path.join(image_dir, f"{idx}_A_fake_A.png")

            # 真实标签图 -> class IDs
            gt_img = np.array(Image.open(gt_path).convert("RGB"))
            gt_labels = rgb_to_class_id(gt_img)

            # 生成的照片 -> SegFormer 分割 -> Cityscapes 19类 class IDs
            fake_photo = Image.open(fake_photo_path).convert("RGB")
            inputs = processor(images=fake_photo, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits  # (1, 19, H/4, W/4)

            # 上采样到原始尺寸
            pred = torch.nn.functional.interpolate(
                logits, size=gt_labels.shape, mode="bilinear", align_corners=False
            )
            pred_labels = pred.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

            conf_matrix += compute_confusion_matrix(pred_labels, gt_labels)

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(indices)} 张图像...")

    per_pixel_acc, per_class_acc, mean_iou, per_class_iou = metrics_from_confusion_matrix(conf_matrix)

    print(f"\n{'指标':<20} {'数值':>10}")
    print("-" * 32)
    print(f"{'Per-pixel accuracy':<20} {per_pixel_acc:>10.4f}")
    print(f"{'Per-class accuracy':<20} {per_class_acc:>10.4f}")
    print(f"{'Mean class IoU':<20} {mean_iou:>10.4f}")

    print(f"\n各类别 IoU:")
    for cls_name, cls_iou in zip(CITYSCAPES_CLASSES, per_class_iou):
        print(f"  {cls_name:<20} {cls_iou:.4f}")

    return per_pixel_acc, per_class_acc, mean_iou


# ============================================================
# Step 5: 主函数 - 汇总输出结果
# ============================================================

def main():
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cityscapes_cyclegan", "test_latest", "images")

    if not os.path.isdir(image_dir):
        print(f"错误: 未找到测试结果目录: {image_dir}")
        print("请先运行 CycleGAN 测试生成结果图像。")
        return

    # 评估 Photo→Labels (Table 3)
    p2l_ppa, p2l_pca, p2l_iou = evaluate_photo_to_labels(image_dir)

    # 评估 Labels→Photo (Table 2, FCN score)
    l2p_ppa, l2p_pca, l2p_iou = evaluate_labels_to_photo(image_dir)

    # 汇总
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"\n论文 Table 3 - Photo→Labels (CycleGAN):")
    print(f"  Per-pixel acc: {p2l_ppa:.2f}")
    print(f"  Per-class acc: {p2l_pca:.2f}")
    print(f"  Class IoU:     {p2l_iou:.2f}")
    print(f"\n论文 Table 3 - Photo→Labels (论文报告值):")
    print(f"  Per-pixel acc: 0.58")
    print(f"  Per-class acc: 0.22")
    print(f"  Class IoU:     0.16")

    if l2p_ppa is not None:
        print(f"\n论文 Table 2 - Labels→Photo FCN score (CycleGAN):")
        print(f"  Per-pixel acc: {l2p_ppa:.2f}")
        print(f"  Per-class acc: {l2p_pca:.2f}")
        print(f"  Class IoU:     {l2p_iou:.2f}")
        print(f"\n论文 Table 2 - Labels→Photo (论文报告值):")
        print(f"  Per-pixel acc: 0.52")
        print(f"  Per-class acc: 0.17")
        print(f"  Class IoU:     0.11")


if __name__ == "__main__":
    main()