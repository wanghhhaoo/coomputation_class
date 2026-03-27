"""
data/dataset.py — 多数据集支持 (v5.2)

v2:   增加 num_classes 属性（Tiny-ImageNet）
v5.2: 新增 NWPU-RESISC45 数据集支持
      - RESISC45Dataset: root/class_name/xxx.jpg 结构
      - build_resisc45_dataloaders: 80/20 stratified split，固定种子
      - 增强: RandomResizedCrop + RandomHorizontalFlip + RandomVerticalFlip
              + ColorJitter + AutoAugment（遥感图像上下翻转合理）
"""

import os
import json
import torch
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def build_transforms(image_size: int = 64, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_strong_transforms(image_size: int = 64):
    """Stage 3 强数据增强（含 AutoAugment）"""
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.IMAGENET
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class TinyImageNetTrain(Dataset):
    def __init__(self, train_dir: str, transform=None):
        self.transform = transform
        self.samples   = []

        class_dirs = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        assert len(class_dirs) == 200, \
            f"期望200个类别目录，实际发现 {len(class_dirs)} 个"

        self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
        self.classes      = class_dirs
        self.num_classes   = len(class_dirs)      # v2: 新增

        for cls in class_dirs:
            img_dir = os.path.join(train_dir, cls, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((
                        os.path.join(img_dir, fname),
                        self.class_to_idx[cls]
                    ))

        print(f"[Train] {len(self.samples):,} 张图  |  {self.num_classes} 类")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImageNetVal(Dataset):
    def __init__(self, val_dir: str, class_to_idx: dict, transform=None):
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.samples      = []

        ann_file  = os.path.join(val_dir, "val_annotations.txt")
        image_dir = os.path.join(val_dir, "images")

        assert os.path.exists(ann_file),  f"找不到标注文件: {ann_file}"
        assert os.path.isdir(image_dir),  f"找不到图片目录: {image_dir}"

        with open(ann_file, "r") as f:
            for line in f:
                parts    = line.strip().split("\t")
                fname    = parts[0]
                word_id  = parts[1]
                if word_id not in class_to_idx:
                    continue
                label    = class_to_idx[word_id]
                img_path = os.path.join(image_dir, fname)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

        print(f"[Val]   {len(self.samples):,} 张图  |  {len(class_to_idx)} 类")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── NWPU-RESISC45 ─────────────────────────────────────────

def build_resisc45_transforms(image_size: int = 224, is_train: bool = True,
                               use_strong_aug: bool = False):
    """
    遥感图像增强：支持垂直翻转（航拍视角合理）
    训练: RandomResizedCrop + H/V Flip + ColorJitter [+ AutoAugment]
    验证: Resize(256) + CenterCrop(224)
    """
    mean = [0.3680, 0.3810, 0.3436]   # RESISC45 近似统计值
    std  = [0.1454, 0.1356, 0.1320]

    if is_train:
        augs = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ]
        if use_strong_aug:
            augs.append(transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.IMAGENET
            ))
        augs += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        return transforms.Compose(augs)
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class RESISC45Dataset(Dataset):
    """
    NWPU-RESISC45 数据集
    目录结构: root_dir/class_name/class_name_NNN.jpg

    两种构造方式：
      1. 传 root_dir → 自动扫描所有图像（用于第一次扫描）
      2. 传 samples  → 直接使用提供的 (path, label) 列表（用于 train/val split）
    """

    def __init__(self, root_dir: str = None, samples: list = None,
                 transform=None):
        self.transform = transform

        if samples is not None:
            self.samples     = samples
            all_labels       = sorted(set(label for _, label in samples))
            self.num_classes = len(all_labels)
        else:
            assert root_dir is not None, "需要提供 root_dir 或 samples"
            all_classes = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            assert len(all_classes) > 0, f"未找到类别目录: {root_dir}"

            self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
            self.classes      = all_classes
            self.num_classes  = len(all_classes)

            self.samples = []
            for cls in all_classes:
                cls_dir = os.path.join(root_dir, cls)
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            os.path.join(cls_dir, fname),
                            self.class_to_idx[cls]
                        ))

            print(f"[RESISC45] 扫描: {len(self.samples):,} 张图  |  {self.num_classes} 类")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_resisc45_dataloaders(
    data_dir: str,
    image_size: int  = 224,
    batch_size: int  = 64,
    num_workers: int = 16,
    pin_memory: bool = True,
    train_ratio: float = 0.8,
    use_strong_aug: bool = False,
):
    """
    NWPU-RESISC45 数据加载器
    - 按类别做 stratified split（固定随机种子=42，保证可复现）
    - 训练集 80%，验证集 20%
    """
    # 1. 扫描所有样本
    full_ds   = RESISC45Dataset(root_dir=data_dir)
    all_samps = full_ds.samples
    num_cls   = full_ds.num_classes

    # 2. Stratified split（按类别均衡分割）
    by_class = defaultdict(list)
    for i, (_, label) in enumerate(all_samps):
        by_class[label].append(i)

    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx = [], []
    for label in sorted(by_class.keys()):
        indices = by_class[label]
        n_train = int(len(indices) * train_ratio)
        perm    = torch.randperm(len(indices), generator=gen).tolist()
        train_idx.extend([indices[p] for p in perm[:n_train]])
        val_idx.extend([indices[p] for p in perm[n_train:]])

    # 3. 用不同 transform 构建 train/val 数据集
    train_ds = RESISC45Dataset(
        samples=[all_samps[i] for i in train_idx],
        transform=build_resisc45_transforms(image_size, is_train=True,
                                             use_strong_aug=use_strong_aug),
    )
    val_ds = RESISC45Dataset(
        samples=[all_samps[i] for i in val_idx],
        transform=build_resisc45_transforms(image_size, is_train=False),
    )

    print(f"[RESISC45] Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"Classes: {num_cls}  Split: {train_ratio:.0%}/{1-train_ratio:.0%}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, num_cls


# ── Tiny-ImageNet-200（保留向后兼容）──────────────────────

def build_dataloaders(
    train_dir: str,
    val_dir: str,
    image_size: int  = 64,
    batch_size: int  = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    class_map_file: str = None,
    use_strong_aug: bool = False,        # v2: 支持强增强
):
    if use_strong_aug:
        train_transform = build_strong_transforms(image_size)
    else:
        train_transform = build_transforms(image_size, is_train=True)

    train_dataset = TinyImageNetTrain(
        train_dir, transform=train_transform
    )
    val_dataset = TinyImageNetVal(
        val_dir,
        class_to_idx=train_dataset.class_to_idx,
        transform=build_transforms(image_size, is_train=False)
    )

    if class_map_file:
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        os.makedirs(os.path.dirname(class_map_file) or ".", exist_ok=True)
        with open(class_map_file, "w") as f:
            json.dump(idx_to_class, f, indent=2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, train_dataset.num_classes
