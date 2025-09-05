import os
import time
import random
import argparse
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import math
import csv
import fcntl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class RaceDataset(Dataset):
    def __init__(self, csv_path, img_dir, num_classes=7, transform=None, encoding="gbk"):
        self.df = pd.read_csv(csv_path, encoding=encoding)
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        labels_str = str(self.df.iloc[idx, 1])
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        if labels_str and labels_str == labels_str:
            labels_idx = [int(x) for x in labels_str.split(",") if x != ""]
            label[labels_idx] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise e

def load_backbone_ignore_fc(model: nn.Module, pretrain_pth: str):
    if not (pretrain_pth and os.path.isfile(pretrain_pth)):
        print(f"[Pre-training loading] The path is invalid or does not exist：{pretrain_pth}")
        return

    sd = safe_torch_load(pretrain_pth)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model_sd = model.state_dict()
    new_sd = {}
    skipped = []

    for k, v in sd.items():
        k_clean = k[7:] if k.startswith("module.") else k
        if k_clean.startswith("fc.") or k_clean.startswith("classifier."):
            skipped.append(k)
            continue
        if k_clean in model_sd and model_sd[k_clean].shape == v.shape:
            new_sd[k_clean] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[Pre-training loading] Successfully loaded {len(new_sd)} weights; skip {len(skipped)}; "
          f"Missing keys in the model {len(missing)}; Unexpected key {len(unexpected)}")

def compute_multilabel_specificity_sensitivity(y_true: np.ndarray, y_pred: np.ndarray):
    eps = 1e-12
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
    specificity = tn / (tn + fp + eps)
    sensitivity = tp / (tp + fn + eps)
    return specificity, sensitivity, float(np.mean(specificity)), float(np.mean(sensitivity))

def compute_pos_weight_from_csv(csv_path: str, num_classes: int, encoding="gbk"):
    df = pd.read_csv(csv_path, encoding=encoding)
    pos = np.zeros(num_classes, dtype=np.int64)
    N = len(df)
    for posi in range(N):
        labels_str = str(df.iloc[posi, 1])
        if labels_str and labels_str == labels_str:
            idxs = [int(x) for x in labels_str.split(",") if x != ""]
            for k in idxs:
                pos[k] += 1
    pos = np.clip(pos, 1, None)
    pos_weight = (N - pos) / pos
    return torch.tensor(pos_weight, dtype=torch.float32)

arch_map = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

paths18 = [
    "/home/jliu3422/pth/resnet18-5c106cde.pth",
    "/home/jliu3422/pth/resnet18-f37072fd.pth"
]

paths34 = [
    "/home/jliu3422/pth/resnet34-333f7ec4.pth",
    "/home/jliu3422/pth/resnet34-b627a593.pth"
]

paths50 = [
    "/home/jliu3422/pth/resnet50-0676ba61.pth",
    "/home/jliu3422/pth/resnet50-11ad3fa6.pth",
    "/home/jliu3422/pth/resnet50-19c8e357.pth"
]

paths101 = [
    "/home/jliu3422/pth/resnet101-5d3b4d8f.pth",
    "/home/jliu3422/pth/resnet101-63fe2227.pth",
    "/home/jliu3422/pth/resnet101-cd907fc2.pth"
]

paths152 = [
    "/home/jliu3422/pth/resnet152-394f9c45.pth",
    "/home/jliu3422/pth/resnet152-b121ed2d.pth",
    "/home/jliu3422/pth/resnet152-f82ba261.pth"
]

paths = {
    "resnet18": paths18,
    "resnet34": paths34,
    "resnet50": paths50,
    "resnet101": paths101,
    "resnet152": paths152,
}

tricks_com = [
    ('LR','Re','WU'),
    ('Re','WU'),
    ('LR','Re'),
    ('Re',),
    ('WU',),
    (),
]

num_classes = 7
epochs = 200

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

class RandomErasingWithRegionMean(object):
    def __init__(self, p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), max_attempts=10):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts

    def __call__(self, img: torch.Tensor):
        if torch.rand(1).item() > self.p:
            return img

        c, h, w = img.shape
        area = h * w
        log_ratio_min, log_ratio_max = math.log(self.ratio[0]), math.log(self.ratio[1])

        for _ in range(self.max_attempts):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect = math.exp(torch.empty(1).uniform_(log_ratio_min, log_ratio_max).item())

            erase_h = int(round(math.sqrt(target_area * aspect)))
            erase_w = int(round(math.sqrt(target_area / aspect)))
            if erase_h < h and erase_w < w and erase_h > 0 and erase_w > 0:
                top  = torch.randint(0, h - erase_h + 1, (1,)).item()
                left = torch.randint(0, w - erase_w + 1, (1,)).item()

                patch = img[:, top:top+erase_h, left:left+erase_w]
                mean = patch.reshape(c, -1).mean(dim=1).view(c, 1, 1)

                img[:, top:top+erase_h, left:left+erase_w] = mean
                return img

        return img

def masked_partial_bce_with_logits(logits: torch.Tensor,
                                   labels: torch.Tensor,
                                   alpha_neg: float = 0.05,
                                   eps: float = 0.0):
    if eps > 0.0:
        targets = labels * (1.0 - eps)
    else:
        targets = labels

    loss_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    pos_mask = (labels > 0.5).float()
    neg_mask = 1.0 - pos_mask

    pos_denom = pos_mask.sum().clamp_min(1.0)
    neg_denom = neg_mask.sum().clamp_min(1.0)

    loss_pos = (loss_elem * pos_mask).sum() / pos_denom
    loss_neg = (loss_elem * neg_mask).sum() / neg_denom

    return loss_pos + alpha_neg * loss_neg, loss_pos.detach(), loss_neg.detach()

def mixup_batch(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[idx]
    y = lam * y + (1 - lam) * y[idx]
    return x, y, lam

weight_decay = 2e-4
step_size = 10
gamma = 0.1
num_workers = 10
seed = 42
warmup_epochs = max(1, epochs // 20)
warmup_start_factor = 0.1
warmup_ratio = 0.12
alpha_neg = 0.10      
label_smoothing = 0.00
use_mixup = True
mixup_p = 0.20
mixup_alpha = 0.10
head_lr_mult = 4.0
freeze_epochs = 2 
use_amp = True    

def main():
    train_csv = "Dataset/label_train.csv"
    train_dir = "Dataset/train"
    val_csv = "Dataset/label_val_cropped.csv"
    val_dir = "Dataset/val_cropped"
    csv_encoding = "gbk"

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    if torch.cuda.is_available():
        try:
            print(torch.cuda.get_device_name(0))
        except Exception:
            pass

    lrs = [2e-5, 3e-5, 5e-5]
    batch_sizes = [64, 32, 16, 8, 4]
    for batch_size in batch_sizes:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                # mean=[0.4825167953968048, 0.35792502760887146, 0.304506778717041],
                # std=[0.20769663155078888, 0.18046212196350098, 0.1696053296327591]
                mean=IMNET_MEAN,
                std=IMNET_STD
            ),
        ])

        for lr in lrs:
            for i, pth in enumerate(paths[args.arch]):
                for trick in tricks_com:
                    combined_trick = ''.join(trick)

                    fileHead = f"train3_{args.arch}_pth{i + 1}_{combined_trick}_{lr}"
                    fileName = f"{fileHead}_size{batch_size}"
                    
                    filePath = f"/{fileName}"

                    if 'Re' in trick:
                        train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                # mean=[0.4825167953968048, 0.35792502760887146, 0.304506778717041],
                                # std=[0.20769663155078888, 0.18046212196350098, 0.1696053296327591]
                                mean=IMNET_MEAN,
                                std=IMNET_STD
                            ),
                            RandomErasingWithRegionMean(
                                p=0.10, scale=(0.02, 0.33), ratio=(0.3, 3.3)
                            ),
                        ])
                    else:
                        train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                # mean=[0.4825167953968048, 0.35792502760887146, 0.304506778717041],
                                # std=[0.20769663155078888, 0.18046212196350098, 0.1696053296327591]
                                mean=IMNET_MEAN,
                                std=IMNET_STD
                            ),
                        ])
                    train_dataset = RaceDataset(train_csv, train_dir, num_classes=num_classes, transform=train_transform, encoding=csv_encoding)
                    val_dataset   = RaceDataset(val_csv, val_dir, num_classes=num_classes, transform=val_transform, encoding=csv_encoding)


                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=False,
                          persistent_workers=(num_workers > 0), prefetch_factor=4)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False,
                        persistent_workers=(num_workers > 0), prefetch_factor=4)

                    print("Train sample:", train_dataset[0][0].shape, train_dataset[0][1], train_dataset[0][2])
                    print("Val sample:", val_dataset[0][0].shape, val_dataset[0][1], val_dataset[0][2])

                    logdir = "./logs"
                    weights_dir = "./weights"

                    set_seed(seed)

                    build = arch_map[args.arch]
                    model = build(pretrain_pth=None, num_classes=num_classes)

                    load_backbone_ignore_fc(model, pth)

                    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear) and model.fc.out_features != num_classes:
                        in_f = model.fc.in_features
                        model.fc = nn.Linear(in_f, num_classes)

                    model = model.to(device)

                    pos_weight = compute_pos_weight_from_csv(train_csv, num_classes, encoding=csv_encoding).to(device)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                    backbone_params, head_params = [], []
                    for name, p in model.named_parameters():
                        if not p.requires_grad: 
                            continue
                        if name.startswith("fc.") or "classifier" in name:
                            head_params.append(p)
                        else:
                            backbone_params.append(p)
                    
                    if 'LR' in trick:
                        base_lr = lr * batch_size / 256
                    else:
                        base_lr = lr
                    
                    optimizer = optim.AdamW(
                        [
                            {"params": backbone_params, "lr": base_lr},
                            {"params": head_params,     "lr": base_lr * head_lr_mult},
                        ],
                        weight_decay=weight_decay
                    )
                    
                    schedule_by_step = ('WU' in trick)
                    if schedule_by_step:
                        steps_per_epoch = max(1, len(train_loader))
                        total_steps = epochs * steps_per_epoch
                        warmup_steps = max(1, int(warmup_ratio * total_steps))

                        warmup_sched = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
                        cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
                        scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])
                    else:
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                    
                    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

                    def set_requires_grad(module, flag: bool):
                        for p in module.parameters():
                            p.requires_grad = flag

                    def apply_freeze(model, epoch, state, freeze_epochs: int):
                        if epoch == 0 and not state["applied"]:
                            if hasattr(model, 'layer1'): set_requires_grad(model.layer1, False)
                            if hasattr(model, 'layer2'): set_requires_grad(model.layer2, False)
                            state["applied"] = True

                        if epoch == freeze_epochs:
                            if hasattr(model, 'layer1'): set_requires_grad(model.layer1, True)
                            if hasattr(model, 'layer2'): set_requires_grad(model.layer2, True)

                    run_dir = os.path.join(logdir, filePath.strip("/"))
                    save_dir = os.path.join(weights_dir, filePath.strip("/"))
                    os.makedirs(run_dir, exist_ok=True)
                    os.makedirs(save_dir, exist_ok=True)
                    writer = SummaryWriter(run_dir)
                    best_model_path = os.path.join(save_dir, "best_model.pth")
                    best_pred_csv = os.path.join(save_dir, "prediction.csv")

                    best_val_acc = 0.0
                    best_epoch = -1
                    best_stats = {}

                    freeze_state = {"applied": False}

                    for epoch in range(epochs):
                        start_ts = time.time()
                        apply_freeze(model, epoch, freeze_state, freeze_epochs)
                        model.train()
                        epoch_loss = epoch_pos = epoch_neg = 0.0
                        num_correct = 0
                        num_counted = 0

                        for images, label, _names in tqdm(train_loader, desc=f"[{fileName}] Train {epoch}"):
                            images = images.to(device, non_blocking=True)
                            label = label.to(device, non_blocking=True).float()

                            do_mix = (use_mixup and random.random() < mixup_p)
                            if do_mix:
                                images, label, _ = mixup_batch(images, label, alpha=mixup_alpha)

                            optimizer.zero_grad(set_to_none=True)
                                
                            with autocast(enabled=(use_amp and device.type == "cuda")):
                                logits = model(images)
                                if do_mix:
                                    loss = F.binary_cross_entropy_with_logits(logits, label, reduction='mean')
                                    lp = ln = torch.tensor(0.0, device=logits.device)
                                else:
                                    loss, lp, ln = masked_partial_bce_with_logits(
                                        logits, label, alpha_neg=alpha_neg, eps=label_smoothing
                                    )

                            scaler.scale(loss).backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()

                            if schedule_by_step:
                                scheduler.step()

                            bsz = images.size(0)
                            epoch_loss += loss.item() * bsz
                            epoch_pos  += lp.item() * bsz if isinstance(lp, torch.Tensor) else 0.0
                            epoch_neg  += ln.item() * bsz if isinstance(ln, torch.Tensor) else 0.0

                            if not do_mix:
                                preds = logits.argmax(dim=1)
                                gt_single = label.argmax(dim=1)
                                num_correct += (preds == gt_single).sum().item()
                                num_counted += bsz

                        train_loss = epoch_loss / len(train_loader.dataset)
                        train_acc = (num_correct / num_counted) if num_counted > 0 else 0.0
                        writer.add_scalar("train/loss", train_loss, epoch)
                        writer.add_scalar("train/acc", train_acc, epoch)
                        writer.add_scalar("train/acc_counted_samples", num_counted, epoch)
                        writer.add_scalar("train/loss_pos", epoch_pos / len(train_loader.dataset), epoch)
                        writer.add_scalar("train/loss_neg", epoch_neg / len(train_loader.dataset), epoch)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

                        model.eval()
                        val_loss_accum = 0.0

                        all_true_multi, all_prob_multi = [], []
                        varK_correct, varK_total = 0, 0

                        all_logits = []
                        
                        with torch.no_grad():
                            for images, labels_multi, names in tqdm(val_loader, desc=f"[{fileName}] Val {epoch}"):
                                images = images.to(device, non_blocking=True)
                                labels_multi = labels_multi.to(device).float()
                        
                                logits1 = model(images)
                                logits2 = model(torch.flip(images, dims=[3]))
                                logits  = (logits1 + logits2) / 2.0
                                loss_bce = criterion(logits, labels_multi)
                                val_loss_accum += loss_bce.item() * images.size(0)
                        
                                probs = torch.sigmoid(logits)
                                all_true_multi.append(labels_multi.cpu().numpy())
                                all_prob_multi.append(probs.cpu().numpy())
                                all_logits.append(logits.detach().cpu().numpy())
                        
                                B = images.size(0)
                                for b in range(B):
                                    gt = labels_multi[b]
                                    k = int(gt.sum().item())
                                    if k > 0:
                                        topk_idx = probs[b].topk(k).indices
                                        gt_idx = gt.nonzero(as_tuple=False).squeeze(1)
                                        if set(topk_idx.tolist()) == set(gt_idx.tolist()):
                                            varK_correct += 1
                                        varK_total += 1
                        
                        val_loss = val_loss_accum / max(1, len(val_loader.dataset))
                        y_true = np.concatenate(all_true_multi, axis=0)
                        logits_all = np.concatenate(all_logits, axis=0)

                        def sigmoid_np(x):
                            x = np.clip(x, -30.0, 30.0)
                            return 1.0 / (1.0 + np.exp(-x))
                        
                        T_grid = np.unique(np.concatenate([
                            np.linspace(0.3, 1.2, 10),
                            np.array([0.25, 0.35, 0.45, 0.55])
                        ]))
                        thr_grid = np.linspace(0.01, 0.99, 50)

                        best_T, best_thr = 1.0, np.full(num_classes, 0.5, dtype=np.float32)
                        best_f1_macro = -1.0
                        best_prob = None

                        for T in T_grid:
                            y_prob_T = sigmoid_np(logits_all / T)
                            thr_T = np.zeros(num_classes, dtype=np.float32)

                            for c in range(num_classes):
                                yt = y_true[:, c]; yp = y_prob_T[:, c]
                                f1_best, t_best = -1.0, 0.5
                                for t in thr_grid:
                                    ypc = (yp >= t).astype(np.int32)
                                    _, _, f1c, _ = precision_recall_fscore_support(yt, ypc, average='binary', zero_division=0)
                                    if f1c > f1_best:
                                        f1_best, t_best = f1c, float(t)
                                thr_T[c] = t_best

                            y_pred_T = (y_prob_T >= thr_T[None, :]).astype(np.int32)
                            _, _, f1_macro_T, _ = precision_recall_fscore_support(y_true, y_pred_T, average='macro', zero_division=0)

                            if f1_macro_T > best_f1_macro:
                                best_f1_macro = f1_macro_T
                                best_T = float(T)
                                best_thr = thr_T.copy()
                                best_prob = y_prob_T
                            
                        writer.add_scalar("val/temperature_T", best_T, epoch)

                        y_prob = best_prob
                        y_pred = (y_prob >= best_thr[None, :]).astype(np.int32)

                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, average=None, zero_division=0
                        )
                        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
                            y_true, y_pred, average="macro", zero_division=0
                        )
                        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
                            y_true, y_pred, average="micro", zero_division=0
                        )

                        val_acc_varK = varK_correct / max(1, varK_total)

                        writer.add_scalar("val/loss", float(val_loss), epoch)
                        writer.add_scalar("val/f1_macro", float(f1_macro), epoch)
                        writer.add_scalar("val/f1_micro", float(f1_micro), epoch)
                        writer.add_scalar("val/acc_varK_topk_set", float(val_acc_varK), epoch)
                        for c in range(num_classes):
                            writer.add_scalar(f"val/class_{c}_precision", precision[c], epoch)
                            writer.add_scalar(f"val/class_{c}_recall", recall[c], epoch)
                            writer.add_scalar(f"val/class_{c}_f1", f1[c], epoch)
                        
                        try:
                            ap_per_class = []
                            for c in range(num_classes):
                                if y_true[:, c].sum() > 0:
                                    ap_per_class.append(average_precision_score(y_true[:, c], y_prob[:, c]))
                                else:
                                    ap_per_class.append(np.nan)
                            map_macro = np.nanmean(ap_per_class)
                            writer.add_scalar("val/mAP_macro", float(map_macro), epoch)
                        except Exception:
                            pass

                        try:
                            auc_valid = []
                            for c in range(num_classes):
                                tc = y_true[:, c]
                                if tc.min() != tc.max():
                                    auc_valid.append(roc_auc_score(tc, y_prob[:, c]))
                                else:
                                    auc_valid.append(np.nan)
                            auc_macro = np.nanmean(auc_valid)
                            writer.add_scalar("val/ROC_AUC_macro", float(auc_macro), epoch)
                        except Exception:
                            pass

                        dur = time.time() - start_ts
                        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                            f"val_loss {val_loss:.4f} | f1_macro {f1_macro:.4f} | f1_micro {f1_micro:.4f} | "
                            f"val_acc(varK) {val_acc_varK:.4f} | time {dur:.1f}s")

                        score_for_select = f1_macro
                        if score_for_select > best_val_acc:
                            best_val_acc = score_for_select
                            best_epoch = epoch
                            best_stats = {
                                "precision": precision.tolist(),
                                "recall": recall.tolist(),
                                "f1": f1.tolist(),
                                "precision_macro": float(prec_macro),
                                "recall_macro": float(rec_macro),
                                "f1_macro": float(f1_macro),
                                "precision_micro": float(prec_micro),
                                "recall_micro": float(rec_micro),
                                "f1_micro": float(f1_micro),
                                "val_loss": float(val_loss),
                                "val_acc_varK": float(val_acc_varK),
                                "best_thr": best_thr.tolist(),
                                "best_T": float(best_T),
                            }
                            torch.save(model.state_dict(), best_model_path)
                            df_pred = pd.DataFrame(
                                np.concatenate([y_prob, y_true], axis=1),
                                columns=[f"prob_{probi}" for probi in range(num_classes)] + [f"true_{probi}" for probi in range(num_classes)]
                            )
                            pred_str = []
                            for r in range(y_prob.shape[0]):
                                idx = [str(idxi) for idxi in range(num_classes) if y_prob[r, idxi] >= best_thr[idxi]]
                                pred_str.append(",".join(idx))
                            df_pred.insert(0, "pred_labels", pred_str)
                            df_pred.to_csv(best_pred_csv, index=False, encoding="utf-8")
                            print(f"[BEST] predictions saved to {best_pred_csv}")

                        if not schedule_by_step:
                            scheduler.step()

                    writer.close()

                    rec = np.array(best_stats.get("recall", []), dtype=float)
                    f1_arr = np.array(best_stats.get("f1", []), dtype=float)
                    pre = np.array(best_stats.get("precision", []), dtype=float)
                    thr_arr = np.array(best_stats.get("best_thr", []), dtype=float)

                    def pad(arr, n):
                        if len(arr) >= n:
                            return arr[:n]
                        return np.pad(arr, (0, n - len(arr)), constant_values=0.0)

                    rec = pad(rec, num_classes)
                    f1_arr = pad(f1_arr, num_classes)
                    pre = pad(pre, num_classes)
                    thr_list_for_csv = [f"{t:.2f}" for t in thr_arr]

                    base_cols = [
                        "fileName", "batch_size", "lr",
                        "best_epoch", "f1_macro", "f1_micro", "val_acc_varK"
                    ]
                    per_class_cols = (
                        [f"rec_{numi}" for numi in range(num_classes)] + ["rec_mean"] +
                        [f"f1_{f1i}" for f1i in range(num_classes)] + ["f1_mean"] +
                        [f"precision_{prei}" for prei in range(num_classes)] + ["precision_mean"]
                    )
                    tail_cols = [
                        "val_loss", "prediction_csv", "best_model_path", "timestamp", "best_thr", "best_T"
                    ]
                    columns = base_cols + per_class_cols + tail_cols

                    row = {
                        "fileName": fileName,
                        "batch_size": batch_size,
                        "lr": lr,
                        "best_epoch": best_epoch,
                        "f1_macro": float(best_stats["f1_macro"]),
                        "f1_micro": float(best_stats["f1_micro"]),
                        "val_acc_varK": float(best_stats["val_acc_varK"]),
                        "val_loss": float(best_stats["val_loss"]),
                        "prediction_csv": best_pred_csv,
                        "best_model_path": best_model_path,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "best_thr": ",".join(thr_list_for_csv),
                        "best_T": float(best_stats.get("best_T", 1.0)),
                    }

                    for classi in range(num_classes):
                        row[f"rec_{classi}"] = float(rec[classi])
                        row[f"f1_{classi}"] = float(f1_arr[classi])
                        row[f"precision_{classi}"] = float(pre[classi])

                    row["rec_mean"] = float(rec.mean()) if num_classes > 0 else 0.0
                    row["f1_mean"] = float(f1_arr.mean()) if num_classes > 0 else 0.0
                    row["precision_mean"] = float(pre.mean()) if num_classes > 0 else 0.0

                    results_csv = os.path.join(".", "results.csv")
                    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

                    write_header = (not os.path.exists(results_csv)) or (os.path.getsize(results_csv) == 0)

                    with open(results_csv, "a", newline="", encoding="utf-8") as f:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        except Exception:
                            pass

                        writer = csv.DictWriter(f, fieldnames=columns)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)

                        f.flush()
                        os.fsync(f.fileno())

                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass

                    print(f"[DONE] Results appended to {results_csv}")
                    print(f"[BEST] Prediction CSV path: {best_pred_csv}")

                    print("\n[Per-Class Metrics]")
                    for classesi in range(num_classes):
                        thr_str = f", thr={thr_arr[classesi]:.2f}" if classesi < len(thr_arr) else ""
                        print(f"  class {classesi}: P={pre[classesi]:.4f}, R={rec[classesi]:.4f}, F1={f1_arr[classesi]:.4f}{thr_str}")

                    print(f"\n[Temp Scaling] best_T = {best_T:.2f}")

if __name__ == "__main__":
    main()
