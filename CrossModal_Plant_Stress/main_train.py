import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import *
from voc_registry import build_knowledge_bank_tensors, get_stress_to_voc_mask, VOC_KNOWLEDGE_BASE
from data_loader_sensor import load_records_from_zip, GasResponseDataset
from model_fusion import CrossModalNetwork


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_fold(fold_id, train_recs, val_recs, out_dir, device, g_batch, fp_batch, voc_mask_true):
    ds_tr = GasResponseDataset(train_recs)
    ds_va = GasResponseDataset(val_recs, mean=ds_tr.mean, std=ds_tr.std)

    # 类别平衡采样器，防止 Batch 内标签极度不平衡导致模型抽风
    period_labels = torch.tensor([r["period_idx"] for r in train_recs], dtype=torch.long)
    class_counts = torch.bincount(period_labels, minlength=5)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = class_weights[period_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model = CrossModalNetwork(d_sensor=D_MODEL, d_mol=MOL_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    # 余弦退火学习率调度器，后期平稳收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_score = 0
    fold_dir = Path(out_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    g_batch = {k: v.to(device) for k, v in g_batch.items()}
    fp_batch = fp_batch.to(device)
    voc_mask_true = voc_mask_true.to(device)

    history = {"epoch": [], "acc_s": [], "acc_p": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y_s, y_p in dl_tr:
            x, y_s, y_p = x.to(device), y_s.to(device), y_p.to(device)
            opt.zero_grad()
            out = model(x, g_batch, fp_batch)
            loss = model.loss(out, y_s, y_p, voc_mask_true)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        scheduler.step()

        model.eval()
        ys_all, ps_all, yp_all, pp_all, attn_all = [], [], [], [], []
        with torch.no_grad():
            for x, y_s, y_p in dl_va:
                x = x.to(device)
                out = model(x, g_batch, fp_batch)

                ys_all.append(y_s.cpu())
                ps_all.append(out["logits_stress"].argmax(dim=1).cpu())

                yp_all.append(y_p.cpu())
                # 【关键修复点】：这里使用 logits_period_cls
                pp_all.append(out["logits_period_cls"].argmax(dim=1).cpu())

                attn_all.append(out["attn_voc"].cpu())

        ys_all = torch.cat(ys_all)
        ps_all = torch.cat(ps_all)
        yp_all = torch.cat(yp_all)
        pp_all = torch.cat(pp_all)
        attn_all = torch.cat(attn_all)

        acc_s = (ys_all == ps_all).float().mean().item()
        acc_p = (yp_all == pp_all).float().mean().item()
        acc_joint = ((ys_all == ps_all) & (yp_all == pp_all)).float().mean().item()

        history["epoch"].append(ep)
        history["acc_s"].append(acc_s)
        history["acc_p"].append(acc_p)

        score = 0.5 * acc_s + 0.5 * acc_p

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), fold_dir / "best_model.pt")

            attn_df = pd.DataFrame(attn_all.numpy(), columns=[v["en"] for v in VOC_KNOWLEDGE_BASE])
            attn_df["True_Stress"] = ys_all.numpy()
            attn_df["True_Period"] = yp_all.numpy()
            attn_df.to_csv(fold_dir / "voc_attention_analysis.csv", index=False)

        current_lr = opt.param_groups[0]['lr']
        print(f"[Fold {fold_id}] Epoch {ep:03d} | LR: {current_lr:.2e} | "
              f"Stress: {acc_s:.4f} | Period: {acc_p:.4f} | Joint: {acc_joint:.4f} | "
              f"Best: {best_score:.4f}")

    return history


def plot_cv5_curves(all_histories, out_dir):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for fold in range(5):
        plt.plot(all_histories[fold]["epoch"], all_histories[fold]["acc_s"],
                 label=f"Fold {fold}", linewidth=2, alpha=0.9)
    plt.title("Validation Stress Accuracy (5-Fold CV)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    for fold in range(5):
        plt.plot(all_histories[fold]["epoch"], all_histories[fold]["acc_p"],
                 label=f"Fold {fold}", linewidth=2, alpha=0.9)
    plt.title("Validation Period Accuracy (5-Fold CV)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    save_path = out_dir / "cv5_accuracy_curves_fundamental.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 基于底层优化后的5折精度曲线图已保存至: {save_path}")


def main():
    set_seed(SEED)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building Chemical Knowledge Bank...")
    g_batch, fp_batch = build_knowledge_bank_tensors()
    voc_mask_true = get_stress_to_voc_mask()

    print("Loading sensor data from zip...")
    records = load_records_from_zip(ZIP_PATH, START_ROW, TARGET_LEN)
    print(f"Loaded records: {len(records)}")

    if len(records) == 0:
        print("未提取到任何数据，请检查 ZIP_PATH 是否正确。")
        return

    all_histories = []

    for fold in range(5):
        tr = [r for r in records if r["cv_fold"] != fold]
        va = [r for r in records if r["cv_fold"] == fold]
        print(f"\n===== Starting Fold {fold} (Train: {len(tr)}, Val: {len(va)}) =====")

        fold_history = train_one_fold(fold, tr, va, out_dir, device, g_batch, fp_batch, voc_mask_true)
        all_histories.append(fold_history)

    plot_cv5_curves(all_histories, out_dir)


if __name__ == "__main__":
    main()