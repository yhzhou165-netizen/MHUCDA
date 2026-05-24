import time
import torch
import torch.nn.functional as F
import warnings
import random
from datapro import CVEdgeDataset
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score as sklearn_f1
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.font_manager as _fm

plt.rcParams['font.family'] = ['Times New Roman', 'Times', 'serif']
plt.rcParams['axes.unicode_minus'] = False

from model import MHUCDA
from hypergraph_encoder import (ContrastiveMultiViewEmbeddingM,
                                ContrastiveMultiViewEmbeddingD)


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = target * pred + (1.0 - target) * (1.0 - pred)
    weight = (1.0 - p_t).pow(gamma)
    if alpha is not None:
        alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
        weight = alpha_t * weight
    return (weight * bce).mean()


class ROCPRPlotter:
    FOLD_COLORS    = ["#6c90c2", "#ebe974", "#609f8e", "#7de489", "#a7dcd8"]
    BASELINE_COLOR = "#94A3B8"
    MEAN_COLOR     = "#FF6B35"

    def __init__(self, save_dir="figs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _compute_mean_curve(self, fold_results, curve_type="roc"):
        common_x = np.linspace(0, 1, 100)
        y_list, metric_vals = [], []
        for r in fold_results:
            y_true, y_scores = r["y_true"], r["y_scores"]
            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                y_list.append(np.interp(common_x, fpr, tpr))
                metric_vals.append(r.get("auc", 0.0))
            else:
                prec, rec, _ = precision_recall_curve(y_true, y_scores)
                y_list.append(np.interp(common_x, rec[::-1], prec[::-1]))
                metric_vals.append(r.get("aupr", 0.0))
        return common_x, np.mean(y_list, axis=0), np.std(y_list, axis=0), float(np.mean(metric_vals))

    def _add_legend_with_tick_font(self, ax, fontsize=None, **kw):
        size = fontsize if fontsize else 9
        return ax.legend(prop=FontProperties(
            family='Times New Roman', style='normal',
            weight='normal', size=size), **kw)

    def plot_5fold_curves(self, fold_results, model_name="MHUCDA",
                          timestamp=None, show_mean=True):
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        roc_curves, pr_curves = [], []
        for r in fold_results:
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_scores"])
            roc_curves.append({"fpr": fpr, "tpr": tpr,
                                "auc": r.get("auc", 0.0), "fold": r.get("fold", 0)})
            prec, rec, _ = precision_recall_curve(r["y_true"], r["y_scores"])
            pr_curves.append({"precision": prec, "recall": rec,
                               "aupr": r.get("aupr", 0.0), "fold": r.get("fold", 0)})

        if show_mean:
            mean_fpr, mean_tpr, _, mean_auc   = self._compute_mean_curve(fold_results, "roc")
            mean_rec, mean_prec, _, mean_aupr = self._compute_mean_curve(fold_results, "pr")

        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12.6, 5.2))
        fig.subplots_adjust(wspace=0.25)
        ax_roc.set_box_aspect(0.75)
        ax_pr.set_box_aspect(0.75)

        for idx, d in enumerate(roc_curves):
            ax_roc.plot(d["fpr"], d["tpr"], lw=1.3, alpha=0.6,
                        color=self.FOLD_COLORS[idx % 5],
                        label=f"Fold {d['fold']} (AUC={d['auc']:.2f}%)")
        if show_mean:
            ax_roc.plot(mean_fpr, mean_tpr, lw=3.0, color=self.MEAN_COLOR,
                        label=f"Mean (AUC={mean_auc:.2f}%)", zorder=10)

        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves (5-fold)")
        ax_roc.grid(True, ls="--", alpha=0.35)
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0.0, 1.01)

        x1r, x2r, y1r, y2r = 0.00, 0.20, 0.80, 1.00
        axins_roc = inset_axes(ax_roc, width="46%", height="46%",
                               loc="lower right", borderpad=0,
                               bbox_to_anchor=(0.03, 0.08, 0.93, 0.94),
                               bbox_transform=ax_roc.transAxes)
        for idx, d in enumerate(roc_curves):
            axins_roc.plot(d["fpr"], d["tpr"], lw=1.0, alpha=0.6,
                           color=self.FOLD_COLORS[idx % 5])
        if show_mean:
            axins_roc.plot(mean_fpr, mean_tpr, lw=2.5, color=self.MEAN_COLOR, zorder=10)
        axins_roc.set_xlim(x1r, x2r); axins_roc.set_ylim(y1r, y2r)
        axins_roc.grid(True, ls="--", alpha=0.30)

        ax_roc.add_patch(plt.Rectangle((x1r, y1r), x2r - x1r, y2r - y1r,
                         fill=False, lw=1.0, ls=(0, (3, 2)), ec=self.BASELINE_COLOR))
        fig.add_artist(ConnectionPatch(
            xyA=(x2r, y1r), coordsA=ax_roc.transData,
            xyB=(0.0, 1.0), coordsB=axins_roc.transAxes,
            lw=1.0, ls=(0, (3, 2)), color=self.BASELINE_COLOR))

        for idx, d in enumerate(pr_curves):
            ax_pr.plot(d["recall"], d["precision"], lw=1.3, alpha=0.6,
                       color=self.FOLD_COLORS[idx % 5],
                       label=f"Fold {d['fold']} (AUPR={d['aupr']:.2f}%)")
        if show_mean:
            ax_pr.plot(mean_rec, mean_prec, lw=3.0, color=self.MEAN_COLOR,
                       label=f"Mean (AUPR={mean_aupr:.2f}%)", zorder=10)

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("PR Curves (5-fold)")
        ax_pr.grid(True, ls="--", alpha=0.35)
        ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.01)

        x1p, x2p, y1p, y2p = 0.80, 1.00, 0.80, 1.01
        axins_pr = inset_axes(ax_pr, width="46%", height="46%",
                              loc="lower left", borderpad=0,
                              bbox_to_anchor=(0.11, 0.08, 0.96, 0.94),
                              bbox_transform=ax_pr.transAxes)
        for idx, d in enumerate(pr_curves):
            axins_pr.plot(d["recall"], d["precision"], lw=1.0, alpha=0.6,
                          color=self.FOLD_COLORS[idx % 5])
        if show_mean:
            axins_pr.plot(mean_rec, mean_prec, lw=2.5, color=self.MEAN_COLOR, zorder=10)
        axins_pr.set_xlim(x1p, x2p); axins_pr.set_ylim(y1p, y2p)
        axins_pr.grid(True, ls="--", alpha=0.30)

        ax_pr.add_patch(plt.Rectangle((x1p, y1p), x2p - x1p, y2p - y1p,
                        fill=False, lw=1.0, ls=(0, (3, 2)), ec=self.BASELINE_COLOR))
        fig.add_artist(ConnectionPatch(
            xyA=(x1p, y1p), coordsA=ax_pr.transData,
            xyB=(1.0, 1.0), coordsB=axins_pr.transAxes,
            lw=1.0, ls=(0, (3, 2)), color=self.BASELINE_COLOR))

        for ax in (ax_roc, ax_pr):
            ax.tick_params(axis="y", which="major", length=6.0)
            ax.tick_params(axis="y", which="minor", length=6.0)

        self._add_legend_with_tick_font(
            ax_roc, loc="upper right", bbox_to_anchor=(0.98, 0.88),
            handlelength=1.5, handletextpad=0.5,
            labelspacing=0.3, borderpad=0.3, framealpha=0.95, fontsize=9)
        self._add_legend_with_tick_font(
            ax_pr, loc="upper left", bbox_to_anchor=(0.02, 0.88),
            handlelength=1.5, handletextpad=0.5,
            labelspacing=0.3, borderpad=0.3, framealpha=0.95, fontsize=9)

        save_path = os.path.join(self.save_dir,
                                 f"ROC_PR_5fold_{model_name}_{timestamp}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path


def save_predictions_labels(test_score, test_label, save_path):
    results = np.vstack((test_label, test_score))
    results_df = pd.DataFrame(results.T, columns=["Labels", "Predictions"])
    results_df.to_csv(save_path, index=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_metrics(score, label):
    return caculate_metrics(score, label)


def caculate_metrics(pre_score, real_score):
    y_true = real_score
    y_pre = pre_score

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    precision_u, recall_u, _ = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)

    best_threshold = 0.5
    best_f1_val = 0.0
    for t in np.linspace(0.3, 0.8, 51):
        y_tmp = [1 if j >= t else 0 for j in y_pre]
        try:
            f_tmp = sklearn_f1(y_true, y_tmp)
        except Exception:
            continue
        if f_tmp > best_f1_val:
            best_f1_val = f_tmp
            best_threshold = t

    y_score = [1 if j >= best_threshold else 0 for j in y_pre]
    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print(f"  AUC={auc:.4f}  AUPR={aupr:.4f}  Acc={acc:.4f}  "
          f"F1={f1:.4f}  Recall={recall:.4f}  Precision={precision:.4f}")
    return metric_result


def _eval_model(model, simData, validLoader, param):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for val_edges, val_lbls in validLoader:
            val_edges = val_edges.to(param.device)
            val_pre = model(simData, val_edges, return_contrastive=False)
            scores = np.append(scores, val_pre.cpu().numpy())
            labels = np.append(labels, val_lbls.numpy())
    auc = roc_auc_score(labels, scores)
    return scores, labels, auc


def _average_state_dicts(state_dicts):
    avg = {}
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts]
        avg[key] = torch.stack(tensors).mean(0)
        if state_dicts[0][key].dtype != torch.float32:
            avg[key] = avg[key].to(state_dicts[0][key].dtype)
    return avg


def train_test(simData, train_data, param, state, output_folder,
               graph_data=None, all_meta_paths=None, model_path=None,
               run_seed=42):
    all_metrics = []
    fold_plot_results = []

    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']

    kfolds = param.kfold

    if state == 'valid':
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=run_seed)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges, train_labels):
            train_idx.append(train_index)
            valid_idx.append(valid_index)

        for i in range(kfolds):
            fold_id = i + 1
            print(f'\n{"#"*20} Fold {fold_id}/{kfolds} {"#"*20}')

            fold_seed = run_seed + fold_id * 17
            setup_seed(fold_seed)

            warnings.filterwarnings(
                'ignore',
                message='The epoch parameter in `scheduler.step\\(\\)`',
                category=UserWarning
            )

            model = MHUCDA(
                param,
                ContrastiveMultiViewEmbeddingM(param),
                ContrastiveMultiViewEmbeddingD(param),
                graph_data,
                all_meta_paths
            )
            model = model.to(param.device)

            lr = getattr(param, 'lr', 0.002)
            contrastive_params = list(model.Xm.parameters()) + list(model.Xd.parameters())
            contrastive_param_ids = set(id(p) for p in contrastive_params)
            other_params = [p for p in model.parameters() if id(p) not in contrastive_param_ids]
            optimizer = torch.optim.Adam([
                {'params': contrastive_params, 'lr': lr * 1.2},
                {'params': other_params, 'lr': lr}
            ], weight_decay=1e-4)

            warmup_epochs = getattr(param, 'warmup_epochs', min(25, max(5, int(param.epoch * 0.10))))
            cosine_epochs = max(1, param.epoch - warmup_epochs)
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=5e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
            )

            pos_w = getattr(param, 'pos_weight', 1.0)
            focal_gamma = getattr(param, 'focal_gamma', 2.0)
            focal_alpha = getattr(param, 'focal_alpha', 0.25)
            label_smooth_eps = 0.01
            contrastive_scale = getattr(param, 'contrastive_scale', 0.007)
            swa_start = getattr(param, 'swa_start', 185)
            swa_freq = getattr(param, 'swa_freq', 5)
            swa_state_dicts = []

            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]

            trainLoader = DataLoader.DataLoader(
                CVEdgeDataset(edges_train, labels_train),
                batch_size=param.batchSize, shuffle=True, num_workers=0
            )
            validLoader = DataLoader.DataLoader(
                CVEdgeDataset(edges_valid, labels_valid),
                batch_size=param.batchSize, shuffle=False, num_workers=0
            )

            patience = getattr(param, 'patience', 30)
            best_auc = 0.0
            no_improve = 0
            best_model_path = os.path.join(output_folder, f"fold_{fold_id}_best.pkl")

            for e in range(param.epoch):
                model.train()
                running_loss = 0.0
                contrastive_running_loss = 0.0
                start = time.time()

                for batch_edges, batch_labels in trainLoader:
                    batch_edges = batch_edges.to(param.device)
                    batch_labels = batch_labels.to(param.device)

                    pre_score, contrastive_loss = model(
                        simData, batch_edges,
                        return_contrastive=True,
                        train_labels=batch_labels
                    )

                    smooth_labels = batch_labels * (1 - label_smooth_eps) + 0.5 * label_smooth_eps
                    if pos_w != 1.0:
                        effective_alpha = pos_w / (pos_w + 1.0)
                        main_loss = focal_loss(pre_score, smooth_labels,
                                               gamma=focal_gamma, alpha=effective_alpha)
                    else:
                        main_loss = focal_loss(pre_score, smooth_labels,
                                               gamma=focal_gamma, alpha=focal_alpha)

                    contrastive_warmup_epochs = min(50, max(5, int(param.epoch * 0.3)))
                    contrastive_weight_current = model.contrastive_weight * min(
                        1.0, (e + 1) / contrastive_warmup_epochs
                    )

                    total_loss = main_loss + contrastive_weight_current * contrastive_loss * contrastive_scale

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    running_loss += main_loss.item()
                    contrastive_running_loss += contrastive_loss.item()

                scheduler.step()
                end = time.time()

                _, _, cur_auc = _eval_model(model, simData, validLoader, param)

                print(f"Epoch {e+1:>3}/{param.epoch}  "
                      f"Loss={running_loss:.4f}  CLoss={contrastive_running_loss:.4f}  "
                      f"ValAUC={cur_auc:.4f}  "
                      f"LR={optimizer.param_groups[0]['lr']:.6f}  "
                      f"T={end-start:.1f}s")

                if cur_auc > best_auc:
                    best_auc = cur_auc
                    no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stop at epoch {e+1}, best AUC={best_auc:.4f}")
                        break

                if e >= swa_start and (e - swa_start) % swa_freq == 0:
                    swa_state_dicts.append(
                        {k: v.clone().cpu() for k, v in model.state_dict().items()}
                    )

                model.train()

            if len(swa_state_dicts) >= 3:
                avg_state = _average_state_dicts(swa_state_dicts)
                model.load_state_dict({k: v.to(param.device) for k, v in avg_state.items()})
                _, _, swa_auc = _eval_model(model, simData, validLoader, param)
                if swa_auc > best_auc:
                    best_auc = swa_auc
                    torch.save({k: v.to(param.device) for k, v in avg_state.items()}, best_model_path)
                    print(f"  [SWA] Using SWA model, AUC={swa_auc:.4f}")
                else:
                    print(f"  [SWA] Keeping best single-epoch model, AUC={best_auc:.4f}")

            model.load_state_dict(torch.load(best_model_path, map_location=param.device))
            model.eval()
            valid_score, valid_label = [], []
            with torch.no_grad():
                for batch_edges, batch_labels in validLoader:
                    batch_edges = batch_edges.to(param.device)
                    pre_score = model(simData, batch_edges, return_contrastive=False)
                    valid_score = np.append(valid_score, pre_score.cpu().numpy())
                    valid_label = np.append(valid_label, batch_labels.numpy())

            metric = get_metrics(valid_score, valid_label)
            all_metrics.append(metric)

            fold_plot_results.append({
                "y_true":   valid_label,
                "y_scores": valid_score,
                "auc":      metric[0] * 100,
                "aupr":     metric[1] * 100,
                "fold":     fold_id,
            })

            torch.save(model.state_dict(), os.path.join(output_folder, f"fold_{fold_id}.pkl"))
            print(f"[Fold {fold_id}] AUC={metric[0]:.4f}  AUPR={metric[1]:.4f}  F1={metric[3]:.4f}")

            attention_weights = model.get_attention_weights()
            if attention_weights is not None:
                torch.save(attention_weights,
                           os.path.join(output_folder, f"attention_fold_{fold_id}.pt"))

        mean_metrics = np.mean(all_metrics, axis=0)
        std_metrics = np.std(all_metrics, axis=0)
        with open(os.path.join(output_folder, "metrics.txt"), 'w') as f:
            for fold_m in all_metrics:
                f.write('\t'.join(map(str, fold_m)) + '\n')
            f.write("Mean:\n")
            f.write('\t'.join(map(str, mean_metrics)) + '\n')

        print("\n===== 5-Fold Results =====")
        for name, mean_v, std_v in zip(
            ['AUC', 'AUPR', 'Accuracy', 'F1', 'Recall', 'Precision'],
            mean_metrics, std_metrics
        ):
            print(f"  {name:<12}: {mean_v:.4f} ± {std_v:.4f}")

        try:
            plotter = ROCPRPlotter(save_dir=os.path.join(output_folder, "figures"))
            plotter.plot_5fold_curves(
                fold_plot_results, model_name="MHUCDA",
                timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                show_mean=True,
            )
        except Exception as plot_err:
            print(f"[Plot] {plot_err}")

        return {
            'seed':         run_seed,
            'AUC':          mean_metrics[0],
            'AUPR':         mean_metrics[1],
            'Accuracy':     mean_metrics[2],
            'F1':           mean_metrics[3],
            'Recall':       mean_metrics[4],
            'Precision':    mean_metrics[5],
            'fold_metrics': all_metrics,
        }

    else:
        testLoader = DataLoader.DataLoader(
            CVEdgeDataset(test_edges, test_labels),
            batch_size=param.batchSize, shuffle=False, num_workers=0
        )
        model = MHUCDA(
            param,
            ContrastiveMultiViewEmbeddingM(param),
            ContrastiveMultiViewEmbeddingD(param),
            graph_data,
            all_meta_paths
        )
        if model_path is None:
            model_path = './savemodel/MHUCDA/fold_1.pkl'
        model.load_state_dict(torch.load(model_path, map_location=param.device))
        model = model.to(param.device)
        model.eval()

        test_score, test_label = [], []
        with torch.no_grad():
            for batch_edges, batch_labels in testLoader:
                batch_edges = batch_edges.to(param.device)
                pre_score = model(simData, batch_edges, return_contrastive=False)
                test_score = np.append(test_score, pre_score.cpu().numpy())
                test_label = np.append(test_label, batch_labels.numpy())

        return get_metrics(test_score, test_label)