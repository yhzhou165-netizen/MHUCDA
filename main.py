import os
import torch
import numpy as np
from datapro import Simdata_pro, loading_data, load_dataset
from train import train_test, setup_seed


class Config:
    def __init__(self):
        self.datapath = './datasets'
        self.kfold = 5
        self.batchSize = 128
        self.ratio = 0.2
        self.epoch = 250
        self.patience = 150
        self.gcn_layers = 7
        self.gcn_layers1 = 2
        self.view = 3
        self.fm = 128
        self.fd = 128
        self.inSize = 128
        self.outSize = 128
        self.hiddenSize = 64
        self.PVN = 1 / 32
        self.Dropout = 0.15
        self.hdnDropout = 0.15
        self.fcDropout = 0.15
        self.num_heads1 = 2
        self.maskMDA = False
        self.lr = 0.0007
        self.pos_weight = 2.0
        self.focal_gamma = 2.0
        self.focal_alpha = 0.45
        self.k_neighbors = 10
        self.cl_temperature = 0.40
        self.contrastive_scale = 0.007
        self.warmup_epochs = 30
        self.swa_start = 180
        self.swa_freq = 4
        self.device = torch.device('cpu')


def main():
    param = Config()
    RUN_SEEDS = [456]

    simData = Simdata_pro(param)
    train_data = loading_data(param)
    graph_data, all_meta_paths = load_dataset()

    output_folder = './savemodel/MHUCDA/'
    os.makedirs(output_folder, exist_ok=True)

    all_run_results = []

    for run_idx, seed in enumerate(RUN_SEEDS):
        print(f"\n{'='*60}")
        print(f"  Run {run_idx+1}/{len(RUN_SEEDS)}  (seed={seed})")
        print(f"{'='*60}")

        run_folder = os.path.join(output_folder, f'seed_{seed}')
        os.makedirs(run_folder, exist_ok=True)

        result = train_test(
            simData, train_data, param,
            state='valid',
            output_folder=run_folder,
            graph_data=graph_data,
            all_meta_paths=all_meta_paths,
            run_seed=seed,
        )
        all_run_results.append(result)

        print(f"\n  [Run {run_idx+1}] seed={seed}  "
              f"AUC={result['AUC']:.4f}  AUPR={result['AUPR']:.4f}  F1={result['F1']:.4f}")

    print(f"\n{'='*60}")
    print(f"  Aggregate ({len(RUN_SEEDS)} runs)")
    print(f"{'='*60}")

    metrics_keys = ['AUC', 'AUPR', 'Accuracy', 'F1', 'Recall', 'Precision']
    for k in metrics_keys:
        vals = [r[k] for r in all_run_results]
        print(f"  {k:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    summary_path = os.path.join(output_folder, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"MHUCDA — Multi-seed summary ({len(RUN_SEEDS)} runs: {RUN_SEEDS})\n\n")
        for k in metrics_keys:
            vals = [r[k] for r in all_run_results]
            f.write(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n")
        f.write("\nPer-run details:\n")
        for r in all_run_results:
            f.write(f"  seed={r['seed']}  AUC={r['AUC']:.4f}  "
                    f"AUPR={r['AUPR']:.4f}  F1={r['F1']:.4f}\n")


if __name__ == "__main__":
    main()