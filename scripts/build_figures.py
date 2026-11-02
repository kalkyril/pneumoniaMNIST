"""Plot the presentation-reported metrics. This script does not train models."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#697b8a', '#1d8586', '#e59d39', '#b35c58']


def main():
    with (ROOT / 'results/presentation_metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    selected = [rows[i] for i in [0, 1, 4, 5]]
    labels = ['SVM\nPCA', 'SVM\nHOG', 'CNN\nSGD (seeded)', 'CNN\nAdam (seeded)']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for ax, key, title in zip(axes, ['auc', 'accuracy', 'f1', 'brier'],
                             ['ROC AUC (higher is better)', 'Accuracy (higher is better)',
                              'F1 (higher is better)', 'Brier score (lower is better)']):
        vals = [float(r[key]) for r in selected]
        bars = ax.bar(np.arange(4), vals, color=COLORS, width=.65)
        ax.set_xticks(np.arange(4), labels, fontsize=8)
        ax.set_ylim(0, 1.08 if key != 'brier' else .14)
        ax.set_title(title, fontsize=10, pad=12)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        for b, val, row in zip(bars, vals, selected):
            ax.text(b.get_x()+b.get_width()/2, val+(0.018 if key!='brier' else .003),
                    row[key], ha='center', fontsize=9)
    fig.suptitle('PneumoniaMNIST: SVM features vs. learned CNN features', fontsize=15, y=1.02)
    fig.text(.5, -.03, 'Rounded values transcribed from slides 33 and 37; historical runs, not newly reproduced results.',
             ha='center', fontsize=10, color='#555555')
    fig.tight_layout()
    fig.savefig(ROOT / 'assets/model-comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, title in zip(axes, ['auc','accuracy','brier'], ['ROC AUC','Accuracy','Brier score']):
        x=np.arange(2); before=[float(rows[i][key]) for i in [2,3]]; after=[float(rows[i][key]) for i in [4,5]]
        ax.bar(x-.18,before,width=.36,label='Seed off',color='#aab5bd')
        ax.bar(x+.18,after,width=.36,label='Seed on',color='#1d8586')
        ax.set_xticks(x,['CNN + SGD','CNN + Adam']);ax.set_title(title)
        ax.set_ylim(0,1.08 if key!='brier' else .28);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for pos,val in zip(list(x-.18)+list(x+.18),before+after):
            ax.text(pos,val+(.015 if key!='brier' else .005),f'{val:.3f}',ha='center',fontsize=8)
    axes[0].legend(frameon=False,loc='lower left',fontsize=8)
    fig.suptitle('Recorded optimizer runs: seeded and unseeded settings',fontsize=13,y=1.02)
    fig.text(.5,-.04,'One reported run per setting; this comparison does not estimate seed sensitivity or uncertainty.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout();fig.savefig(ROOT/'assets/optimizer-comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('Built two figures from presentation_metrics.csv')


if __name__ == '__main__':
    main()
