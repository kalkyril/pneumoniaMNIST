# Evidence, corrections, and publication changes

Prepared on 2 November 2026. This document separates historical evidence from publication maintenance. A new full benchmark run has not been performed.

## Sources used during preparation

- Original PowerPoint: 42 presentation slides, used to extract the original curve PNGs. It is intentionally not included in the published repository; its PDF export is included.
- `pneumonia_MatteoLugli.pdf`: the original 42-page slide export, not a written paper.
- `pneumonia.py`: the original compact comparison script, preserved byte-for-byte in `archive/pneumonia_original.py` before publication edits.
- `README.md`: the original Italian README, preserved in `archive/README_original.md`.
- `requirements.txt` and a local Python 3.9 virtual environment.

The new report is `PneumoniaMNIST_MatteoLugli.tex` / `.pdf`, in English, with the typography, two-column format, and academic footer used by the earlier texture project. The new README and documentation are written for repository readers.

## Numerical provenance

| Evidence | Source slides | Included representation |
| --- | --- | --- |
| SVM metrics | 33, repeated in 36 | `results/presentation_metrics.csv` |
| Unseeded CNN metrics | 34, repeated in 35 | Same CSV |
| Seeded CNN metrics | 37, repeated in 38 | Same CSV |
| Lower-SGD-learning-rate metrics | 39 | Same CSV; missing Brier left empty |
| Runtime summary | 41 | `results/presentation_timings.csv` |
| Original loss and validation curves | 35, 38, 40 | 12 unchanged PNGs and a provenance manifest |

The CSVs are human-readable transcriptions of presentation values, not output files from a new execution. AUC values were displayed as approximate and all values are rounded. The source does not include raw predictions, confidence intervals, repeated-run summaries, or the original JSON histories and checkpoints. Adam curves appear in slide 40 without a separate numeric result table for that experiment; no missing values have been invented.

The report treats the scalar values as the historical final evaluation summary, consistent with the script's final test-metric output. Learning curves are explicitly validation AUC, not test AUC. The missing raw run artifacts prevent an independent reconstruction of the evaluation bookkeeping.

## Dataset verification

The supplied local `pneumoniamnist.npz` has 4,708 training images, 524 validation images, and 624 test images, each 28 × 28 grayscale. Class counts are recorded in `results/dataset_summary.csv`. They were counted directly from the label arrays and agree with the official split sizes.

Slide 3 describes the data as approximately balanced. In fact the training split is about 74.2% pneumonia, and the test split is 62.5% pneumonia. The report corrects this description. No image data are copied into the publication package.

## Interpretation corrections

- HOG improves reported AUC over PCA, but its Brier score is slightly worse (0.115 versus 0.112).
- The recorded seeded CNNs perform best, but the unseeded CNNs do not uniformly outperform the SVMs. A single run per condition does not establish statistical significance or a general seed benefit.
- Both SGD configurations use momentum 0.9. There is no independently recorded momentum-free comparison in the numeric results.
- Lowering SGD's learning rate improves displayed accuracy and F1 but reduces AUC. Seed, epoch budget, and other missing run details prevent isolation of a learning-rate effect.
- The seeded SGD curve in slide 38 has a pronounced early validation dip. Statements of uniformly smooth seeded behavior are not supported by that plot.
- The Brier score combines multiple aspects of probabilistic prediction quality. Without calibration plots, lower Brier alone is not proof of better calibration.
- Reported runtimes are approximate and lack hardware metadata. The SVM timer includes a grid search; the CNN timer covers one configuration, so the budgets are not equivalent.
- Standard stochastic-approximation step-size conditions do not guarantee convergence to a global optimum for this CNN.

## Source review and maintenance

### Checkpoint snapshot

The original `best_state = model.state_dict()` retains references to model tensors. Later training can change those tensors, so the original code does not reliably restore the epoch with best validation AUC. The publication entry point uses `copy.deepcopy(model.state_dict())`. This follows the [PyTorch saving/loading guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

The correction changes future execution; it does not revise the slide metrics. The old evaluated checkpoint cannot be identified reliably without the missing artifacts.

### Explicit execution settings

The publication script adds `--seed`, `--sgd-lr`, `--sgd-epochs`, `--adam-epochs`, `--output-dir`, and `--no-plots`, plus `run_config.json`. Seeded runs seed Python, NumPy, PyTorch, CUDA, and SVM probability fitting. The default still follows the supplied main script: unseeded SGD at LR 0.01 for up to 40 epochs, Adam at LR 0.001 for up to 30 epochs. Plotting-enabled runs save PNGs before displaying them. `--no-plots` skips both displaying and saving those plots.

The source's commented `SEED = 42` is not proof of the exact seed used for the presentation's seeded results. The examples use 42 as an explicit new-run choice. No MPS backend support is added; device selection remains CUDA or CPU.

### Cross-validation limitation retained

Scaling and PCA are fitted on the entire training split before `GridSearchCV`, so the held-out CV folds influence preprocessing. External validation/test images are not used to fit these transformations, but the CV score can be optimistic. A future controlled revision should put the transformations inside a scikit-learn `Pipeline`. The publication script retains the original preprocessing structure so that a methodological redesign is not silently presented as the historical experiment.

Also, grid-search AUC uses SVC's decision scores, whereas the final metric helper uses `predict_proba`. These are not interchangeable evaluation records in every fitting configuration. SVM probability fitting uses randomness; the entire probabilistic pipeline is not strictly deterministic without controlled settings.

## Reproduction environment

`requirements-reproduction.txt` records package versions read from the supplied virtual environment: torch 2.8.0, torchvision 0.23.0, medmnist 3.0.2, scikit-learn 1.6.1, scikit-image 0.24.0, NumPy 2.0.2, Matplotlib 3.9.4, and tqdm 4.67.1. This snapshot is not a historical lockfile and no fresh installation has been claimed.

## Validation scope

The publication preparation includes report compilation and visual review, preservation of source presentation files, curve checksum checks, transcribed-value and local-link checks, CLI loading, and lightweight runtime checks of the CNN shape/parameter count, HOG dimension, metrics, and best-checkpoint restoration. It does not include retraining all SVM and CNN configurations or reproducing the historical performance figures.
