# Sources, authorship, and reuse

## Original course material

Matteo Lugli prepared the original Python project, the 42-slide `Progetto CSL.pptx`, and its PDF export `pneumonia_MatteoLugli.pdf` for Computational and Statistical Learning at UniMoRe. They are the primary sources for the new English report and README.

The original slide PDF is included unchanged. The editable PowerPoint is intentionally not included in the published repository. The exact original Python script and README are saved in `archive`. The publication script contains the limited changes described in `RESULTS_AND_LIMITATIONS.md`.

The 12 learning-curve PNGs are extracted unchanged from the presentation. `results/figure_sources.json` records each original PowerPoint media filename, slide, and SHA-256 checksum. The two new comparison charts visualize values manually transcribed from the slide tables; their source CSVs are included. No raw prediction arrays or numerical epoch histories were reconstructed from the raster images.

## Dataset

PneumoniaMNIST is distributed within [MedMNIST](https://medmnist.com/). Use the official package or dataset download links. The dataset itself is not redistributed in this repository.

- Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification*. Scientific Data 10, 41 (2023). [Paper](https://doi.org/10.1038/s41597-022-01721-8).
- [Official dataset metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py) documents the pneumonia labels, resolution, splits, and **CC BY 4.0** dataset license.
- The [MedMNIST implementation](https://github.com/MedMNIST/MedMNIST) is distributed under Apache 2.0. These code terms are distinct from the dataset terms.

The official project describes MedMNIST as an educational/research benchmark, not intended for clinical use.

## Software and technical references

The implementation imports PyTorch, torchvision, medmnist, scikit-learn, scikit-image, NumPy, Matplotlib, and tqdm. Those libraries retain their own licenses. The report additionally refers to official documentation for:

- [SVC probability estimates and randomness](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.SVC.html).
- [Cross-validation and preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [Metric definitions and interpretation](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [PyTorch checkpoint copying](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

No project-wide license has been selected for the original course material. These credits identify sources and do not relicense third-party content.
