"""Extract the original learning-curve PNGs without modifying their pixels."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    'unseeded_sgd_loss.png': ('image23.png', 35),
    'unseeded_sgd_auc.png': ('image24.png', 35),
    'unseeded_adam_loss.png': ('image13.png', 35),
    'unseeded_adam_auc.png': ('image17.png', 35),
    'seeded_sgd_loss.png': ('image25.png', 38),
    'seeded_sgd_auc.png': ('image26.png', 38),
    'seeded_adam_loss.png': ('image27.png', 38),
    'seeded_adam_auc.png': ('image28.png', 38),
    'lower_lr_sgd_loss.png': ('image31.png', 40),
    'lower_lr_sgd_auc.png': ('image32.png', 40),
    'lower_lr_adam_loss.png': ('image29.png', 40),
    'lower_lr_adam_auc.png': ('image30.png', 40),
}


def main():
    parser = argparse.ArgumentParser(
        description="Optional re-extraction from a separately supplied original PowerPoint."
    )
    parser.add_argument("--pptx", type=Path, required=True,
                        help="Original PowerPoint path; it is not bundled in this repository.")
    args = parser.parse_args()
    if not args.pptx.is_file():
        parser.error("The supplied PowerPoint file does not exist. Existing extracted PNGs are already included.")
    if not zipfile.is_zipfile(args.pptx):
        parser.error("The supplied file is not a valid PPTX archive.")
    out = ROOT / 'assets/slide-curves'
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(args.pptx) as deck:
        for name, (source, slide) in FIGURES.items():
            payload = deck.read('ppt/media/' + source)
            (out / name).write_bytes(payload)
            manifest.append({'file': 'assets/slide-curves/' + name,
                             'source_media': 'ppt/media/' + source,
                             'source_slide': slide,
                             'sha256': hashlib.sha256(payload).hexdigest()})
    (ROOT / 'results/figure_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Extracted', len(manifest), 'original learning curves')


if __name__ == '__main__':
    main()
