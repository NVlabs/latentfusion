import json
from pathlib import Path
import argparse

from PIL import Image
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='image_dir', type=Path)
    parser.add_argument(dest='out_dir', type=Path)
    args = parser.parse_args()

    paths = list(args.image_dir.glob('**/*.png'))
    pbar = tqdm(paths)
    for path in pbar:
        in_image = Image.open(str(path)).convert('RGB')
        out_image = in_image.resize((512, 512), Image.LANCZOS)
        out_path = Path(args.out_dir, path.parts[-2], f'{path.stem}.jpg')
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)

        out_image.save(out_path)
        pbar.set_description(str(out_path))


if __name__ == '__main__':
    main()
