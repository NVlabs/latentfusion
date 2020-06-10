import json
from pathlib import Path
import argparse

from tqdm import tqdm

from latentfusion import meshutils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='shapenet_dir', type=Path)
    parser.add_argument(dest='out_path', type=Path)
    args = parser.parse_args()

    paths = list(args.shapenet_dir.glob('**/uv_unwrapped.obj'))

    normal_paths = []
    corrupt_paths = []

    corrupt_ids = []

    pbar = tqdm(paths)
    for path in pbar:
        synset_id = path.parent.parent.parent.name
        model_id = path.parent.parent.name

        obj = meshutils.Object3D(path)
        if obj.are_normals_corrupt():
            corrupt_paths.append(path)
            corrupt_ids.append((synset_id, model_id))
        else:
            normal_paths.append(path)

        frac_corrupt = len(corrupt_paths) / (len(normal_paths) + len(corrupt_paths))
        pbar.set_description(f"{frac_corrupt * 100:.02f}% corrupt, {synset_id}/{model_id}")

    args.out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(args.out_path, 'w') as f:
        json.dump(corrupt_ids, f)


if __name__ == '__main__':
    main()
