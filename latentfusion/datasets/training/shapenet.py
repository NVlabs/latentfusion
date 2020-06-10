import json
import os
from pathlib import Path

import structlog
import torch

from latentfusion.datasets.training.pyrender import PyrenderDataset

_package_dir = Path(os.path.dirname(os.path.realpath(__file__)))
_resources_dir = _package_dir.parent.parent.parent / 'resources'

logger = structlog.get_logger(__name__)


# ShapeNet uses +Y as up. YCB uses +Z as up. Swap these.
SHAPENET_TO_YCB_POSE = torch.tensor((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))


def _load_blacklist():
    with open(_resources_dir / 'shapenet_blacklist.json') as f:
        blacklist_ids = json.load(f)

    return set(tuple(o) for o in blacklist_ids)


def _load_taxonomy():
    with open(_resources_dir / 'shapenet_taxonomy.json') as f:
        taxonomy = json.load(f)

    taxonomy = {d['synsetId']: d for d in taxonomy}

    return taxonomy


def _gather_synset_ids(taxonomy, synset_id):
    synset_ids = []
    stack = [synset_id]
    while len(stack) > 0:
        current_id = stack.pop()
        synset_ids.append(current_id)
        synset = taxonomy[current_id]
        stack.extend(synset['children'])

    return synset_ids


def _category_to_synset_ids(taxonomy, category, include_children=True):
    synset_ids = []
    for synset_id, synset_dict in taxonomy.items():
        names = synset_dict['name'].split(',')
        if category in names:
            if include_children:
                synset_ids.extend(_gather_synset_ids(taxonomy, synset_id))
            else:
                synset_ids.append(synset_id)

    return synset_ids


def get_shape_paths(dataset_dir, blacklist_synsets=None):
    """
    Returns shape paths for ShapeNet.

    Args:
        dataset_dir: the directory containing the dataset
        blacklist_synsets: a list of synsets to exclude

    Returns:

    """
    shape_index_path = (dataset_dir / 'paths.txt')
    if shape_index_path.exists():
        with open(shape_index_path, 'r') as f:
            paths = [Path(dataset_dir, p.strip()) for p in f.readlines()]
    else:
        paths = list(dataset_dir.glob('**/uv_unwrapped.obj'))

    if blacklist_synsets is not None:
        num_filtered = sum(1 for p in paths if p.parent.parent.parent.name in blacklist_synsets)
        paths = [p for p in paths
                 if p.parent.parent.parent.name not in blacklist_synsets]
        logger.info("filtered blacklisted shapes", num_filtered=num_filtered, num_remaining=len(paths))

    return paths


class ShapeNetDataset(PyrenderDataset):

    def __init__(self,
                 shapes_dir,
                 *args,
                 blacklist_categories=None,
                 random_materials=True,
                 use_textures=True,
                 **kwargs):

        # Load blacklist.
        self.taxonomy = _load_taxonomy()
        self.blacklist_categories = set()
        if blacklist_categories is not None:
            self.blacklist_categories.update(blacklist_categories)
        self.blacklist_synsets = set()
        if blacklist_categories is not None:
            for category in blacklist_categories:
                self.blacklist_synsets.update(_category_to_synset_ids(self.taxonomy, category))

        logger.info("loaded blacklist",
                    categories=self.blacklist_categories,
                    synsets=self.blacklist_synsets)

        self.shapes_dir = Path(shapes_dir)
        logger.info("indexing shapes", path=self.shapes_dir)
        shape_paths = get_shape_paths(self.shapes_dir, self.blacklist_synsets)

        super().__init__(shape_paths,
                         *args,
                         obj_default_pose=SHAPENET_TO_YCB_POSE,
                         random_materials=random_materials,
                         use_textures=use_textures,
                         **kwargs)
