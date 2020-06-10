import os
from pathlib import Path

import structlog
import torch

from latentfusion.datasets.training.pyrender import PyrenderDataset

_package_dir = Path(os.path.dirname(os.path.realpath(__file__)))
_resources_dir = _package_dir.parent.parent / 'resources'

logger = structlog.get_logger(__name__)


# ModelNet uses +Y as up. YCB uses +Z as up. Swap these.
MODELNET_TO_YCB_POSE = torch.tensor((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))


def get_shape_paths(dataset_dir, categories, split_type):
    """
    Returns shape paths for ModelNet.

    Args:
        dataset_dir: the directory containing the dataset
        categories: the list of ModelNet categories to include
        split_type: 'train' or 'test'

    Returns:
        (list): a list of shape paths
    """
    paths = []
    for category in categories:
        category_dir = dataset_dir / category / split_type
        if not category_dir.exists():
            raise FileNotFoundError(f'Category directory {category_dir!s} does not exist')

        category_paths = sorted(category_dir.glob('*.off'))
        paths.extend(category_paths)

    return paths


class ModelNetDataset(PyrenderDataset):

    def __init__(self, shapes_dir, *args, categories, split_type, **kwargs):
        """
        Args:
            shapes_dir: the directory containing the dataset
            categories: the list of ModelNet categories to include
            split_type: 'train' or 'test'
        """
        self.shapes_dir = Path(shapes_dir)
        self.categories = categories
        self.split_type = split_type
        logger.info("indexing shapes", path=self.shapes_dir, categories=categories, split_type=split_type)
        shape_paths = get_shape_paths(self.shapes_dir, self.categories, self.split_type)

        super().__init__(shape_paths, *args, obj_default_pose=MODELNET_TO_YCB_POSE, **kwargs)
