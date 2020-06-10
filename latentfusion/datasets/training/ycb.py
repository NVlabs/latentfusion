import os
from pathlib import Path

import structlog
import torch

from latentfusion.datasets.training.pyrender import PyrenderDataset

_package_dir = Path(os.path.dirname(os.path.realpath(__file__)))
_resources_dir = _package_dir.parent.parent / 'resources'

logger = structlog.get_logger(__name__)


DEFAULT_POSE = torch.tensor((
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
))


def get_shape_paths(dataset_dir, objects):
    """
    Returns shape paths for ModelNet.

    Args:
        dataset_dir: the directory containing the dataset
        objects: the list of ModelNet categories to include
        split_type: 'train' or 'test'

    Returns:
        (list): a list of shape paths
    """
    paths = []
    for obj in objects:
        obj_path = dataset_dir / 'models' / obj / 'textured.obj'
        if not obj_path.exists():
            raise FileNotFoundError(f'Object path {obj_path!s} does not exist')

        paths.append(obj_path)

    return paths


class YCBDataset(PyrenderDataset):

    def __init__(self, shapes_dir, *args, objects, **kwargs):
        """
        Args:
            shapes_dir: the directory containing the dataset
            objects: the list of ModelNet categories to include
        """
        self.shapes_dir = Path(shapes_dir)
        self.objects = objects
        logger.info("indexing shapes", path=self.shapes_dir, objects=objects)
        shape_paths = get_shape_paths(self.shapes_dir, self.objects)

        super().__init__(shape_paths, *args, obj_default_pose=DEFAULT_POSE, **kwargs)
