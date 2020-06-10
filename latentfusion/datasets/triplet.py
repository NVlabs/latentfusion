import random

import structlog
import torch
from torch.utils.data import Dataset


__all__ = ['TripletDataset']


logger = structlog.get_logger(__name__)


class TripletDataset(Dataset):

    def __init__(self, dataset,
                 data_indices,
                 dist_matrix,
                 k=3,
                 easy_neg_prob=0.0):
        super().__init__()

        self.log = logger.bind(path=dataset.path)
        self.log.info("initializing triplet dataset",
                      k=k,
                      num_indices=len(data_indices),
                      easy_neg_prob=easy_neg_prob)

        self.dataset = dataset
        self.dist_matrix = dist_matrix
        self.data_indices = data_indices
        self.easy_neg_prob = easy_neg_prob

        self.k = k

    def __len__(self):
        return len(self.data_indices)

    def _sample_positive(self, anchor_idx):
        """
        Samples a positive from the dataset.
        """
        pos_dists, pos_inds = torch.topk(self.dist_matrix[anchor_idx], k=self.k, largest=False)
        pos_select = random.randrange(1, len(pos_inds))
        pos_ann_idx = int(pos_inds[pos_select])  # Exclude self.
        pos_dist = pos_dists[pos_select]
        pos_data_idx = self.data_indices[pos_ann_idx]

        return pos_data_idx, pos_dist.item()

    def _sample_easy_negative(self, anchor_idx, frac=0.25):
        """
        Samples an negative from the bottom ``frac`` of neighbors furthest from
        the ``feature``.
        Args:
            feature: The anchor feature.
            frac: The bottom percentile to sample from.
        Returns:
            int: An index of the negative sample.
        """
        bottom_k = int(frac * len(self.data_indices))
        neg_dists, neg_inds = torch.topk(self.dist_matrix[anchor_idx], k=bottom_k, largest=True)

        neg_select = random.randrange(1, len(neg_inds))
        neg_idx = int(neg_inds[neg_select])  # Exclude self.
        neg_dist = neg_dists[neg_select]
        neg_data_idx = self.data_indices[neg_idx]

        return neg_data_idx, neg_dist.item()

    def _sample_random_negative(self, anchor_idx):
        neg_idx = random.randrange(len(self.data_indices))
        neg_data_idx = self.data_indices[neg_idx]
        neg_dist = self.dist_matrix[anchor_idx, neg_idx]

        return neg_data_idx, neg_dist.item()

    def _sample_negative(self, anchor_idx, pos_dist):
        """
        Sample a negative that's at least farther than the positive.
        Args:
            anchor_idx: The anchor feature.
            pos_dist: The distance of the positive sample.
        Returns:
            int: An index of the negative sample.
        """
        neg_data_idx = None
        neg_dist = -1
        num_tries = 0
        max_tries = 100

        while neg_dist <= pos_dist:
            if self.easy_neg_prob > 0.0 and random.random() < self.easy_neg_prob:
                neg_data_idx, neg_dist = self._sample_easy_negative(anchor_idx)
            else:
                neg_data_idx, neg_dist = self._sample_random_negative(anchor_idx)

            num_tries += 1
            if num_tries >= max_tries:
                self.log.warning("could not find negative farther than positive",
                                 num_tries=num_tries, max_tries=max_tries)
                break

        return neg_data_idx, neg_dist

    def __getitem__(self, idx):
        out = {}

        anchor_data_idx = self.data_indices[idx].item()

        pos_data_idx, pos_dist = self._sample_positive(idx)
        neg_data_idx, neg_dist = self._sample_negative(idx, pos_dist)

        anchor = self.dataset[anchor_data_idx]
        positive = self.dataset[pos_data_idx]
        negative = self.dataset[neg_data_idx]

        out.update({
            'anchor': anchor,
            'anchor_dist': 0,
            'positive': positive,
            'positive_dist': pos_dist,
            'negative': negative,
            'negative_dist': neg_dist,
        })

        return out
