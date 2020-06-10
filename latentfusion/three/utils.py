import torch


def farthest_points(data, n_clusters: int, dist_func, return_center_indexes=False,
                    return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.

      Args:
        data (torch.tensor): data points.
        n_clusters (int): number of clusters.
        dist_func (Callable): distance function that is used to compare two data points.
        return_center_indexes (bool): if True, returns the indexes of the center of clusters.
        return_distances (bool): if True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters (torch.tensor): the cluster index for each element in data.
        centers (torch.tensor): the integer index of each center.
        distances (torch.tensor): closest distances of each point to any of the cluster centers.
    """
    if n_clusters >= data.shape[0]:
        if return_center_indexes:
            return (torch.arange(data.shape[0], dtype=torch.long),
                    torch.arange(data.shape[0], dtype=torch.long))

        return torch.arange(data.shape[0], dtype=torch.long)

    clusters = torch.full((data.shape[0],), fill_value=-1, dtype=torch.long)
    distances = torch.full((data.shape[0],), fill_value=1e7, dtype=torch.float32)
    centers = torch.zeros(n_clusters, dtype=torch.long)
    for i in range(n_clusters):
        center_idx = torch.argmax(distances)
        centers[i] = center_idx

        broadcasted_data = data[center_idx].unsqueeze(0).expand(data.shape[0], -1)
        new_distances = dist_func(broadcasted_data, data)
        distances = torch.min(distances, new_distances)
        clusters[distances == new_distances] = i
        if verbose:
            print('farthest points max distance : {}'.format(torch.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, centers, distances
        return clusters, centers

    return clusters
