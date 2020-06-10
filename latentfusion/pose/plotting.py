import numpy as np
import seaborn as sns
import sklearn.metrics


def plot_add(ax, metrics, metric_type, object_name, label):
    thresholds = np.linspace(0.0, 0.10, 1000)
    x_range = thresholds.max() - thresholds.min()
    results = np.array([m[metric_type] for m in metrics])
    accuracies = [(results <= t).sum() / len(results) for t in thresholds]
    auc = sklearn.metrics.auc(thresholds, accuracies) / x_range

    ax = sns.lineplot(thresholds, accuracies, label=f"{label} ({auc:.04f})", ax=ax)
    ax.set_title(f"{object_name} {metric_type}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Threshold (m)')

    return auc


def plot_add_s(ax, metrics, object_name, label):
    thresholds = np.linspace(0.0, 0.1, 1000)
    x_range = thresholds.max() - thresholds.min()
    results = np.array([m['add_s'] for m in metrics])
    accuracies = [(results <= t).sum() / len(results) for t in thresholds]
    auc = sklearn.metrics.auc(thresholds, accuracies) / x_range
    ax = sns.lineplot(thresholds, accuracies, label=f"{label} ({auc:.04f})", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{object_name} ADD-S")
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Threshold (m)')

    return auc


def plot_proj2d(ax, metrics, object_name, label):
    thresholds = np.linspace(0, 40.0, 1000)
    x_range = thresholds.max() - thresholds.min()
    results = np.array([m['proj2d'] for m in metrics])
    accuracies = [(results <= t).sum() / len(results) for t in thresholds]
    auc = sklearn.metrics.auc(thresholds, accuracies) / x_range
    ax = sns.lineplot(thresholds, accuracies, label=f"{label} ({auc:.04f})", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{object_name} Proj. 2D")
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Threshold (px)')

    return auc
