import math

import torch
from tabulate import tabulate

from latentfusion import three
from .metrics import concat_camera_metrics


def metrics_table(metrics, tablefmt='github'):
    rows = [
        ['Rotation Dist', format_rotation_err(metrics['rotation_dist'])],
        ['Translation Dist', format_translation_err(metrics['translation_dist'])],
    ]
    if 'add'in metrics:
        rows.append(['ADD', format_point_add(metrics['add'])])
    if 'add_s' in metrics:
        rows.append(['ADD-S', format_point_add(metrics['add_s'])])
    if 'proj2d' in metrics:
        rows.append(['Proj2D', format_point_proj2d(metrics['proj2d'])])
    return tabulate(rows, tablefmt=tablefmt)


def metrics_table_multiple(metrics_list, headers, tablefmt='github'):
    table = [
        [
            headers[i],
            format_rotation_err(m['rotation_dist']),
            format_translation_err(m['translation_dist']),
            format_point_add(m['add']),
            format_point_add(m['add_sym']),
            format_point_add(m['add_s']),
            format_point_proj2d(m['proj2d']),
        ]
        for i, m in enumerate(metrics_list)
    ]
    return tabulate(
        table,
        headers=[
            'Rotation Error',
            'Translation Error',
            'ADD',
            'ADD (sym)',
            'ADD-S',
            'Proj2D',
        ],
        tablefmt=tablefmt)


def format_rotation_err(rotation):
    rotation = rotation / math.pi * 180
    return f"{rotation:.02f}°"


def format_translation_err(translation):
    return f"{translation:.04f} m"


def format_point_add(add):
    return f"{add:.04f} m"


def format_point_proj2d(proj2d):
    return f"{proj2d:.02f} px"


def metrics_summary_table(metrics, tablefmt='github'):
    if isinstance(metrics, list):
        metrics: dict = concat_camera_metrics(metrics)

    return tabulate([
        [
            "Rotation Dist",
            *[format_rotation_err(x) for x in summarize_stats(metrics['rotation_dist'])],
        ],
        [
            "Translation Dist",
            *[format_translation_err(x) for x in summarize_stats(metrics['translation_dist'])],
        ],
        [
            "ADD",
            *[format_point_add(x) for x in summarize_stats(metrics['add'])],
        ],
        [
            "ADD-S",
            *[format_point_add(x) for x in summarize_stats(metrics['add_s'])],
        ],
        [
            "Proj2D",
            *[format_point_proj2d(x) for x in summarize_stats(metrics['proj2d'])],
        ],
    ], tablefmt=tablefmt, headers=["", "Median", "MAD", "Mean", "Std.", "Min", "Max"])


def summarize_stats(stats):
    if isinstance(stats, list):
        stats = torch.tensor(stats)
    return [
        stats.median().item(),
        three.stats.mad(stats).item(),
        stats.mean().item(),
        stats.std().item(),
        stats.min().item(),
        stats.max().item(),
    ]