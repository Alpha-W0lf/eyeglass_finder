"""Generates and saves various plots for data analysis and reporting.

This module provides a suite of functions for creating visualizations from the
pipeline's output data. Each function is responsible for generating a specific
plot (e.g., histograms, scatter plots) using Matplotlib and Seaborn, and
saving it to a file.

On import, this module sets a global Seaborn theme (`whitegrid`) that is
applied to all plots it generates.

These visualizations are crucial for:
- Understanding the distribution of the data (e.g., face sizes, model confidences).
- Debugging the pipeline by identifying unexpected patterns.
- Communicating results in the final Markdown report.
"""
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# --- Seaborn Styling ---
sns.set_theme(style="whitegrid")


def plot_confidence_histogram(scores: list[float], output_path: Path):
    if not scores:
        logger.warning("No confidence scores provided. Skipping histogram plot.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=50, kde=False, color='blue')
    plt.title("Face Detection Confidence Scores", fontsize=16)
    plt.xlabel("Confidence Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_face_size_distribution(df: pd.DataFrame, output_path: Path):
    logger.info(f"Plotting face size distribution to {output_path}...")
    if 'face_size' not in df.columns:
        logger.warning("face_size column not found. Skipping plot.")
        return
    def extract_face_area(face_size):
        try:
            if hasattr(face_size, '__len__') and len(face_size) >= 2:
                return float(face_size[0]) * float(face_size[1])
            else:
                return 0
        except (IndexError, TypeError, ValueError):
            return 0
    face_areas = df['face_size'].apply(extract_face_area)
    valid_areas = face_areas[face_areas > 0]
    if len(valid_areas) == 0:
        logger.warning("No valid face areas found. Skipping plot.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_areas, bins=50, kde=True)
    plt.title('Distribution of Detected Face Areas (pixels^2)')
    plt.xlabel('Face Area (pixels^2)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confidence_vs_face_size(df: pd.DataFrame, output_path: Path):
    logger.info(f"Plotting confidence vs. face size to {output_path}...")
    if 'face_score' not in df.columns:
        logger.warning("Required columns for confidence/face size plot not found. Skipping.")
        return
    if 'face_size' in df.columns:
        def extract_face_dimensions(face_size):
            try:
                if hasattr(face_size, '__len__') and len(face_size) >= 2:
                    return float(face_size[0]), float(face_size[1])
                else:
                    return 0, 0
            except (IndexError, TypeError, ValueError):
                return 0, 0
        face_dims = df['face_size'].apply(extract_face_dimensions)
        df['face_width'] = [dims[0] for dims in face_dims]
        df['face_height'] = [dims[1] for dims in face_dims]
        df['face_area'] = df['face_width'] * df['face_height']
        valid_data = df[df['face_area'] > 0]
        if len(valid_data) == 0:
            logger.warning("No valid face areas found. Cannot create scatter plot. Skipping plot.")
            return
        df = valid_data
    else:
        logger.warning("face_size column not found. Cannot calculate face area. Skipping plot.")
        return
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='face_area', y='face_score', alpha=0.5)
    plt.title('Face Detection Confidence vs. Face Area')
    plt.xlabel('Face Area (pixels^2)')
    plt.ylabel('Detection Confidence')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_image_mode_distribution(df: pd.DataFrame, output_path: Path):
    logger.info(f"Plotting image mode distribution to {output_path}...")
    if 'original_image_mode' not in df.columns:
        logger.warning("original_image_mode column not found. Skipping plot.")
        return
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['original_image_mode'], order=df['original_image_mode'].value_counts().index)
    plt.title('Distribution of Original Image Modes')
    plt.xlabel('Count')
    plt.ylabel('Image Mode')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_resource_utilization(data: List[Dict[str, Any]], output_path: Path):
    if not data:
        logger.warning("No resource utilization data provided. Skipping plot.")
        return
    logger.info(f"Plotting resource utilization to {output_path}...")
    import pandas as pd
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('System Resource Utilization During Processing', fontsize=16)
    axes[0].plot(df['timestamp'], df['cpu_percent'], color='tab:blue', label='CPU Usage (%)')
    axes[0].set_ylabel('CPU Utilization (%)')
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc='upper left')
    axes[0].grid(True)
    axes[1].plot(df['timestamp'], df['memory_mb'], color='tab:red', label='Memory Usage (MB)')
    axes[1].set_ylabel('Memory Usage (MB)')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)
    axes[2].plot(df['timestamp'], df['disk_read_mbps'], color='tab:green', label='Disk Read (MB/s)')
    axes[2].plot(df['timestamp'], df['disk_write_mbps'], color='tab:orange', label='Disk Write (MB/s)')
    axes[2].set_ylabel('Disk I/O (MB/s)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='upper left')
    axes[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path)
    plt.close()


def plot_worker_performance_histogram(worker_times: List[float], output_path: Path):
    if not worker_times:
        logger.warning("No worker timing data provided. Skipping histogram plot.")
        return
    logger.info(f"Plotting worker performance histogram to {output_path}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(worker_times, bins=30, kde=True, color='purple')
    plt.title("Distribution of Worker Chunk Processing Times", fontsize=16)
    plt.xlabel("Processing Time per Chunk (seconds)", fontsize=12)
    plt.ylabel("Frequency (Number of Chunks)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_face_count_distribution(face_count_distribution: dict, output_path: Path):
    if not face_count_distribution:
        logger.warning("No face count distribution data provided. Skipping plot.")
        return
    logger.info(f"Plotting face count distribution to {output_path}...")
    face_count_int_dict = {}
    for key, value in face_count_distribution.items():
        try:
            face_count_int_dict[int(key)] = value
        except (ValueError, TypeError):
            logger.warning(f"Skipping invalid face count key: {key}")
            continue
    face_counts = list(face_count_int_dict.keys())
    image_counts = list(face_count_int_dict.values())
    sorted_data = sorted(zip(face_counts, image_counts))
    face_counts, image_counts = zip(*sorted_data)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(face_counts, image_counts, color='steelblue', alpha=0.7, edgecolor='black')
    for bar, count in zip(bars, image_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(image_counts)*0.01, f'{count}', ha='center', va='bottom', fontsize=10)
    plt.title('Distribution of Faces per Image\n(Diagnostic for High Face Detection Count)', fontsize=16, pad=20)
    plt.xlabel('Number of Faces Detected per Image', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(face_counts)
    plt.grid(axis='y', alpha=0.3)
    total_images = sum(image_counts)
    total_faces = sum(fc * ic for fc, ic in zip(face_counts, image_counts))
    avg_faces_per_image = total_faces / total_images if total_images > 0 else 0
    max_faces = max(face_counts) if face_counts else 0
    stats_text = f'Total Images: {total_images:,}\n'
    stats_text += f'Total Faces: {total_faces:,}\n'
    stats_text += f'Avg Faces/Image: {avg_faces_per_image:.2f}\n'
    stats_text += f'Max Faces in Image: {max_faces}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
