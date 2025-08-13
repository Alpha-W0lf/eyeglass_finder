"""Provides utility functions for the data processing stage.

This module contains helper functions that support the data processing workflow.
It includes functions for:
1.  Calculating the total number of data chunks for progress tracking.
2.  Safely opening image data from binary buffers, which is crucial for
    handling potentially corrupted or unusually formatted images (e.g.,
    palette-based PNGs) without crashing the pipeline.
"""
import pyarrow.parquet as pq
from pathlib import Path
from loguru import logger
from typing import List, Tuple
from PIL import Image
from io import BytesIO
import cairosvg

# Set a target size for SVG to PNG conversion to prevent memory issues.
# This ensures that even large vector graphics are rasterized to a manageable size.
SVG_TARGET_WIDTH = 2048

def get_total_chunks(data_files: List[Path], chunk_size: int) -> int:
    """
    Calculates the total number of chunks across all provided Parquet files.

    This function uses ceiling division to ensure that the final, potentially
    smaller chunk is included in the count. It is also resilient: if it
    encounters an error reading a file's metadata, it logs the error and
    skips that file, continuing with the rest of the list.

    Args:
        data_files (List[Path]): A list of paths to the Parquet files.
        chunk_size (int): The number of rows per chunk.

    Returns:
        int: The total number of chunks.
    """
    total_chunks = 0
    for file in data_files:
        try:
            parquet_file = pq.ParquetFile(file)
            num_rows = parquet_file.metadata.num_rows
            total_chunks += (num_rows + chunk_size - 1) // chunk_size
        except Exception as e:
            logger.error(f"Could not read metadata from {file.name}: {e}")
    return total_chunks


def safe_image_open(image_bytes: bytes) -> Tuple["Image.Image", str]:
    """
    Safely opens an image from bytes, handling potential decoding errors
    and converting problematic formats like SVG on the fly.

    This function first checks if the image is an SVG. If so, it uses
    `cairosvg` to convert it to a reasonably-sized PNG in memory before
    opening it with Pillow. This is a critical step to prevent memory
    exhaustion from large vector graphics. For all other formats, it uses
    Pillow directly.

    Args:
        image_bytes (bytes): The raw bytes of the image file.

    Returns:
        A tuple containing the Pillow Image object and its original mode.
        Returns (None, None) if the image cannot be opened.
    """
    try:
        # --- Check for SVG ---
        # Look for "<svg" or "<?xml" in the first 100 bytes.
        # This is a good heuristic for identifying SVG files.
        if image_bytes.strip().startswith(b"<svg") or (
            b"<?xml" in image_bytes[:100] and b"<svg" in image_bytes[:400]
        ):
            # Convert SVG to PNG in memory
            png_bytes = cairosvg.svg2png(
                bytestring=image_bytes, output_width=SVG_TARGET_WIDTH
            )
            image_bytes = png_bytes # Replace original bytes with new PNG bytes

        # Open the image (either original or the new PNG) with Pillow
        image = Image.open(BytesIO(image_bytes))
        return image, image.mode
    except Exception:
        # If any error occurs (e.g., still a bad format, corrupted file),
        # return None so the calling function can handle it gracefully.
        return None, None
