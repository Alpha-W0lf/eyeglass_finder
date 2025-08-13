import pandas as pd
import argparse
from pathlib import Path
from loguru import logger
import pyarrow.parquet as pq


def create_sample_file(input_file: Path, output_dir: Path, sample_size: int):
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Reading source file: {input_file.name}...")
    try:
        parquet_file = pq.ParquetFile(input_file)
        first_batch = next(parquet_file.iter_batches(batch_size=sample_size))
        sample_df = first_batch.to_pandas()
    except Exception as e:
        logger.error(f"Failed to read Parquet file {input_file.name}: {e}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"sample_{input_file.name}"
    output_path = output_dir / output_filename

    logger.info(f"Saving {len(sample_df)} rows to {output_path}...")
    sample_df.to_parquet(output_path, index=False)
    logger.info(f"{output_filename} created successfully.")


def main(input_dir: Path, output_dir: Path, sample_size: int):
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return

    input_files = sorted(list(input_dir.glob("*.parquet")))
    if not input_files:
        logger.warning(f"No .parquet files found in {input_dir}")
        return

    logger.info(f"Found {len(input_files)} Parquet files in {input_dir}.")
    for input_file in input_files:
        create_sample_file(input_file, output_dir, sample_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sample files from a directory of Parquet files.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Path to the source directory containing Parquet files.")
    parser.add_argument("--output-dir", type=str, default="data/test", help="Directory to save the sample files.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of rows to include in each sample.")
    args = parser.parse_args()

    main(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir), sample_size=args.sample_size)


