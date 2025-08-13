"""Handles the loading and validation of data from the source Parquet files.

This module provides a generator function that reads Parquet files in chunks,
making it suitable for large datasets. For each chunk, it performs two key
operations before yielding a pandas DataFrame:

1.  It validates the data against the `InputSchema`. The validation is configured
    to be "lazy," meaning it collects all validation failures across the chunk
    before raising an error, which is useful for comprehensive logging.
2.  It enriches the data by adding a `source_file` column for traceability.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from loguru import logger
from typing import Iterator, Dict, Any
from pandera.errors import SchemaError

from .schemas import InputSchema

def stream_data_generator(
    data_files: list[Path], chunk_size: int
) -> Iterator[pd.DataFrame]:
    """
    Creates a generator to stream data from Parquet files in chunks.

    This function iterates through a provided list of Parquet files,
    opens them one by one using pyarrow, and yields their content in
    manageable pandas DataFrame chunks. It is designed for memory efficiency
    and resilience, as it will log errors and skip any individual chunks or
    entire files that fail validation or cause read errors, preventing a
    single bad record from stopping the entire pipeline.

    Args:
        data_files (list[Path]): A list of paths to the Parquet files.
        chunk_size (int): The number of rows for each chunk (batch) to be
                          yielded by the generator.

    Yields:
        Iterator[pd.DataFrame]: A generator that yields pandas DataFrames,
                                each representing a chunk of the data.
    """
    logger.info(f"PyArrow version detected: {pa.__version__}")
    
    if not data_files:
        logger.warning(f"No data files provided to the generator.")
        return

    logger.info(f"Generator will process {len(data_files)} data files.")

    for file in data_files:
        logger.info(f"Streaming data from {file.name}...")
        try:
            # First, try to examine the file structure
            logger.debug(f"Reading parquet file structure for {file.name}")
            parquet_file = pq.ParquetFile(file)
            
            # Define the columns to be loaded based on the input schema
            required_columns = list(InputSchema.to_schema().columns.keys())
            logger.info(f"Loader will only read required columns: {required_columns}")
            
            # Log schema information
            logger.debug(f"Parquet schema for {file.name}: {parquet_file.schema}")
            logger.debug(f"Parquet metadata for {file.name}: {parquet_file.metadata}")
            
            # Try different approaches to read the data
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size, columns=required_columns)):
                try:
                    logger.debug(f"Processing batch {batch_idx} from {file.name}")
                    logger.debug(f"Batch schema: {batch.schema}")
                    
                    # Try multiple conversion strategies
                    chunk_df = None
                    
                    # Strategy 1: Standard conversion
                    try:
                        logger.debug(f"Attempting standard batch.to_pandas() for batch {batch_idx}")
                        chunk_df = batch.to_pandas()
                        logger.debug(f"Standard conversion successful for batch {batch_idx}")
                    except Exception as e1:
                        logger.warning(f"Standard conversion failed for batch {batch_idx}: {e1}")
                        
                        # Strategy 2: Conversion with ignore_metadata
                        try:
                            logger.debug(f"Attempting batch.to_pandas(ignore_metadata=True) for batch {batch_idx}")
                            chunk_df = batch.to_pandas(ignore_metadata=True)
                            logger.debug(f"ignore_metadata conversion successful for batch {batch_idx}")
                        except Exception as e2:
                            logger.warning(f"ignore_metadata conversion failed for batch {batch_idx}: {e2}")
                            
                            # Strategy 3: Convert to table first, then remove metadata
                            try:
                                logger.debug(f"Attempting table conversion with metadata removal for batch {batch_idx}")
                                table = batch.to_table()
                                # Remove all metadata
                                clean_table = table.replace_schema_metadata({})
                                chunk_df = clean_table.to_pandas()
                                logger.debug(f"Clean table conversion successful for batch {batch_idx}")
                            except Exception as e3:
                                logger.error(f"All conversion strategies failed for batch {batch_idx}")
                                logger.error(f"Strategy 1 error: {e1}")
                                logger.error(f"Strategy 2 error: {e2}")
                                logger.error(f"Strategy 3 error: {e3}")
                                continue
                    
                    if chunk_df is not None:
                        chunk_df["source_file"] = file.name

                        try:
                            # Validate the input schema before yielding
                            InputSchema.validate(chunk_df, lazy=True)
                            logger.info(f"Generator yielding a chunk of {len(chunk_df)} rows from {file.name}.")
                            yield chunk_df
                        except SchemaError as err:
                            logger.warning(
                                f"Skipping chunk from {file.name} due to schema validation error. "
                                f"Errors: {err.failure_cases}"
                            )
                            continue
                    else:
                        logger.error(f"Failed to convert batch {batch_idx} from {file.name}")

                except Exception as batch_error:
                    logger.error(f"Failed to process batch {batch_idx} from {file.name}: {batch_error}")
                    continue

        except Exception as e:
            logger.error(f"Failed to stream data file {file.name}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            continue
