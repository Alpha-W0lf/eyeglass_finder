"""Pandera schema that validates each raw input chunk as it is loaded from the source Parquet files."""
import pandera as pa
from pandera.typing import Series

class InputSchema(pa.DataFrameModel):
    """
    Schema to validate the raw input data chunks.

    This schema is intentionally non-strict (`strict = False`) to allow for
    flexibility in the input data. It only validates that the essential columns
    ('image_url', 'image') are present. It uses a broad `object` type for the
    'image' column, but the downstream expectation is that it contains raw
    image data as bytes.
    """
    image_url: Series[str] = pa.Field(nullable=False)
    image: Series[object] = pa.Field(nullable=True)  # Made nullable to handle complex structures

    class Config:
        # Allow other columns to exist in the input data without raising an error.
        strict = False
        coerce = True
        # Add additional flexibility for complex data types
        add_missing_columns = False
