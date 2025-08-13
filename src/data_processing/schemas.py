"""Defines data schemas for input and final output validation.

This module uses the Pandera library to define strict schemas for data
validation. This ensures data integrity for the initial data ingress and
the final data egress of the pipeline.

The schemas defined here are used to validate:
1.  The raw input data as it is loaded from the source Parquet files.
2.  The final, filtered output data before it is saved by the artifact
    generation script.

Note: There is currently no schema to validate the intermediate
`annotated_faces.parquet` artifact.
"""
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


class OutputSchema(pa.DataFrameModel):
    """
    Schema to validate the final processed DataFrame before saving.

    This schema is intentionally strict (`strict = True`) to enforce a precise
    and consistent contract for the final output dataset. It ensures that only
    the specified columns exist, providing reliability for downstream consumers.

    The `face_bounding_box` column uses the `object` type because it contains
    a list of integers, e.g., `[x1, y1, x2, y2]`.
    """
    image_url: Series[str] = pa.Field(nullable=False)
    source_file: Series[str] = pa.Field(nullable=False)
    face_bounding_box: Series[object] # Stored as list
    face_confidence_score: Series[float]
    eyeglasses_prediction: Series[bool]
    sunglasses_prediction: Series[bool]
    is_target: Series[bool]
    cropped_face_jpeg: Series[bytes]

    class Config:
        # Ensure no extra columns are accidentally added to the final output.
        strict = True
        coerce = True
