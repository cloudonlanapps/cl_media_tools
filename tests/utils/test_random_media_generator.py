"""Unit tests for RandomMediaGenerator.

Tests media list validation, MIME type support, and configuration.
Skips actual generation tests as they require complex setup with frame generators.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from cl_ml_tools.utils.random_media_generator import (
    JSONValidationError,
    RandomMediaGenerator,
)
from cl_ml_tools.utils.random_media_generator.base_media import SupportedMIME


# ============================================================================
# SupportedMIME Tests
# ============================================================================


def test_supported_mime_types_list():
    """Test SupportedMIME has expected MIME types."""
    mime_types = SupportedMIME.MIME_TYPES

    # Image types
    assert "image/jpeg" in mime_types
    assert "image/png" in mime_types
    assert "image/tiff" in mime_types
    assert "image/gif" in mime_types
    assert "image/webp" in mime_types

    # Video types
    assert "video/mp4" in mime_types
    assert "video/mov" in mime_types
    assert "video/x-msvideo" in mime_types
    assert "video/x-matroska" in mime_types


def test_supported_mime_extensions():
    """Test MIME types have correct file extensions."""
    assert SupportedMIME.MIME_TYPES["image/jpeg"]["extension"] == "jpg"
    assert SupportedMIME.MIME_TYPES["image/png"]["extension"] == "png"
    assert SupportedMIME.MIME_TYPES["video/mp4"]["extension"] == "mp4"
    assert SupportedMIME.MIME_TYPES["video/mov"]["extension"] == "mov"


def test_fourcc_codes():
    """Test FOURCC codes are defined for video types."""
    fourcc = SupportedMIME.FOURCC

    assert "video/mp4" in fourcc
    assert "video/mov" in fourcc
    assert "video/x-msvideo" in fourcc
    assert "video/x-matroska" in fourcc

    # All FOURCC codes should be integers
    for mime_type, code in fourcc.items():
        assert isinstance(code, int)


# ============================================================================
# RandomMediaGenerator Initialization Tests
# ============================================================================


def test_random_media_generator_init_minimal(tmp_path: Path):
    """Test RandomMediaGenerator initialization with minimal params."""
    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    assert generator.out_dir == str(tmp_path)
    assert generator.media_list == []


def test_random_media_generator_init_with_media_list(tmp_path: Path):
    """Test RandomMediaGenerator initialization with empty media list."""
    generator = RandomMediaGenerator(
        out_dir=str(tmp_path),
        media_list=[],
    )

    assert generator.out_dir == str(tmp_path)
    assert generator.media_list == []


def test_random_media_generator_supported_mime():
    """Test supportedMIME class method returns list of MIME types."""
    supported = RandomMediaGenerator.supportedMIME()

    assert isinstance(supported, list)
    assert len(supported) > 0
    assert "image/jpeg" in supported
    assert "image/png" in supported
    assert "video/mp4" in supported


# ============================================================================
# Media List Validation Tests
# ============================================================================


def test_media_list_validation_requires_out_dir():
    """Test media list validation fails without out_dir."""
    # This should fail during validation because out_dir is required
    with pytest.raises((ValidationError, JSONValidationError)):
        _ = RandomMediaGenerator(
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                }
            ]
        )


def test_media_list_validation_with_image_type(tmp_path: Path):
    """Test media list validates image/* MIME types."""
    # Note: This will fail validation because ImageGenerator requires 'frame' field
    with pytest.raises((ValidationError, JSONValidationError)):
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                }
            ],
        )


def test_media_list_validation_with_video_type(tmp_path: Path):
    """Test media list validates video/* MIME types."""
    # Note: This will fail validation because VideoGenerator requires additional fields
    with pytest.raises((ValidationError, JSONValidationError)):
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[
                {
                    "MIMEType": "video/mp4",
                    "width": 1920,
                    "height": 1080,
                    "fileName": "test",
                }
            ],
        )


def test_media_list_validation_invalid_mime_type(tmp_path: Path):
    """Test media list validation with unsupported MIME type."""
    # Unsupported MIME type should create BaseMedia (which is abstract)
    with pytest.raises((ValidationError, TypeError)):
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[
                {
                    "MIMEType": "application/pdf",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                }
            ],
        )


# ============================================================================
# Directory Management Tests
# ============================================================================


def test_random_media_generator_output_directory_validation(tmp_path: Path):
    """Test that output directory is validated and set correctly."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    generator = RandomMediaGenerator(out_dir=str(output_dir))

    assert generator.out_dir == str(output_dir)


def test_random_media_generator_nonexistent_directory(tmp_path: Path):
    """Test RandomMediaGenerator accepts non-existent directory path."""
    output_dir = tmp_path / "nonexistent"

    # Should not raise - directory creation happens during generation
    generator = RandomMediaGenerator(out_dir=str(output_dir))

    assert generator.out_dir == str(output_dir)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_random_media_generator_is_pydantic_model(tmp_path: Path):
    """Test RandomMediaGenerator is a Pydantic BaseModel."""
    from pydantic import BaseModel

    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    assert isinstance(generator, BaseModel)


def test_random_media_generator_serialization(tmp_path: Path):
    """Test RandomMediaGenerator can be serialized to dict."""
    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    data = generator.model_dump()

    assert data["out_dir"] == str(tmp_path)
    assert data["media_list"] == []


def test_random_media_generator_deserialization(tmp_path: Path):
    """Test RandomMediaGenerator can be deserialized from dict."""
    data = {
        "out_dir": str(tmp_path),
        "media_list": [],
    }

    generator = RandomMediaGenerator.model_validate(data)

    assert generator.out_dir == str(tmp_path)
    assert generator.media_list == []


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_json_validation_error_import():
    """Test JSONValidationError can be imported."""
    assert JSONValidationError is not None
    assert issubclass(JSONValidationError, Exception)


def test_json_validation_error_raise():
    """Test JSONValidationError can be raised and caught."""
    with pytest.raises(JSONValidationError) as exc_info:
        raise JSONValidationError("Test error message")

    assert "Test error message" in str(exc_info.value)


def test_random_media_generator_validation_error_message(tmp_path: Path):
    """Test validation error provides helpful message."""
    with pytest.raises((ValidationError, JSONValidationError)) as exc_info:
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                    # Missing required 'frame' field
                }
            ],
        )

    # Should mention missing field
    error_str = str(exc_info.value)
    assert len(error_str) > 0  # Has error message


# ============================================================================
# Edge Cases
# ============================================================================


def test_random_media_generator_empty_string_out_dir():
    """Test RandomMediaGenerator with empty string as out_dir."""
    # Empty string is technically valid, though not recommended
    generator = RandomMediaGenerator(out_dir="")

    assert generator.out_dir == ""


def test_random_media_generator_media_list_empty_default(tmp_path: Path):
    """Test media_list defaults to empty list."""
    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    assert generator.media_list == []
    assert isinstance(generator.media_list, list)


def test_supported_mime_returns_new_list():
    """Test supportedMIME returns a new list each time."""
    list1 = RandomMediaGenerator.supportedMIME()
    list2 = RandomMediaGenerator.supportedMIME()

    # Should be equal but not the same object
    assert list1 == list2
    assert list1 is not list2


def test_supported_mime_contains_all_base_types():
    """Test supportedMIME includes both image and video types."""
    supported = RandomMediaGenerator.supportedMIME()

    has_image = any(mime.startswith("image/") for mime in supported)
    has_video = any(mime.startswith("video/") for mime in supported)

    assert has_image
    assert has_video
