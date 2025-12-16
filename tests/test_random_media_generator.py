"""Comprehensive test suite for random media generator utility."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from _pytest.capture import CaptureFixture

from cl_ml_tools.utils.random_media_generator import (
    JSONValidationError,
    RandomMediaGenerator,
)
from cl_ml_tools.utils.random_media_generator.base_media import BaseMedia, SupportedMIME
from cl_ml_tools.utils.random_media_generator.exif_metadata import ExifMetadata
from cl_ml_tools.utils.random_media_generator.frame_generator import FrameGenerator
from cl_ml_tools.utils.random_media_generator.image_generator import ImageGenerator
from cl_ml_tools.utils.random_media_generator.scene_generator import SceneGenerator
from cl_ml_tools.utils.random_media_generator.video_generator import VideoGenerator

# ============================================================================
# Test Class 1: JSONValidationError
# ============================================================================


class TestJSONValidationError:
    def test_error_creation(self) -> None:
        error = JSONValidationError("Test error message")
        assert error.message == "Test error message"

    def test_error_default_message(self) -> None:
        error = JSONValidationError()
        assert error.message == "An unknown custom error occurred."

    def test_error_str_representation(self) -> None:
        error = JSONValidationError("Test error")
        assert str(error) == "CustomError: Test error"

    def test_error_is_exception(self) -> None:
        assert isinstance(JSONValidationError("x"), Exception)


# ============================================================================
# Test Class 2: SupportedMIME
# ============================================================================


class TestSupportedMIME:
    def test_supported_image_mime_types(self) -> None:
        assert "image/jpeg" in SupportedMIME.MIME_TYPES
        assert "image/png" in SupportedMIME.MIME_TYPES

    def test_supported_video_mime_types(self) -> None:
        assert "video/mp4" in SupportedMIME.MIME_TYPES

    def test_mime_type_extensions(self) -> None:
        assert SupportedMIME.MIME_TYPES["image/jpeg"]["extension"] == "jpg"
        assert SupportedMIME.MIME_TYPES["video/mp4"]["extension"] == "mp4"


# ============================================================================
# Test Class 3: ExifMetadata
# ============================================================================


class TestExifMetadata:
    def test_exif_metadata_creation(self) -> None:
        metadata = ExifMetadata(MIMEType="image/jpeg")
        assert metadata.CreateDate is None
        assert metadata.UserComments == []

    def test_update_create_date_image(self) -> None:
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        metadata = ExifMetadata(MIMEType="image/jpeg", CreateDate=dt)
        metadata.updateCreateDate()
        assert metadata.has_metadata

    @patch("subprocess.run")
    def test_write_metadata_success(self, mock_run: Mock) -> None:
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        metadata = ExifMetadata(MIMEType="image/jpeg", CreateDate=datetime.now(tz=timezone.utc))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
        try:
            with patch("os.path.exists", return_value=True):
                metadata.write(path)
            mock_run.assert_called_once()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_write_empty_metadata(self, capsys: CaptureFixture[str]) -> None:
        metadata = ExifMetadata(MIMEType="image/jpeg")
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            metadata.write(tmp.name)
        captured = capsys.readouterr()
        assert "Metadata is empty" in captured.out


# ============================================================================
# Test Class 4: FrameGenerator
# ============================================================================


class TestFrameGenerator:
    def test_frame_generator_creation(self) -> None:
        fg = FrameGenerator()
        assert fg.shapes == []

    def test_invalid_color(self) -> None:
        with pytest.raises(JSONValidationError):
            _ = FrameGenerator(background_color=[1, 2])  # pyright: ignore[reportArgumentType]

    def test_create_base_frame(self) -> None:
        frame = FrameGenerator.create_base_frame(10, 10, (255, 0, 0))
        assert np.all(frame[:, :, 0] == 255)  # pyright: ignore[reportAny]


# ============================================================================
# Test Class 5: SceneGenerator
# ============================================================================


class TestSceneGenerator:
    def test_scene_creation(self) -> None:
        scene = SceneGenerator(duration_seconds=1)
        assert scene.num_frames(30) == 30

    def test_zero_duration(self) -> None:
        scene = SceneGenerator(duration_seconds=None)
        writer = Mock()
        scene.render_to(out=writer, fps=30, width=10, height=10)
        writer.write.assert_not_called()  # pyright: ignore[reportAny]


# ============================================================================
# Test Class 6: ImageGenerator
# ============================================================================


class TestImageGenerator:
    def test_image_generator_frame_dict(self) -> None:
        img = ImageGenerator(
            out_dir="/tmp",
            MIMEType="image/jpeg",
            width=10,
            height=10,
            fileName="x",
            frame={"background_color": [255, 0, 0]},  # pyright: ignore[reportArgumentType]
        )
        assert isinstance(img.frame, FrameGenerator)

    def test_missing_frame(self) -> None:
        with pytest.raises(JSONValidationError):
            _ = ImageGenerator(
                out_dir="/tmp",
                MIMEType="image/jpeg",
                width=10,
                height=10,
                fileName="x",
            )


# ============================================================================
# Test Class 7: VideoGenerator
# ============================================================================


class TestVideoGenerator:
    def test_video_generator_scenes_dict(self) -> None:
        vid = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=10,
            height=10,
            fileName="x",
            scenes=[{"duration_seconds": 1}],  # pyright: ignore[reportArgumentType]
            fps=30,
        )
        assert isinstance(vid.scenes[0], SceneGenerator)

    def test_invalid_fps(self) -> None:
        with pytest.raises(ValueError):
            _ = VideoGenerator(
                out_dir="/tmp",
                MIMEType="video/mp4",
                width=10,
                height=10,
                fileName="x",
                scenes=[],
                fps=0,
            )


# ============================================================================
# Test Class 8: BaseMedia
# ============================================================================


class TestBaseMedia:
    def test_media_properties(self) -> None:
        media: BaseMedia = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=10,
            height=10,
            fileName="x",
            scenes=[SceneGenerator(duration_seconds=1)],
        )
        assert media.fileextension == ".mp4"

    def test_timestamp_conversion(self) -> None:
        media = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=10,
            height=10,
            fileName="x",
            scenes=[SceneGenerator(duration_seconds=1)],
            CreateDate=1704067200000,  # pyright: ignore[reportArgumentType]
        )
        assert isinstance(media.CreateDate, datetime)


# ============================================================================
# Test Class 9: RandomMediaGenerator
# ============================================================================


class TestRandomMediaGenerator:
    def test_media_list_dicts(self) -> None:
        gen = RandomMediaGenerator(
            out_dir="/tmp",
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 10,
                    "height": 10,
                    "fileName": "x",
                    "frame": {"background_color": [255, 0, 0]},
                }
            ],  # pyright: ignore[reportArgumentType]
        )
        assert isinstance(gen.media_list[0], ImageGenerator)

    def test_missing_out_dir(self) -> None:
        with pytest.raises(JSONValidationError):
            _ = RandomMediaGenerator(
                out_dir=None,  # pyright: ignore[reportArgumentType]
                media_list=[{"MIMEType": "image/jpeg"}],  # pyright: ignore[reportArgumentType]
            )


# ============================================================================
# Test Class 10: Integration
# ============================================================================


class TestIntegration:
    @patch("cv2.imwrite")
    @patch("os.rename")
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.run")
    def test_generate_image(
        self,
        mock_run: Mock,
        mock_exists: Mock,  # pyright: ignore[reportUnusedParameter]
        mock_rename: Mock,  # pyright: ignore[reportUnusedParameter]
        mock_imwrite: Mock,
    ) -> None:
        mock_imwrite.return_value = True
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        gen = RandomMediaGenerator(
            out_dir="/tmp",
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 10,
                    "height": 10,
                    "fileName": "x",
                    "frame": {"background_color": [255, 255, 255]},
                }
            ],  # pyright: ignore[reportArgumentType]
        )

        _ = gen.media_list[0].generate()
        mock_imwrite.assert_called_once()
