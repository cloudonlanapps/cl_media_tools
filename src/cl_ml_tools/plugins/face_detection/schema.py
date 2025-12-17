"""Face detection parameters and output schemas."""

from pydantic import BaseModel, Field

from ...common.schema_job import BaseJobParams, TaskOutput


class FaceDetectionParams(BaseJobParams):
    """Parameters for face detection task."""

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for face detections (0.0-1.0)",
    )
    nms_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Non-maximum suppression threshold for overlapping boxes (0.0-1.0)",
    )


class BoundingBox(BaseModel):
    """Bounding box for a detected face.

    Coordinates are normalized to [0.0, 1.0] range relative to image dimensions.
    """

    x1: float = Field(..., ge=0.0, le=1.0, description="Left coordinate (normalized)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Top coordinate (normalized)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Right coordinate (normalized)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Bottom coordinate (normalized)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")

    def to_absolute(self, image_width: int, image_height: int) -> dict[str, int]:
        """Convert normalized coordinates to absolute pixel coordinates.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Dict with absolute pixel coordinates (x1, y1, x2, y2)
        """
        return {
            "x1": int(self.x1 * image_width),
            "y1": int(self.y1 * image_height),
            "x2": int(self.x2 * image_width),
            "y2": int(self.y2 * image_height),
        }


class EmbeddingOutput(TaskOutput):
    faces: list[BoundingBox] = Field(
        default_factory=list, description="List of detected faces with bounding boxes"
    )
    num_faces: int = Field(..., description="Total number of faces detected")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
