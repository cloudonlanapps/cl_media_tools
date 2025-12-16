"""Pure image thumbnail computation logic (single file)."""

from pathlib import Path

from PIL import Image


def image_thumbnail(
    *,
    input_path: str | Path,
    output_path: str | Path,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
) -> str:
    """
    Thumbnail a single image and write output.

    Framework-agnostic, single-responsibility function.

    Args:
        input_path: Path to input image
        output_path: Path to output image
        w: Target w
        h: Target h
        maintain_aspect_ratio: Preserve aspect ratio if True

    Returns:
        Output file path as string

    Raises:
        FileNotFoundError: If input image does not exist
        OSError: If Pillow fails to read/write the image
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Default to 256 if both are None
    size: int = width or height or 256
    w: int = width if width is not None else size
    h: int = height if height is not None else size

    with Image.open(input_path) as img:
        if maintain_aspect_ratio:
            img.thumbnail((w, h), Image.Resampling.LANCZOS)
            thumbnail = img
        else:
            thumbnail = img.resize(
                (w, h),
                Image.Resampling.LANCZOS,
            )

        thumbnail.save(output_path)

    return str(output_path)
