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

    with Image.open(input_path) as img:
        original_width, original_height = img.size

        # Calculate target dimensions
        if width is None and height is None:
            # Default to 256x256
            w = h = 256
        elif width is None:
            # Height specified, calculate width to maintain aspect ratio
            h = height  # type: ignore[assignment]
            if maintain_aspect_ratio:
                aspect_ratio = original_width / original_height
                w = int(h * aspect_ratio)
            else:
                w = h
        elif height is None:
            # Width specified, calculate height to maintain aspect ratio
            w = width
            if maintain_aspect_ratio:
                aspect_ratio = original_height / original_width
                h = int(w * aspect_ratio)
            else:
                h = w
        else:
            # Both specified
            w = width
            h = height

        # Resize the image
        thumbnail = img.resize((w, h), Image.Resampling.LANCZOS)
        thumbnail.save(output_path)

    return str(output_path)
