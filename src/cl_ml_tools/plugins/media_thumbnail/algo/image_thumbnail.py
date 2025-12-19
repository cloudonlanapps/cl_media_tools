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

        # Defaults
        DEFAULT_SIZE = 256

        # Determine target dimensions
        if width is None and height is None:
            w = h = DEFAULT_SIZE
        elif width is None:
            assert height is not None
            h = height
            if maintain_aspect_ratio:
                w = int(h * (original_width / original_height))
            else:
                w = h
        elif height is None:
            assert width is not None
            w = width
            if maintain_aspect_ratio:
                h = int(w * (original_height / original_width))
            else:
                h = w
        else:
            # Both width and height specified
            assert width is not None
            assert height is not None
            if maintain_aspect_ratio:
                # Fit within bounds while maintaining aspect ratio
                aspect_ratio = original_width / original_height
                if width / height > aspect_ratio:
                    # Height is the limiting factor
                    h = height
                    w = int(h * aspect_ratio)
                else:
                    # Width is the limiting factor
                    w = width
                    h = int(w / aspect_ratio)
            else:
                w = width
                h = height

        # Prevent upscaling - cap at original size
        w = min(w, original_width)
        h = min(h, original_height)

        # Resize the image
        thumbnail = img.resize((w, h), Image.Resampling.LANCZOS)
        thumbnail.save(output_path)

    return str(output_path)
