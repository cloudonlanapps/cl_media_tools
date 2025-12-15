import hashlib
from io import BytesIO


def get_md5_hexdigest(bytes_io: BytesIO):
    hash_md5 = hashlib.md5()
    _ = bytes_io.seek(0)

    for chunk in iter(lambda: bytes_io.read(4096), b""):
        hash_md5.update(chunk)

    return hash_md5.hexdigest()


def validate_md5String(bytes_io: BytesIO, md5String: str) -> bool:
    hash_md5 = hashlib.md5()

    _ = bytes_io.seek(0)

    for chunk in iter(lambda: bytes_io.read(4096), b""):
        hash_md5.update(chunk)

    return hash_md5.hexdigest() == md5String
