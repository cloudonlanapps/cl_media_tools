from typing import Protocol


class UserLike(Protocol):
    id: str | None
