from typing import Dict
from . import Component


class Permission(Component):
    def __init__(self, id: str, name: str, level: int) -> None:
        super().__init__(id)
        self.name = name
        self.level = level

    def __repr__(self) -> str:
        return f"{self.id}"

    @staticmethod
    def from_json(data: Dict):
        return Permission(
            id=data["permission_id"], name=data["name"], level=data["level"]
        )
