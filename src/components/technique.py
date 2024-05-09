from .component import Component
import random


class Technique(Component):
    def __init__(self, id: str, duration: float, vulnerability: float) -> None:
        super().__init__(id)
        self.duration = duration
        self.vulnerability = vulnerability

    def __repr__(self) -> str:
        return f"{self.id}"

    @staticmethod
    def from_json(data: dict):
        return Technique(
            id=data["technique_id"],
            duration=data["duration"],
            vulnerability=data["vulnerability"],
        )

    def is_too_difficult(self):
        return random.random() > self.vulnerability
