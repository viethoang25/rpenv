from typing import List, Dict
from .component import Component
from .technique import Technique


class Firewall(Component):
    def __init__(
        self,
        id: str,
        inbound_rules: List[str] = list(),
        outbound_rules: List[str] = list(),
    ) -> None:
        super().__init__(id)
        self.inbound_rules = {item: None for item in inbound_rules}
        self.outbound_rules = {item: None for item in outbound_rules}

    def __repr__(self) -> str:
        return f"{self.id} -- {self.inbound_rules} -- {self.outbound_rules}"

    @staticmethod
    def from_json(data: Dict):
        return Firewall(
            id=data["firewall_id"],
            inbound_rules=data["inbound_rules"],
            outbound_rules=data["outbound_rules"],
        )

    def update_objects(self, techniques: Dict[str, Technique]) -> None:
        for id, technique in techniques.items():
            if id in self.inbound_rules:
                self.inbound_rules[id] = technique
            if id in self.outbound_rules:
                self.outbound_rules[id] = technique
