from typing import List
from src.components import Network, Subnet, Technique, Device
import random
from rich.console import Console
from rich.table import Table


class Action:
    def __init__(self, target: Subnet | Device, technique: Technique):
        self.target = target
        self.technique = technique
        self.type = type(target)

    def __repr__(self):
        return "{:11} -- {:10}".format(self.target.id, self.technique.id)

    def __str__(self):
        return "{:11} -- {:10}".format(self.target.id, self.technique.id)


class ActionSpace:
    def __init__(self, network: Network):
        self.space: List[Action] = list()
        for subnet in network.subnets.values():
            for technique in network.subnet_techniques.values():
                self.space.append(Action(subnet, technique))

        for device in network.devices.values():
            for technique in network.device_techniques.values():
                self.space.append(Action(device, technique))

        self.n = len(self.space)

    def get_action(self, index):
        return self.space[index]

    def sample(self):
        index = random.randrange(self.n)
        return index

    def show(self):
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("ID")
        table.add_column("Target")
        table.add_column("Technique")
        for id, action in enumerate(self.space):
            table.add_row(str(id), action.target.id, action.technique.id)
        Console().print(table)
