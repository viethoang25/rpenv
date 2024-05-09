from __future__ import annotations
from typing import List, Dict, Set, Tuple
from . import *

from collections import Counter
from src.constants import Constants


class SubnetState:
    def __init__(self) -> None:
        self.explorable = False
        self.attackable = False
        self.successful_techniques: List[Technique] = list()


class Subnet(Component):
    def __init__(
        self,
        id: str,
        devices: Dict[str, str | Device],
        firewall: str | Firewall,
        exploration_techniques: List[str | Technique],
    ) -> None:
        super().__init__(id)

        self.devices = {item: None for item in devices}
        self.firewall = firewall
        self.exploration_techniques = exploration_techniques
        self.connected_subnets: Set[Subnet] = set()

        self.state = SubnetState()

    def __repr__(self) -> str:
        return f"""
        Subnet: {self.id}
            Firewall: {self.firewall}
            Connected subnets: {[subnet.id for subnet in self.connected_subnets]}
            Devices: {self.devices.values()}"""

    @staticmethod
    def from_json(data: dict):
        return Subnet(
            id=data["subnet_id"],
            devices=data["devices"],
            firewall=data["firewall"],
            exploration_techniques=data["exploration_techniques"],
        )

    def add_connected_subnet(self, subnet: Subnet) -> None:
        self.connected_subnets.add(subnet)

    def update_objects(
        self,
        devices: Dict[str, Device],
        firewalls: Dict[str, Firewall],
        techniques: Dict[str, Technique],
    ) -> None:
        old_devices = self.devices
        self.devices: Dict[str, Device] = dict()
        for device_id in old_devices.keys():
            self.devices[device_id] = devices[device_id]
            self.devices[device_id].subnet = self

        if self.firewall:
            firewall_id = self.firewall
            self.firewall = firewalls[firewall_id]

        self.exploration_techniques = [
            techniques[id] for id in self.exploration_techniques
        ]

    def update_explorable(self, explorable: bool):
        self.state.explorable = explorable
        if not explorable:
            self.state.attackable = False
        else:
            if len(self.exploration_techniques) == 0:
                self.state.attackable = True
                # Make all connected subnet explorable
                for subnet in self.connected_subnets:
                    if not subnet.state.explorable:  # This to avoid maximum recursion
                        subnet.update_explorable(True)

    def perform_attack(self, technique: Technique):
        # If current subnet is not explorable, we cannot perform exploration techniques on it
        if not self.state.explorable:
            return False, Constants.ATTACK_SUBNET_MSG.UNEXPLOITABLE_SUBNET
        if technique not in self.exploration_techniques:
            return False, Constants.ATTACK_SUBNET_MSG.UNAVAILABLE_TECHNIQUE
        if technique in self.state.successful_techniques:
            return False, Constants.ATTACK_SUBNET_MSG.ALREADY_USED_TECHNIQUE
        if technique.is_too_difficult():
            return False, Constants.ATTACK_SUBNET_MSG.TOO_DIFFICULT

        # At this point the attack is succesful
        self.state.successful_techniques.append(technique)

        # Check if all exploration techniques are successfully attacked
        if Counter(self.state.successful_techniques) == Counter(
            self.exploration_techniques
        ):
            # Allow to attack devices in the subnet
            self.state.attackable = True
            # Make all connected subnet explorable
            for subnet in self.connected_subnets:
                subnet.update_explorable(True)

        return True, Constants.ATTACK_SUBNET_MSG.SUCCESSFUL

    def check_attack_feasibility(self, technique: Technique) -> Tuple[bool, str]:
        # Cannot use technique on devices when the subnet is not attackable
        if not self.state.attackable:
            return False, Constants.ATTACK_DEVICE_MSG.SUBNET_UNATTACKABLE

        # If there is at least one device (in current subnet) is compromised,
        # the inbound and outbound rules will not be applied
        # AKA source device is from current subnet
        for device in self.devices.values():
            if device.state.compromised:
                return True, ""

        # Check inbound and outbound fules of firewalls
        # Check if the technique can be used on this subnet without being filtered by firewall
        if (
            self.firewall is not None
            and technique in self.firewall.inbound_rules.values()
        ):
            return False, Constants.ATTACK_DEVICE_MSG.INBOUND_BLOCKED

        # Check if any compromised devices from connected subnets
        # can by pass their firewall to attack the target device
        source_devices = (
            list()
        )  # Find the list of potential devices that can be the source for attacking
        for subnet in self.connected_subnets:
            # If the used technique is in outbound rules of the source subnet they can be used.
            if (
                subnet.firewall is not None
                and technique in subnet.firewall.outbound_rules.values()
            ):
                continue
            for device in subnet.devices.values():
                if device.state.compromised:
                    source_devices.append(device)
        # If we cannot find any source devices for attacking, the attack will be unsuccessful
        if len(source_devices) == 0:
            return False, Constants.ATTACK_DEVICE_MSG.SOURCE_NOT_FOUND
        else:
            return True, ""
