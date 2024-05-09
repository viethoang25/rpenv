from typing import List, Any, Dict, Tuple
from . import *
from src.constants import Constants


class DeviceState:
    def __init__(self) -> None:
        self.compromised = False
        self.successful_techniques: List[
            Technique
        ] = list()  # Store all techniques that successfully used for this device


class Device(Component):
    def __init__(
        self,
        id: str,
        name: str,
        os: str,
        type: str,
        gained_permissions: str | Permission,
        required_permissions: Dict[str, List[str | Permission]],
        compromise_sequences: List[Any],
        sensitive: bool = False,
    ) -> None:
        # Basic information
        super().__init__(id)
        self.subnet: Subnet = None  # Cross-reference for the subnet that device are in
        self.name = name
        self.os = os
        self.type = type
        self.sensitive = sensitive
        self.gained_permissions = gained_permissions
        self.required_permissions = required_permissions
        self.compromise_sequences = compromise_sequences

        self.state = DeviceState()

    def __repr__(self) -> str:
        return f"{self.id} -- {self.type} -- {self.required_permissions} -- {self.compromise_sequences}"

    @staticmethod
    def from_json(data: Dict):
        """Get device object from json data

        Args:
            data (dict): json data

        Returns:
            Device: device object
        """
        sensitive = False
        name = None
        os = None
        if "sensitive" in data:
            sensitive = data["sensitive"]
        if "name" in data:
            name = data["name"]
        if "OS" in data:
            os = data["OS"]
        return Device(
            id=data["device_id"],
            name=name,
            type=data["type"],
            os=os,
            sensitive=sensitive,
            gained_permissions=data["gained_permissions"],
            required_permissions=data["required_permissions"],
            compromise_sequences=data["compromise_sequences"],
        )

    def update_objects(
        self, techniques: Dict[str, Technique], permissions: Dict[str, Permission]
    ) -> None:
        # Update gained_permissions
        self.gained_permissions = [
            permissions[permission_id] for permission_id in self.gained_permissions
        ]

        # Update required_permissions
        # From permission ids to permission objects
        old_permissions = self.required_permissions
        new_permissions = dict()
        for technique_id, permissions_ids in old_permissions.items():
            new_permissions[technique_id] = [permissions[id] for id in permissions_ids]
        self.required_permissions = new_permissions

        # Update compromise_sequences
        # From technique ids to technique objects
        old_sequences = self.compromise_sequences
        new_sequences = list()
        for sequence in old_sequences:
            sequence_obj = list()
            for technique_id in sequence:
                sequence_obj.append(techniques[technique_id])
            new_sequences.append(sequence_obj)
        self.compromise_sequences = new_sequences

    def get_available_techniques(self) -> List[Technique]:
        """Get current available techniques that can be used to attack the device

        Returns:
            List[Technique]: _description_
        """
        available_techniques = list()
        for technique_sequence in self.compromise_sequences:
            sequence_done = True
            for technique in technique_sequence:
                if technique not in self.state.successful_techniques:
                    sequence_done = False
                    available_techniques.append(technique)
                    break
            if sequence_done:
                available_techniques.append("done")
        return available_techniques

    def perform_attack(
        self, technique: Technique, permissions: List[Permission]
    ) -> Tuple[bool, str, Permission]:
        # If the device have already been compromised, perform other techniques will not be counted anymore
        if self.state.compromised:
            return False, Constants.ATTACK_DEVICE_MSG.ALREADY_COMPROMISED, None

        feasible, reason = self.subnet.check_attack_feasibility(technique)
        if not feasible:
            return False, reason, None

        available_techniques = self.get_available_techniques()
        if technique not in available_techniques:
            return False, Constants.ATTACK_DEVICE_MSG.UNAVAILABLE_TECHNIQUE, None

        if not set(self.required_permissions[technique.id]).issubset(permissions):
            return False, Constants.ATTACK_DEVICE_MSG.NOT_HAVE_PERMISSION, None

        if technique.is_too_difficult():
            return False, Constants.ATTACK_DEVICE_MSG.TOO_DIFFICULT, None

        # At this point the technique is performed successfully
        self.state.successful_techniques.append(technique)

        # Check again if this device is compromised after the attack
        available_techniques = self.get_available_techniques()
        if "done" in available_techniques:
            # This device is compromised -> Gain permissions
            self.state.compromised = True
            message = (
                Constants.ATTACK_DEVICE_MSG.SUCCESSFUL_COMPROMISED
                if self.type != "honeypot"
                else Constants.ATTACK_DEVICE_MSG.ATTACKED_HONEYPOT
            )
            return (
                True,
                message,
                self.gained_permissions,
            )
        else:
            # This device is NOT compromised yet
            return True, Constants.ATTACK_DEVICE_MSG.SUCCESSFUL, None
