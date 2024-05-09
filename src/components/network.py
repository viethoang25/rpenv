from __future__ import annotations
from typing import Dict, List, Any, Callable, Set, Tuple
from .subnet import Subnet, SubnetState
from .permission import Permission
from .technique import Technique
from .firewall import Firewall
from .device import Device, DeviceState


class NetworkState:
    def __init__(self) -> None:
        self.used_duration: float = 0
        self.obtained_permissions: Set[Permission] = set()


class Network:
    def __init__(self) -> None:
        self.subnets: Dict[str, Subnet] = dict()
        self.device_techniques: Dict[str, Technique] = dict()
        self.subnet_techniques: Dict[str, Technique] = dict()
        self.firewalls: Dict[str, Firewall] = dict()
        self.permissions: Dict[str, Permission] = dict()
        self.devices: Dict[str, Device] = dict()
        self.duration_limit: float = 100

        self.state = NetworkState()
        self.init_state = None

    def _create_instance(self, data: Any, init_func: Callable):
        instance = dict()
        for entity_data in data:
            entity = init_func(entity_data)
            instance[entity.id] = entity
        return instance

    def _init_state(self):
        self.duration_limit = self.init_state["duration_limit"]
        for subnet_id in self.init_state["explorable_subnets"]:
            self.subnets[subnet_id].update_explorable(True)
        for device_id in self.init_state["compromised_devices"]:
            self.devices[device_id].state.compromised = True
            self.devices[device_id].subnet.state.explorable = True
            self.state.obtained_permissions.update(
                self.devices[device_id].gained_permissions
            )
        for permission_id in self.init_state["obtained_permissions"]:
            self.state.obtained_permissions.add(self.permissions[permission_id])

    def create(self, data: Dict) -> Network:
        # Create techniques
        self.device_techniques = self._create_instance(
            data["techniques"]["device_scope"], Technique.from_json
        )
        self.device_techniques = dict(sorted(self.device_techniques.items()))
        self.subnet_techniques = self._create_instance(
            data["techniques"]["subnet_scope"], Technique.from_json
        )
        self.subnet_techniques = dict(sorted(self.subnet_techniques.items()))

        # Create permissions
        self.permissions = self._create_instance(
            data["permissions"], Permission.from_json
        )
        self.permissions = dict(
            sorted(self.permissions.items(), key=lambda item: item[1].level)
        )

        # Create firewalls
        self.firewalls = self._create_instance(data["firewalls"], Firewall.from_json)
        for firewall in self.firewalls.values():
            firewall.update_objects(techniques=self.device_techniques)

        # Create devices
        self.devices = self._create_instance(data["devices"], Device.from_json)
        for device in self.devices.values():
            device.update_objects(
                techniques=self.device_techniques, permissions=self.permissions
            )

        # Create subnets
        self.subnets = self._create_instance(data["subnets"], Subnet.from_json)

        # Create connected subnets for each subnet
        for connection_data in data["connections"]:
            subnet_1 = self.subnets[connection_data[0]]
            subnet_2 = self.subnets[connection_data[1]]
            subnet_1.add_connected_subnet(subnet_2)
            subnet_2.add_connected_subnet(subnet_1)

        for subnet in self.subnets.values():
            subnet.update_objects(self.devices, self.firewalls, self.subnet_techniques)

        # Set init state
        self.init_state = data["init_state"]
        self._init_state()

        return self

    def attack_device(
        self, device: str | Device, technique: str | Technique
    ) -> Tuple[bool, str]:
        if type(device) is str:
            device = self.devices[device]
        if type(technique) is str:
            technique = self.device_techniques[technique]
        success, message, gained_permissions = device.perform_attack(
            technique, self.state.obtained_permissions
        )

        # Update the duration used
        self.state.used_duration += technique.duration
        # If obtain new permissions from attacking the device, add it into obtained_permissions
        if gained_permissions:
            self.state.obtained_permissions.update(gained_permissions)

        return success, message

    def attack_subnet(self, subnet: str | Subnet, technique: str | Technique):
        if type(subnet) is str:
            subnet = self.subnets[subnet]
        if type(technique) is str:
            technique = self.subnet_techniques[technique]
        # Update the duration used
        self.state.used_duration += technique.duration
        return subnet.perform_attack(technique)

    def get_remaining_duration(self) -> float:
        return self.duration_limit - self.state.used_duration

    def reset(self):
        for device in self.devices.values():
            device.state = DeviceState()
        for subnet in self.subnets.values():
            subnet.state = SubnetState()
        self.state = NetworkState()
        self._init_state()
