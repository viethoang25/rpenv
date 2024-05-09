from typing import List
from src.components import Network, Subnet, Technique, Device, Permission

from rich.console import Console
from rich.table import Table


class State:
    TECHNIQUE = {"attacked": 1, "available": 0, "unavailable": -1}
    PERMISSION = {"available": 1, "unavailable": 0, "unknown": -1}

    def get_observation(self, network: Network) -> List[float]:
        state: List[float] = list()
        # Get state from network
        state.extend(self.get_network_obs(network))
        # Get state from subnets
        for subnet in network.subnets.values():
            state.extend(
                self.get_subnet_obs(subnet, network.subnet_techniques.values())
            )
        # Get state from devices
        for device in network.devices.values():
            state.extend(
                self.get_device_obs(
                    device,
                    network.device_techniques.values(),
                    network.permissions.values(),
                )
            )
        return state

    def get_network_obs(self, network: Network):
        """Get network observation

        Dimensionality = 1 (duration) + no. permission
        """
        state: List[float] = list()

        # Add remaining duration
        state.append(network.get_remaining_duration() / network.duration_limit)

        for permission in network.permissions.values():
            if permission in network.state.obtained_permissions:
                # If a permission is obtained, set its state to "available"
                state.append(State.PERMISSION["available"])
            else:
                state.append(State.PERMISSION["unavailable"])
        return state

    def get_subnet_obs(self, subnet: Subnet, techniques: List[Technique]):
        """Get subnet observation

        Dimensionality = 2 (explorable and attackable state) + no. subnet-scope techniques
        """
        state: List[float] = list()
        state.append(1 if subnet.state.explorable else 0)
        state.append(1 if subnet.state.attackable else 0)

        # Notice: there is no permission needed for technique at subnet level
        for technique in techniques:
            if technique not in subnet.exploration_techniques:
                # Unavailable techniques
                state.append(State.TECHNIQUE["unavailable"])
            elif technique not in subnet.state.successful_techniques:
                # Technique that are available but not being attacked yet
                state.append(State.TECHNIQUE["available"])
            else:
                # Techniques that are succesfully attacked
                state.append(State.TECHNIQUE["attacked"])

        return state

    def get_device_obs(
        self, device: Device, techniques: List[Technique], permissions: List[Permission]
    ) -> List[float]:
        """Observation of a device.

        Dimensionality = 1 (compromised) + no. device-scope permissions + no. techniques * (1 + no. device-scope permissions)

        The observation has this structure:
        - Compromised or not
        - List of gained permission (if successfully attack the device)
        - The state of each technique
        \t- Is the technique attacked, available or unavailable (1, 0, -1)
        \t- The required permission for the technique
        """
        state: List[float] = list()
        # Gained compromised state
        state.append(1 if device.state.compromised else 0)
        # Gained permission state, the permission gained after successfully attack the device
        state.extend(
            [
                State.PERMISSION["available"]
                if p in device.gained_permissions
                else State.PERMISSION["unavailable"]
                for p in permissions
            ]
        )
        for technique in techniques:
            if technique not in device.get_available_techniques():
                # Unavailable techniques
                state.append(State.TECHNIQUE["unavailable"])
                # Permission vector for each unavailable techniques is the same
                # Since they are unavailable, there is no permission set
                state.extend([State.PERMISSION["unknown"]] * len(permissions))
            else:
                if technique not in device.state.successful_techniques:
                    # Technique that are available but not being attacked yet
                    state.append(State.TECHNIQUE["available"])
                else:
                    # Techniques that are succesfully attacked
                    state.append(State.TECHNIQUE["attacked"])
                # Add permission vector for available techniques (both attacked and non-attacked)
                state.extend(
                    [
                        State.PERMISSION["available"]
                        if p in device.required_permissions[technique.id]
                        else State.PERMISSION["unavailable"]
                        for p in permissions
                    ]
                )

        return state


class TableState:
    def get_observation(self, network: Network) -> List[int]:
        self.console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Remaining")
        for permission in network.permissions.values():
            table.add_column(permission.id)
        table.add_row(
            str(network.get_remaining_duration()), *self.get_network_obs(network)
        )
        self.console.print(table)

        # Create table for subnets
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Subnet")
        table.add_column("Explorable")
        table.add_column("Attackable")
        for technique in network.subnet_techniques.values():
            table.add_column(technique.id)
        for subnet in network.subnets.values():
            table.add_row(
                *self.get_subnet_obs(subnet, network.subnet_techniques.values())
            )
        self.console.print(table)

        # Create table for devices
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Device")
        table.add_column("Compromised")
        table.add_column("Gained P.")
        for technique in network.device_techniques.values():
            table.add_column(technique.id)
            table.add_column("Perm")
        for device in network.devices.values():
            table.add_row(
                *self.get_device_obs(device, network.device_techniques.values())
            )
        self.console.print(table)

    def get_network_obs(self, network: Network):
        state = list()

        for permission in network.permissions.values():
            if permission in network.state.obtained_permissions:
                state.append("Obtained")
            else:
                state.append("Not-obtained")
        return tuple(state)

    def get_subnet_obs(self, subnet: Subnet, techniques: List[Technique]):
        state = list()

        state.append(subnet.id)
        state.append("True" if subnet.state.explorable else "False")
        state.append("True" if subnet.state.attackable else "False")

        for technique in techniques:
            if technique in subnet.state.successful_techniques:
                state.append("Attacked")
            elif technique in subnet.exploration_techniques:
                state.append("Available")
            else:
                state.append("Unavailable")
        return tuple(state)

    def get_device_obs(self, device: Device, techniques: List[Technique]):
        state = list()
        state.append(device.id)
        state.append("True" if device.state.compromised else "False")
        if not device.subnet.state.attackable:
            state.extend(["-"] * (len(techniques) * 2 + 1))
            return tuple(state)
        state.append(",".join([p.id for p in device.gained_permissions]))

        for technique in techniques:
            if technique in device.state.successful_techniques:
                state.append("Attacked")
                permissions = ",".join(
                    [p.id for p in device.required_permissions[technique.id]]
                )
                state.append(permissions)
            elif technique in device.get_available_techniques():
                state.append("Avaialble")
                permissions = ",".join(
                    [p.id for p in device.required_permissions[technique.id]]
                )
                state.append(permissions)
            else:
                state.append("Unavailable")
                state.append("-")
        return tuple(state)
