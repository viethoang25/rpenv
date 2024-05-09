import json
from src.components import Network, Device, Subnet
from . import *


class EnvironmentHistory:
    def __init__(self) -> None:
        self.actions = []
        self.done_type: str

    def to_json(self):
        json_actions = []
        for step in self.actions:
            json_actions.append(
                dict(
                    action=dict(
                        target=step[1].target.id,
                        technique=step[1].technique.id,
                        type=step[1].type.__name__,
                    ),
                    reward=step[2],
                    info=step[3],
                )
            )
        return dict(actions=json_actions, done_type=self.done_type)


class MitreEnvironment:
    def __init__(self, network_scenario: str) -> None:
        json_file = open(network_scenario)
        data = json.load(json_file)
        self.network = Network().create(data)
        self.action_space = ActionSpace(self.network)
        self.state = State()
        self.history = EnvironmentHistory()  # History of current run

        print("Environment Information:")
        print("\tType: normal")
        print("\tAction space:", self.action_space.n)

    def _get_obs(self):
        return self.state.get_observation(self.network)

    def reset(self):
        self.network.reset()
        self.history = EnvironmentHistory()
        return self._get_obs()

    def render(self) -> str:
        TableState().get_observation(self.network)
        self.action_space.show()

    def step(self, action: int):
        action_id = action
        action: Action = self.action_space.get_action(action)
        success = False
        reward = Reward.INIT
        info = {"exploit_sensitive": False, "reason": "", "success": False}
        # The environment finishes when:
        # 1: All sensitive devices are compromised
        # 2: The time is run out
        # 3: Attacked a honeypot device
        done = False
        done_type = None

        # Get reward and additional info
        if action.type is Subnet:
            success, message = self.network.attack_subnet(
                subnet=action.target, technique=action.technique
            )
            reward = Reward.SUBNET if success else reward
            info["reason"] = message
            info["success"] = success
        elif action.type is Device:
            success, message = self.network.attack_device(
                device=action.target, technique=action.technique
            )
            info["reason"] = message
            info["success"] = success
            reward = Reward.DEVICE if success else reward
            if success and action.target.state.compromised:
                reward = Reward.COMPROMISED
                if action.target.type == "honeypot":
                    # If the device is honeypot
                    reward = Reward.HONEYPOT
                    done = True
                    done_type = "honeypot"
                elif action.target.sensitive:
                    # If the device is sensitive
                    reward = Reward.SENSITIVE
                    info["exploit_sensitive"] = True

                    # Check if all sensitive devices is compromised
                    done = True
                    done_type = "sensitive"
                    for devices in self.network.devices.values():
                        if devices.sensitive and not devices.state.compromised:
                            done = False
                            done_type = None
                            break

        if self.network.get_remaining_duration() <= 0:
            done = True
            done_type = "timeout"

        cur_state = self._get_obs()

        self.history.actions.append((action_id, action, reward, info))
        if done:
            self.history.done_type = done_type

        return cur_state, reward, done, info
