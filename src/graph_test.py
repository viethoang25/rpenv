from src.components import Network
import matplotlib.pyplot as plt
import networkx as nx
import json

if __name__ == "__main__":
    json_file = open("scenarios/example.json")
    data = json.load(json_file)
    network = Network().create(data)
    graph = nx.Graph()
    for subnet in network.subnets.values():
        firewall_name = f"None-{subnet.id}"
        if subnet.firewall:
            graph.add_node(subnet.firewall.id, type="firewall")
            firewall_name = subnet.firewall.id
        else:
            graph.add_node(firewall_name, type="firewall")

        for device_id in subnet.devices.keys():
            graph.add_node(device_id, type="device")
            # Create edges between devices and firewalls
            graph.add_edge(device_id, firewall_name)

        # Create edges for every pairs of devices in the subnet
        device_key_list = list(subnet.devices.keys())
        for i in range(len(device_key_list)):
            for j in range(i + 1, len(device_key_list)):
                graph.add_edge(
                    subnet.devices[device_key_list[i]].id,
                    subnet.devices[device_key_list[j]].id,
                )

    # Create edges between firewalls
    checked_subnets = []
    for subnet in network.subnets.values():
        firewall_name_1 = f"None-{subnet.id}"
        if subnet.firewall:
            firewall_name_1 = subnet.firewall.id
        for connected_subnet in subnet.connected_subnets:
            firewall_name_2 = f"None-{connected_subnet.id}"
            if connected_subnet.firewall:
                firewall_name_2 = connected_subnet.firewall.id
            if firewall_name_1 not in checked_subnets:
                graph.add_edge(firewall_name_1, firewall_name_2)
        checked_subnets.append(firewall_name_1)

    # Create edges between techniques and devices
    for device in network.devices.values():
        for technique_id in device.required_permissions.keys():
            node_name = f"{device.id}-{technique_id}"
            graph.add_node(node_name, type="technique")
            graph.add_edge(device.id, node_name)

    node_color = []
    for node in graph.nodes(data=True):
        if "firewall" in node[1]["type"]:
            node_color.append("red")
        elif "device" in node[1]["type"]:
            node_color.append("blue")
        elif "technique" in node[1]["type"]:
            node_color.append("green")

    nx.draw(graph, with_labels=True, node_color=node_color)
    plt.show()
