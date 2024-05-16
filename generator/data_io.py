# this file implements several functions to write/read instances to/from system files

import torch


def write_tsp_instances_to_file(instances, file_path):
    """
    write normalized tsp instances (without solution) to the given file
    :param instances: a (num, size, 2) tensor or a (size, 2) tensor, one or multiple tsp instances
    :param file_path: the output data path
    :return: None, but data written into the file
    """
    if instances.dim() == 2:
        instances = instances.unsqueeze(dim=0)

    with open(file_path, 'a+', encoding='utf8') as write_file:
        for instance in instances:
            contents = " ".join([f"{node[0].item()},{node[1].item()}" for node in instance])
            write_file.write(f"{contents}\n")


def read_tsp_instances_from_file(file_path):
    """
    read instances from the given file (should follow the rules in write_tsp_instances_to_file())
    :param file_path: the input data path
    :return: a (num, size, 2) tensor in cpu, multiple tsp instances
    """
    tsp_instances = []
    with open(file_path, 'r', encoding='utf8') as read_file:
        line_text = read_file.readline()
        while line_text:
            splitted_text = line_text.strip().split(" ")
            tsp_instance = []
            for node_text in splitted_text:
                tsp_instance.append([float(val) for val in node_text.split(",")])
            tsp_instances.append(tsp_instance)
            line_text = read_file.readline()
    return torch.Tensor(tsp_instances)


def write_cvrp_instances_to_file(depot, nodes, demands, capacity, file_path):
    if depot.dim() == 1:
        depot = depot.unsqueeze(dim=0)
    if nodes.dim() == 2:
        nodes = nodes.unsqueeze(dim=0)
    if demands.dim() == 1:
        demands = demands.unsqueeze(dim=0)
    if capacity.dim() == 1:
        capacity = capacity.unsqueeze(dim=0)

    size = nodes.size(1)
    assert size == demands.size(1)
    num = nodes.size(0)
    assert num == nodes.size(0)
    assert num == demands.size(0)
    assert num == capacity.size(0)
    print(capacity, capacity.size(1))
    assert capacity.size(1) == 1

    with open(file_path, 'a+', encoding='utf8') as write_file:
        for i in range(num):
            instance_depot = depot[i]
            instance_nodes = nodes[i]
            instance_demands = demands[i]
            instance_capacity = capacity[i]

            contents = ""
            contents += f"{instance_depot[0].item()},{instance_depot[1].item()}"
            contents += " .|. "
            contents += " ".join([f"{x[0].item()},{x[1].item()}" for x in instance_nodes])
            contents += " .|. "
            contents += " ".join([f"{x.item()}" for x in instance_demands])
            contents += " .|. "
            contents += f"{instance_capacity.item()}"

            write_file.write(f"{contents}\n")


def read_cvrp_instances_from_file(file_path):
    depot = []
    nodes = []
    demands = []
    capacity = []

    with open(file_path, 'r', encoding='utf8') as read_file:
        line_text = read_file.readline()
        while line_text:
            splitted_text = line_text.strip().split(" .|. ")

            instance_depot = [float(x) for x in splitted_text[0].strip().split(",")]

            instance_nodes = []
            for node_text in splitted_text[1].strip().split(" "):
                instance_nodes.append([float(x) for x in node_text.split(",")])

            instance_demands = [int(x) for x in splitted_text[2].strip().split(" ")]
            instance_capacity = int(splitted_text[3])

            depot.append(instance_depot)
            nodes.append(instance_nodes)
            demands.append(instance_demands)
            capacity.append(instance_capacity)

            line_text = read_file.readline()

    return torch.tensor(depot), torch.tensor(nodes), torch.tensor(demands), torch.tensor(capacity)


def read_settings_from_file_name(file_name):
    """
    read basic settings from file_name
    file_name = f"{problem_type}{size}_{distribution}{group}[_seed{seed}].txt"
    :param file_name: the input file name
    :return: a tuple of settings (problem_type, size, distribution, group, seed)
    """
    file_stem = file_name.rstrip(".txt")
    splitted_file_stem = file_stem.split("_")
    problem_text = splitted_file_stem[0]
    distribution_text = splitted_file_stem[1]
    seed_text = splitted_file_stem[2] if len(splitted_file_stem) >= 3 else ""

    if problem_text.startswith("tsp"):
        problem_type = "tsp"
    elif problem_text.startswith("cvrp"):
        problem_type = "cvrp"
    else:
        print(f"[!] Unexpected file_name {file_name}: cannot recognize problem type")
        exit(0)

    size = int(problem_text.lstrip(problem_type))

    if distribution_text.startswith("uniform"):
        distribution = "uniform"
    elif distribution_text.startswith("clustered"):
        distribution = "clustered"
    elif distribution_text.startswith("explosion"):
        distribution = "explosion"
    elif distribution_text.startswith("implosion"):
        distribution = "implosion"
    else:
        print(f"[!] Unexpected file_name {file_name}: cannot recognize distribution")
        exit(0)

    group = int(distribution_text.lstrip(distribution)) if not distribution_text.endswith(distribution) else None

    seed = int(seed_text.lstrip("seed")) if seed_text.startswith("seed") else None

    return problem_type, size, distribution, group, seed


