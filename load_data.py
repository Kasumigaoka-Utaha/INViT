import torch
from pathlib import Path


def read_solutions_from_file(file_path):
    tour_storage = []
    tour_len_storage = []
    ellapsed_time_storage = []
    with open(file_path, 'r', encoding='utf8') as read_file:
        line_text = read_file.readline()
        while line_text:
            tour_text, tour_len_text, ellapsed_time_text = line_text.strip().split(" ")

            tour = [int(val) for val in tour_text.split(",")]
            tour_storage.append(tour)

            tour_len = float(tour_len_text)
            tour_len_storage.append(tour_len)

            ellapsed_time = float(ellapsed_time_text)
            ellapsed_time_storage.append(ellapsed_time)

            line_text = read_file.readline()

    # for tour in tour_storage:
    #     print(len(tour))

    tours = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tour_storage], batch_first=True, padding_value=0)
    tour_lens = torch.tensor(tour_len_storage)
    time_consumptions = torch.tensor(ellapsed_time_storage)
    return tours, tour_lens, time_consumptions


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

    return torch.Tensor(depot), torch.Tensor(nodes), torch.Tensor(demands), torch.Tensor(capacity)


def load_tsp_instances_with_baselines(root, problem_type, size, distribution):
    assert problem_type == "tsp"
    assert size in (100, 1000, 5000, 10000)
    assert distribution in ('uniform', 'clustered1', 'clustered2', 'explosion', 'implosion')

    if size == 100:
        baseline = "Gurobi"
    elif size == 1000:
        baseline = "LKH3_runs10"
    elif size == 5000:
        baseline = "LKH3_runs1"
    elif size == 10000:
        baseline = "LKH3_runs1"

    instance_root = Path(root)
    instance_dir = f"data_farm/{problem_type}/{problem_type}{size}/"
    instance_name = f"{problem_type}{size}_{distribution}.txt"
    instance_file = instance_root.joinpath(instance_dir).joinpath(instance_name)

    tsp_instances = read_tsp_instances_from_file(instance_file)
    # num = tsp_instances.size(0)
    # print(tsp_instances.size())

    solution_root = Path(root)
    solution_dir = f"solution_farm/{problem_type}{size}_{distribution}/"
    solution_name = f"{baseline}.txt"
    solution_file = solution_root.joinpath(solution_dir).joinpath(solution_name)
    baseline_tours, baseline_lens, _ = read_solutions_from_file(solution_file)

    return tsp_instances, baseline_tours, baseline_lens


def load_cvrp_instances_with_baselines(root, problem_type, size, distribution):
    assert problem_type == "cvrp"
    assert size in (50, 500, 5000)
    assert distribution in ('uniform', 'clustered1', 'clustered2', 'explosion', 'implosion')

    # TODO: compare HGS and LKH3 after finishing LKH3 baselines and set final baseline
    baseline = "HGS"

    instance_root = Path(root)
    instance_dir = f"data_farm/{problem_type}/{problem_type}{size}/"
    instance_name = f"{problem_type}{size}_{distribution}.txt"
    instance_file = instance_root.joinpath(instance_dir).joinpath(instance_name)

    cvrp_instances = read_cvrp_instances_from_file(instance_file)
    # num = tsp_instances.size(0)
    # print(tsp_instances.size())

    solution_root = Path(root)
    solution_dir = f"solution_farm/{problem_type}{size}_{distribution}/"
    solution_name = f"{baseline}.txt"
    solution_file = solution_root.joinpath(solution_dir).joinpath(solution_name)
    baseline_tours, baseline_lens, _ = read_solutions_from_file(solution_file)

    return cvrp_instances, baseline_tours, baseline_lens


def load_instances_with_baselines(root, problem_type, size, distribution):
    assert problem_type in ("tsp", "cvrp")
    if problem_type == "tsp":
        return load_tsp_instances_with_baselines(root, problem_type, size, distribution)
    elif problem_type == "cvrp":
        return load_cvrp_instances_with_baselines(root, problem_type, size, distribution)


def main():
    root = f"./data/"
    problem_type = "cvrp"
    size = 50
    distribution = "clustered1"
    instances, b_tours, b_lens = load_instances_with_baselines(root, problem_type, size, distribution)
    depot, nodes, demands, capacity = instances

    print(depot.size(), depot.device)
    print(nodes.size(), nodes.device)
    print(demands, demands.device)
    print(capacity.size(), capacity.device)
    print(b_tours.size(), b_tours.device)
    print(b_lens.size(), b_lens.device)


if __name__ == "__main__":
    main()

