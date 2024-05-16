import torch
import os
import tempfile
import numpy as np
import hygese as hgs
from subprocess import check_output


def solve_cvrp_instance_by_HGS(points, demand, capacity, dist_matrix, scale=1000):
    size = demand.size(0)
    data = dict()
    data['x_coordinates'] = (scale * points[:, 0]).float().tolist()
    data['y_coordinates'] = (scale * points[:, 1]).float().tolist()
    data['distance_matrix'] = (scale * dist_matrix).tolist()
    data['num_vehicles'] = size
    data['depot'] = 0
    data['demands'] = [0] + demand.tolist()
    data['vehicle_capacity'] = capacity.item()
    data['service_times'] = np.zeros(len(data['demands']))

    # Solver initialization
    # TODO default timelimit for makeup evaluation when the first trial exceed a total time limit
    # in normal case, should be ap = hgs.AlgorithmParameters()
    ap = hgs.AlgorithmParameters()
    if size >= 1000:
        ap = hgs.AlgorithmParameters(timeLimit=14400)
    hgs_solver = hgs.Solver(parameters=ap, verbose=False)

    # Solve
    result = hgs_solver.solve_cvrp(data)

    tour = [0]
    for route in result.routes:
        tour += route
        tour += [0]

    return torch.tensor(tour)


def write_cvrp_instance_for_lkh(cvrp_instance, problem_file):
    depot, nodes, demands, capacity = cvrp_instance
    depot = depot.tolist()
    nodes = nodes.tolist()
    demands = demands.tolist()
    capacity = capacity.item()
    size = len(nodes)

    amp = 100000
    border = 1
    depot = [round(depot[0] * amp / border), round(depot[1] * amp / border)]
    nodes = [[round(x * amp / border), round(y * amp / border)] for (x, y) in nodes]

    with open(problem_file, 'w+', encoding='utf8') as file:
        # basic information
        file.write(f"NAME : problem\n")
        file.write(f"TYPE : CVRP\n")
        file.write(f"DIMENSION : {size + 1}\n")
        file.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write(f"CAPACITY : {capacity}\n")
        file.write(f"\n")

        # depot and nodes
        file.write(f"NODE_COORD_SECTION\n")
        file.write(f"1\t{depot[0]}\t{depot[1]}\n")
        for index in range(size):
            x, y = nodes[index]
            file.write(f"{index + 2}\t{x}\t{y}\n")
        file.write(f"\n")

        # demands
        file.write(f"DEMAND_SECTION\n")
        file.write(f"1\t0\n")
        for index in range(size):
            file.write(f"{index + 2}\t{demands[index]}\n")
        file.write(f"\n")

        file.write(f"DEPOT_SECTION\n")
        file.write(f"1\n")
        file.write(f"-1\n")
        file.write(f"EOF\n")


def write_lkh_params(param_file, params):
    default_params = {
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0,
    }
    with open(param_file, 'w') as f:
        for key, value in {**default_params, **params}.items():
            if value is None:
                f.write(f"{key}\n")
            else:
                f.write(f"{key} = {value}\n")


def read_lkh_outputs(output_file):
    if not os.path.exists(output_file):
        return None

    tour = []
    with open(output_file, 'r+', encoding='utf8') as file:
        contents = file.readlines()
        for line in contents:
            line = line.strip("\n")
            if line.isdigit():
                tour.append(int(line))
    return tour


def solve_cvrp_instance_by_LKH3(cvrp_instance, lkh3_executable, params):
    nodes = cvrp_instance[1]
    size = nodes.size(0)

    with tempfile.TemporaryDirectory() as tempdir:
        problem_file = os.path.join(tempdir, "problem.vrp")
        output_file = os.path.join(tempdir, "output.tour")
        param_file = os.path.join(tempdir, "params.par")

        write_cvrp_instance_for_lkh(cvrp_instance=cvrp_instance, problem_file=problem_file)

        params["PROBLEM_FILE"] = problem_file
        params["OUTPUT_TOUR_FILE"] = output_file
        write_lkh_params(param_file, params)

        output = check_output([lkh3_executable, param_file])

        tour = read_lkh_outputs(output_file)
        if tour is None:
            print(f"[-] Detect one failed LKH execution, omit this iteration!")
            return tour

        tour = [x - 1 if x <= size + 1 else 0 for x in tour] + [0]
        return torch.tensor(tour)


def check_cvrp_solution_validity(tour, demands, size):
    tour = tour.tolist()
    visited = []
    cap = 50
    for i in range(len(tour)):
        if tour[i] == 0:
            cap = 50
            continue

        if tour[i] in visited:
            return False

        visited.append(tour[i])
        cap -= demands[tour[i] - 1]
        if cap < 0:
            return False

    if len(visited) != size:
        return False

    return True
