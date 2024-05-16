# this file provides HGS cvrp solver

import torch
import time
import sys
import argparse
from rich_argparse_plus import RichHelpFormatterPlus
from pathlib import Path

sys.path.append(f"generator/")
sys.path.append(f"utils/")
from solve_cvrp_baselines import solve_cvrp_instance_by_HGS
from solve_cvrp_baselines import check_cvrp_solution_validity
from data_io import read_cvrp_instances_from_file
from utilities import avg_list
from utilities import get_dist_matrix
from utilities import calculate_tour_length_by_dist_matrix


def create_cvrp_baselines_by_HGS(args):
    problem_type = args.problem_type.lower()

    file_path = Path(args.path)
    if not file_path.exists():
        print(f"[!] TSP file {file_path} does not exist.")
        exit(0)

    save_dir = Path(args.solution_root)
    solution_dir = save_dir.joinpath(file_path.stem)
    solution_name = f"HGS.txt"
    solution_path = solution_dir.joinpath(solution_name)
    solution_path.parent.mkdir(parents=True, exist_ok=True)

    if solution_path.exists() and not args.overwrite:
        print(f"[!] Solution file {solution_path} already exists. Turn on overwrite flag")
        exit(0)

    if args.overwrite:
        with open(solution_path, 'w+', encoding='utf8'):
            pass

    depot, nodes, demands, capacity = read_cvrp_instances_from_file(file_path)
    num, size, _ = nodes.size()

    tour_len_storage = []
    ellapsed_time_storage = []

    for i in range(num):
        instance_depot = depot[i]
        instance_nodes = nodes[i]
        instance_demands = demands[i]
        instance_capacity = capacity[i]

        start_time = time.time()
        points = torch.cat((instance_depot.unsqueeze(dim=0), instance_nodes), dim=0)
        dist_matrix = get_dist_matrix(points)
        tour = solve_cvrp_instance_by_HGS(points, instance_demands, instance_capacity, dist_matrix)
        end_time = time.time()

        assert check_cvrp_solution_validity(tour, instance_demands, size)
        tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, tour)
        ellapsed_time = end_time - start_time

        tour_text = ",".join([f"{val.item()}" for val in tour])
        solution_text = f"{tour_text} {tour_len.item()} {ellapsed_time}\n"

        tour_len_storage.append(tour_len.item())
        ellapsed_time_storage.append(ellapsed_time)

        with open(solution_path, 'a+', encoding='utf8') as write_file:
            write_file.write(solution_text)

    print(f"HGS Inference Finished!")
    print(f"[*] Summary           : HGS solver")
    print(f"[*] File Location     : {file_path}")
    print(f"[*] Solution Location : {solution_path}")
    print(f"[*] Average length    : {avg_list(tour_len_storage)}")
    print(f"[*] Average time (s)  : {avg_list(ellapsed_time_storage)}")
    print(f"\n" * 5)


def create_baselines_by_HGS(args):
    if args.problem_type.lower() == "tsp":
        print(f"Not implemented.")
        exit(0)
    elif args.problem_type.lower() == "cvrp":
        create_cvrp_baselines_by_HGS(args)


def parse():
    RichHelpFormatterPlus.choose_theme("prince")
    parser = argparse.ArgumentParser(
        description="Hybrid Genetic Search CVRP solver.",
        formatter_class=RichHelpFormatterPlus,
    )

    # general hyperparameters (preferred not to be changed)
    general_args = parser.add_argument_group("General Hyperparameters")
    general_args.add_argument("--no-print-param", action="store_true",
                              help="Do not print the parameter information in log files.")

    # customized hyperparameters (preferred default values)
    customized_args = parser.add_argument_group("Customized Hyperparameters")
    customized_args.add_argument("--solution-root", type=str, default="data/solution_farm/",
                                 help="Path to solutions.")
    customized_args.add_argument("--overwrite", action="store_true",
                                 help="Overwrite the existing solution file.")

    # typical hyperparameters (hyperparameters for variation)
    typical_args = parser.add_argument_group("TYPICAL HYPERPARAMETERS")
    typical_args.add_argument("--problem-type", type=str, default="cvrp", choices=["tsp", "cvrp"],
                              help="Combinatorial Optimization problem type.")
    typical_args.add_argument("--path", type=str, default="data/data_farm/cvrp/cvrp20/cvrp20_uniform.txt",
                              help="Path to CVRP instance file to be evaluated.")

    args = parser.parse_args()

    if not args.no_print_param:
        for key, value in vars(args).items():
            print(f"{key} = {value}")
        print(f"=" * 20)
        print()

    return args


def main():
    args = parse()
    create_baselines_by_HGS(args)


if __name__ == '__main__':
    main()
