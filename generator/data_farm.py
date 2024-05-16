# this file implements interface for generating datasets

import argparse
from rich_argparse_plus import RichHelpFormatterPlus
from pathlib import Path

from generate_instances import generate_uniform_tsp_instance
from generate_instances import generate_clustered_tsp_instance
from generate_instances import generate_explosion_tsp_instance
from generate_instances import generate_implosion_tsp_instance
from generate_instances import generate_uniform_cvrp_instance
from generate_instances import generate_clustered_cvrp_instance
from generate_instances import generate_explosion_cvrp_instance
from generate_instances import generate_implosion_cvrp_instance

from data_io import write_tsp_instances_to_file
from data_io import write_cvrp_instances_to_file


def generate_uniform_tspfarm(args, file_path):
    for i in range(args.num):
        tsp_instance = generate_uniform_tsp_instance(args.size)
        write_tsp_instances_to_file(tsp_instance, file_path)


def generate_clustered_tspfarm(args, file_path):
    for i in range(args.num):
        tsp_instance = generate_clustered_tsp_instance(args.size, args.cluster_num, args.cluster_diversity)
        write_tsp_instances_to_file(tsp_instance, file_path)


def generate_explosion_tspfarm(args, file_path):
    for i in range(args.num):
        tsp_instance = generate_explosion_tsp_instance(args.size, args.range_max, args.range_min, args.rate)
        write_tsp_instances_to_file(tsp_instance, file_path)


def generate_implosion_tspfarm(args, file_path):
    for i in range(args.num):
        tsp_instance = generate_implosion_tsp_instance(args.size, args.range_max, args.range_min)
        write_tsp_instances_to_file(tsp_instance, file_path)


def generate_uniform_cvrpfarm(args, file_path):
    for i in range(args.num):
        depot, nodes, demands, capacity = generate_uniform_cvrp_instance(
            args.size, args.min_demand, args.max_demand, args.capacity)
        write_cvrp_instances_to_file(depot, nodes, demands, capacity, file_path)


def generate_clustered_cvrpfarm(args, file_path):
    for i in range(args.num):
        depot, nodes, demands, capacity = generate_clustered_cvrp_instance(
            args.size, args.min_demand, args.max_demand, args.capacity)
        write_cvrp_instances_to_file(depot, nodes, demands, capacity, file_path)


def generate_explosion_cvrpfarm(args, file_path):
    for i in range(args.num):
        depot, nodes, demands, capacity = generate_explosion_cvrp_instance(
            args.size, args.min_demand, args.max_demand, args.capacity, args.range_max, args.range_min, args.rate)
        write_cvrp_instances_to_file(depot, nodes, demands, capacity, file_path)


def generate_implosion_cvrpfarm(args, file_path):
    for i in range(args.num):
        depot, nodes, demands, capacity = generate_implosion_cvrp_instance(
            args.size, args.min_demand, args.max_demand, args.capacity, args.range_max, args.range_min)
        write_cvrp_instances_to_file(depot, nodes, demands, capacity, file_path)


def generate_data_farm(args):
    problem_type = args.problem_type.lower()
    distribution = args.distribution.lower()
    size = args.size
    group = args.group if args.group is not None else ""

    base_dir = Path(args.data_root)
    problem_dir = base_dir.joinpath(f"{problem_type}")
    data_dir = problem_dir.joinpath(f"{problem_type}{size}")

    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    file_name = f"{problem_type}{size}_{distribution}{group}{suffix}.txt"
    file_path = data_dir.joinpath(file_name)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists() and not args.overwrite and not args.append:
        print(f"[!] Already exists data file {file_path}. Turn on overwrite or append flag.")
        exit(0)

    if args.overwrite:
        with open(file_path, 'w+', encoding='utf8'):
            pass

    if problem_type == "tsp":
        if distribution == "uniform":
            generate_uniform_tspfarm(args, file_path)
        elif distribution == "clustered":
            generate_clustered_tspfarm(args, file_path)
        elif distribution == "explosion":
            generate_explosion_tspfarm(args, file_path)
        elif distribution == "implosion":
            generate_implosion_tspfarm(args, file_path)
    elif problem_type == "cvrp":
        if distribution == "uniform":
            generate_uniform_cvrpfarm(args, file_path)
        elif distribution == "clustered":
            generate_clustered_cvrpfarm(args, file_path)
        elif distribution == "explosion":
            generate_explosion_cvrpfarm(args, file_path)
        elif distribution == "implosion":
            generate_implosion_cvrpfarm(args, file_path)

    print(f"Generation Finished!")
    print(f"[*] Summary       : {args.num} {problem_type}{size} instances under {distribution} distribution")
    print(f"[*] Problem Type  : {problem_type}")
    print(f"[*] Size          : {size}")
    print(f"[*] Number        : {args.num}")
    print(f"[*] Distribution  : {distribution}")
    print(f"[*] File Location : {file_path}")
    print(f"\n" * 5)


def parse():
    RichHelpFormatterPlus.choose_theme("prince")
    parser = argparse.ArgumentParser(
        description="Data Generation --- DataFarm.",
        formatter_class=RichHelpFormatterPlus,
    )

    # general hyperparameters (preferred not to be changed)
    general_args = parser.add_argument_group("General Hyperparameters")
    general_args.add_argument("--no-print-param", action="store_true",
                              help="Do not print the parameter information in log files.")

    # customized hyperparameters (preferred default values)
    customized_args = parser.add_argument_group("Customized Hyperparameters")
    customized_args.add_argument("--data-root", type=str, default="data/data_farm/",
                                 help="Path to instances.")
    customized_args.add_argument("--seed", type=int, default=None,
                                 help="Random seed.")
    customized_args.add_argument("--overwrite", action="store_true",
                                 help="Overwrite the existing data file.")
    customized_args.add_argument("--append", action="store_true",
                                 help="Append data to the existing data file.")
    customized_args.add_argument("--group", type=int, default=None,
                                 help="The group of datasets if need different variation of same distributions.")

    # typical hyperparameters (hyperparameters for variation)
    typical_args = parser.add_argument_group("TYPICAL HYPERPARAMETERS")
    typical_args.add_argument("--problem-type", type=str, default="tsp", choices=["tsp", "cvrp"],
                              help="Combinatorial Optimization problem type.")
    typical_args.add_argument("--size", type=int, default=50,
                              help="Size of instances.")
    typical_args.add_argument("--num", type=int, default=1000,
                              help="Number of instances.")
    typical_args.add_argument("--distribution", type=str, default="uniform",
                              choices=["uniform", "clustered", "explosion", "implosion"],
                              help="Distribution of TSP instances.")

    typical_args.add_argument("--min-demand", type=int, default=1,
                              help="Minimum node demand. Only valid in CVRP.")
    typical_args.add_argument("--max-demand", type=int, default=10,
                              help="Maximum node demand. Only valid in CVRP.")
    typical_args.add_argument("--capacity", type=int, default=50,
                              help="Capacity of vehicle. Only valid in CVRP.")

    typical_args.add_argument("--cluster-num", type=int, default=3,
                              help="Number of clusters (valid only for clustered distribution).")
    typical_args.add_argument("--cluster-diversity", type=float, default=10,
                              help="Diversity of clusters (valid only for clustered distribution).")
    typical_args.add_argument("--range-min", type=float, default=0.1,
                              help="Maximum range (valid only for explosion and implosion distribution).")
    typical_args.add_argument("--range-max", type=float, default=0.5,
                              help="Minimum range (valid only for explosion and implosion distribution).")
    typical_args.add_argument("--rate", type=float, default=10,
                              help="Rate for exponential distribution (valid only for explosion distribution).")

    args = parser.parse_args()

    if not args.no_print_param:
        for key, value in vars(args).items():
            print(f"{key} = {value}")
        print(f"=" * 20)
        print()

    return args


def main():
    args = parse()
    generate_data_farm(args)


if __name__ == '__main__':
    main()
