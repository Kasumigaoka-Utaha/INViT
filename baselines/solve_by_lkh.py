# this file provides LKH3 tsp solver wrapper for our research

import time
import sys
import argparse
from rich_argparse_plus import RichHelpFormatterPlus
from pathlib import Path

sys.path.append(f"generator/")
sys.path.append(f"utils/")
from solve_tsp_baselines import solve_tsp_instance_by_LKH3
from solve_tsp_baselines import check_tsp_solution_validity
from data_io import read_tsp_instances_from_file
from utilities import avg_list
from utilities import get_dist_matrix
from utilities import calculate_tour_length_by_dist_matrix


def create_tsp_baselines_by_LKH3(args):
    problem_type = args.problem_type.lower()

    file_path = Path(args.path)
    if not file_path.exists():
        print(f"[!] TSP file {file_path} does not exist.")
        exit(0)

    save_dir = Path(args.solution_root)
    solution_dir = save_dir.joinpath(file_path.stem)
    solution_name = f"LKH3_runs{args.runs}.txt"
    solution_path = solution_dir.joinpath(solution_name)
    solution_path.parent.mkdir(parents=True, exist_ok=True)

    if solution_path.exists() and not args.overwrite:
        print(f"[!] Solution file {solution_path} already exists. Turn on overwrite flag")
        exit(0)

    if args.overwrite:
        with open(solution_path, 'w+', encoding='utf8'):
            pass

    tsp_instances = read_tsp_instances_from_file(file_path)
    num, size, _ = tsp_instances.size()

    tour_len_storage = []
    ellapsed_time_storage = []

    for instance in tsp_instances:
        start_time = time.time()
        dist_matrix = get_dist_matrix(instance)
        tour = solve_tsp_instance_by_LKH3(dist_matrix, border=args.border, runs=args.runs)
        end_time = time.time()

        assert check_tsp_solution_validity(tour)
        tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, tour)
        ellapsed_time = end_time - start_time

        tour_text = ",".join([f"{val.item()}" for val in tour])
        solution_text = f"{tour_text} {tour_len.item()} {ellapsed_time}\n"

        tour_len_storage.append(tour_len.item())
        ellapsed_time_storage.append(ellapsed_time)

        with open(solution_path, 'a+', encoding='utf8') as write_file:
            write_file.write(solution_text)

        if size == 10000:
            print(f"One inference costs {ellapsed_time}")

    print(f"LKH3 Inference Finished!")
    print(f"[*] Summary           : LKH3 solver with {args.runs} runs")
    print(f"[*] File Location     : {file_path}")
    print(f"[*] Solution Location : {solution_path}")
    print(f"[*] Average length    : {avg_list(tour_len_storage)}")
    print(f"[*] Average time (s)  : {avg_list(ellapsed_time_storage)}")
    print(f"\n" * 5)


def create_baselines_by_LKH3(args):
    if args.problem_type.lower() == "tsp":
        create_tsp_baselines_by_LKH3(args)
    elif args.problem_type.lower() == "cvrp":
        print(f"Not implemented.")
        exit(0)


def parse():
    RichHelpFormatterPlus.choose_theme("prince")
    parser = argparse.ArgumentParser(
        description="LKH solver.",
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
    typical_args.add_argument("--problem-type", type=str, default="tsp", choices=["tsp", "cvrp"],
                              help="Combinatorial Optimization problem type.")
    typical_args.add_argument("--path", type=str, default="data/data_farm/tsp/tsp50/tsp50_uniform.txt",
                              help="Path to TSP instance file to be evaluated.")
    typical_args.add_argument("--border", type=int, default=1000000,
                              help="Maximum scaled values for integer distance matrix.")
    typical_args.add_argument("--runs", type=int, default=1,
                              help="Repetitions for LKH3 algorithm.")

    args = parser.parse_args()

    if not args.no_print_param:
        for key, value in vars(args).items():
            print(f"{key} = {value}")
        print(f"=" * 20)
        print()

    return args


def main():
    args = parse()
    create_baselines_by_LKH3(args)


if __name__ == '__main__':
    main()
