# this file implements wrappers for two SOTA solvers
# 1. Gurobi (exact): not recommended to use for large instance (TSP100~10s, TSP200~2m)
# 2. LKH3 (heuristic)

import elkai
import torch
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB


def solve_tsp_instance_by_LKH3(dist_matrix, border=1000000, runs=10):
    """
    solve an instance (by distance matrix) using LKH3 algorithm
    :param dist_matrix: a (size, size) tensor, the distance matrix for tsp instance
    :param border: the maximum of scaled distance values
    :param runs: repetition of LKH3
    :return: a (size, ) tensor, the solution tour
    """
    n = dist_matrix.size(0)
    amp = border / dist_matrix.max()
    dist_matrix = amp * dist_matrix
    tour = elkai.solve_int_matrix(dist_matrix.int().tolist(), runs=runs)
    assert len(tour) == n
    return torch.tensor(tour)


def solve_tsp_instance_by_Gurobi(dist_matrix):
    """
    copied and adapted from Gurobi-examples/tsp.py
    :param dist_matrix: a (size, size) tensor, the distance matrix for tsp instance
    :return: a (size, ) tensor, the solution tour
    """
    n = dist_matrix.size(0)
    if n > 100:
        print(f"Recommend to use LKH3 for large-instance inference.")

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            tour = subtour(vals)
            if len(tour) < n:
                model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)

    def subtour(vals):
        edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        unvisited = list(range(n))
        cycle = range(n + 1)
        while unvisited:
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    dist = {(i, j): dist_matrix[i, j] for i in range(n) for j in range(i)}
    m = gp.Model()
    m.Params.LogToConsole = 0
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]

    m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))

    m._vars = vars
    m.Params.LazyConstraints = 1
    m.optimize(subtourelim)

    vals = m.getAttr('X', vars)
    tour = subtour(vals)

    assert len(tour) == n
    return torch.tensor(tour)


def check_tsp_solution_validity(tour):
    """
    a valid tour should have a non-overlapping sequential indexing of existing nodes
    (this already ensures the absence of sub-tour)
    :param tour: the solution tour to be checked
    :return: True for valid tour; False for
    """
    return (tour >= 0).all().item() and (tour <= tour.size(0)).all().item() and tour.size() == tour.unique().size()



