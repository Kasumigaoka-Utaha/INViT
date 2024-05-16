import torch
import numpy as np
from pathlib import Path


#######################
# TSP utility functions
#######################

def parse_tsplib_name(tsplib_name):
    return "".join(filter(str.isalpha, tsplib_name)), int("".join(filter(str.isdigit, tsplib_name)))

def parse_cvrplib_name(cvrplib_name):
    problem_set, size, _ = cvrplib_name.split("-")
    size = int("".join(list(filter(str.isdigit, size)))) - 1
    return problem_set, size

def read_tsplib_file(file_path):
    """
    The read_tsplib_file function reads a TSPLIB file and returns the nodes and name of the problem.
    
    :param file_path: Specify the path to the file that is being read
    :return: A list of nodes and a name
    """
    properties = {}
    reading_properties_flag = True
    nodes = []

    with open(file_path, "r", encoding="utf8") as read_file:
        line = read_file.readline()
        while line.strip():
            # read properties
            if reading_properties_flag:
                if ':' in line:
                    key, val = [x.strip() for x in line.split(':')]
                    properties[key] = val
                else:
                    reading_properties_flag = False

            # read node coordinates
            else:
                if line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    pass
                else:
                    line_contents = [x.strip() for x in line.split(" ") if x.strip()]
                    _, x, y = line_contents
                    nodes.append([float(x), float(y)])
            line = read_file.readline()

    return nodes, properties["NAME"]


def read_cvrplib_file(file_path):
    """
    The read_cvrplib_file function reads a CVRP file and returns the depot, nodes, demands and properties.
    
    :param file_path: Specify the path of the file to be read
    :return: A tuple of four elements:
    """
    properties = {}
    reading_properties_flag = True
    reading_nodes_flag = True
    read_demands_flag = True
    read_depot_flag = True

    depot_nodes = []
    demands = []
    depot_check = []

    with open(file_path, "r", encoding="utf8") as read_file:
        line = read_file.readline()
        while line.strip():
            # read properties
            if reading_properties_flag:
                if ':' in line:
                    key, val = [x.strip() for x in line.split(':')]
                    properties[key] = val
                else:
                    reading_properties_flag = False

            # read node coordinates
            elif reading_nodes_flag:
                if line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("DEMAND_SECTION"):
                    reading_nodes_flag = False
                else:
                    line_contents = [x.strip() for x in line.replace(" ", "\t").split("\t") if x.strip()]
                    _, x, y = line_contents
                    depot_nodes.append([float(x), float(y)])

            # read demands coordinates
            elif read_demands_flag:
                if line.startswith("DEMAND_SECTION"):
                    pass
                elif line.startswith("DEPOT_SECTION"):
                    read_demands_flag = False
                else:
                    line_contents = [x.strip() for x in line.replace(" ", "\t").split("\t") if x.strip()]
                    demands.append(int(line_contents[1]))

            # read depot position
            elif read_depot_flag:
                if line.startswith("DEPOT_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    read_depot_flag = False
                else:
                    line_contents = [x.strip() for x in line.replace(" ", "\t").split("\t") if x.strip()]
                    depot_check.append(int(line_contents[0]))
            line = read_file.readline()

    depot = depot_nodes[0]
    nodes = depot_nodes[1:]
    demands = demands[1:]

    return depot, nodes, demands, properties


def choose_bsz(size):
    if size<=200:
        return 64
    elif size<=1000:
        return 32
    elif size<=5000:
        return 16
    else:
        return 4


def load_tsplib_file(root, tsplib_name):
    tsplib_dir = "tsplib"
    file_name = f"{tsplib_name}.tsp"
    file_path = root.joinpath(tsplib_dir).joinpath(file_name)
    instance, name = read_tsplib_file(file_path)

    instance = torch.tensor(instance)
    return instance, name

def load_cvrplib_file(root, cvrplib_name):
    cvrplib_dir = "vrplib"
    file_name = f"{cvrplib_name}.vrp"
    file_path = root.joinpath(cvrplib_dir).joinpath(file_name)
    depot, nodes, demands, properties = read_cvrplib_file(file_path)

    depot = torch.tensor(depot)
    nodes = torch.tensor(nodes)
    demands = torch.tensor(demands)
    capacity = torch.tensor(int(properties["CAPACITY"]))
    name = properties["NAME"]
    return depot, nodes, demands, capacity, name

def avg_list(list_object):
    return sum(list_object) / len(list_object) if len(list_object) > 0 else 0


def normalize_tsp_to_unit_board(tsp_instance):
    """
    normalize a tsp instance to a [0, 1]^2 unit board, prefer to have points on both x=0 and y=0
    :param tsp_instance: a (tsp_size, 2) tensor
    :return: a (tsp_size, 2) tensor, a normalized tsp instance
    """
    normalized_instance = tsp_instance.clone()
    normalization_factor = (normalized_instance.max(dim=0).values - normalized_instance.min(dim=0).values).max()
    normalized_instance = (normalized_instance - normalized_instance.min(dim=0).values) / normalization_factor
    return normalized_instance


def normalize_nodes_to_unit_board(nodes):
    return normalize_tsp_to_unit_board(nodes)


def get_dist_matrix(instance):
    size = instance.shape[0]
    x = instance.unsqueeze(0).repeat((size, 1, 1))
    y = instance.unsqueeze(1).repeat((1, size, 1))
    return torch.norm(x - y, p=2, dim=-1)


def calculate_tour_length_by_dist_matrix(dist_matrix, tours):
    # useful to evaluate one/multiple solutions on one (not-extremely-huge) instance
    if tours.dim() == 1:
        tours = tours.unsqueeze(0)
    tour_shifts = torch.roll(tours, shifts=-1, dims=1)
    tour_lens = dist_matrix[tours, tour_shifts].sum(dim=1)
    return tour_lens


def calculate_tour_length_by_instances(instances, tours):
    # evaluate a batch of solutions
    pass


def check_cvrp_solution_validity(tour, demands, size, capacity):
    tour = tour.tolist()
    demands = demands.tolist()
    visited = []
    remaining = capacity.item()

    for i in range(len(tour)):
        if tour[i] == 0:
            remaining = capacity.item()
            continue

        if tour[i] in visited:
            return False

        visited.append(tour[i])
        remaining -= demands[tour[i] - 1]
        if remaining < 0:
            return False

    if len(visited) != size:
        return False

    return True


def load_tsp_instances(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} does not exist.")

    tsp_instance_list = []
    opt_tour_list = []
    opt_len_list = []

    with open(path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            line_contents = line.strip().split(" | ")
            tsp_instance_string, opt_tour_string, opt_len_string = line_contents

            tsp_instance = []
            for node_string in tsp_instance_string.split(" "):
                node = node_string.split(",")
                tsp_instance.append([float(node[0]), float(node[1])])
            tsp_instance_list.append(np.array(tsp_instance))

            opt_tour = [int(x) for x in opt_tour_string.split(" ")]
            opt_tour_list.append(np.array(opt_tour))

            opt_len_list.append(float(opt_len_string))

    tsp_instances = np.array(tsp_instance_list)
    opt_tours = np.array(opt_tour_list)
    opt_lens = np.array(opt_len_list)

    num = tsp_instances.shape[0]
    size = tsp_instances.shape[1]

    return tsp_instances, opt_tours, opt_lens, size, num


#####################
# TSPLIB information
#####################

tsplib_collections = {
    'eil51': 426,
    'berlin52': 7542,
    'st70': 675,
    'pr76': 108159,
    'eil76': 538,
    'rat99': 1211,
    'kroA100': 21282,
    'kroE100': 22068,
    'kroB100': 22141,
    'rd100': 7910,
    'kroD100': 21294,
    'kroC100': 20749,
    'eil101': 629,
    'lin105': 14379,
    'pr107': 44303,
    'pr124': 59030,
    'bier127': 118282,
    'ch130': 6110,
    'pr136': 96772,
    'pr144': 58537,
    'kroA150': 26524,
    'kroB150': 26130,
    'ch150': 6528,
    'pr152': 73682,
    'u159': 42080,
    'rat195': 2323,
    'd198': 15780,
    'kroA200': 29368,
    'kroB200': 29437,
    'tsp225': 3916,
    'ts225': 126643,
    'pr226': 80369,
    'gil262': 2378,
    'pr264': 49135,
    'a280': 2579,
    'pr299': 48191,
    'lin318': 42029,
    'rd400': 15281,
    'fl417': 11861,
    'pr439': 107217,
    'pcb442': 50778,
    'd493': 35002,
    'u574': 36905,
    'rat575': 6773,
    'p654': 34643,
    'd657': 48912,
    'u724': 41910,
    'rat783': 8806,
    'pr1002': 259045,
    'u1060': 224094,
    'vm1084': 239297,
    'pcb1173': 56892,
    'd1291': 50801,
    'rl1304': 252948,
    'rl1323': 270199,
    'nrw1379': 56638,
    'fl1400': 20127,
    'u1432': 152970,
    'fl1577': 22249,
    'd1655': 62128,
    'vm1748': 336556,
    'u1817': 57201,
    'rl1889': 316536,
    'd2103': 80450,
    'u2152': 64253,
    'u2319': 234256,
    'pr2392': 378032,
    'pcb3038': 137694,
    'fl3795': 28772,
    'fnl4461': 182566,
    'rl5915': 565530,
    'rl5934': 556045,
    'rl11849': 923288,
    'usa13509': 19982859,
    'brd14051': 469385,
    'd15112': 1573084,
    'd18512': 645238
}


cvrplib_collections = {
    "X-n101-k25": 27591,
    "X-n106-k14": 26362,
    "X-n110-k13": 14971,
    "X-n115-k10": 12747,
    "X-n120-k6": 13332,
    "X-n125-k30": 55539,
    "X-n129-k18": 28940,
    "X-n134-k13": 10916,
    "X-n139-k10": 13590,
    "X-n143-k7": 15700,
    "X-n148-k46": 43448,
    "X-n153-k22": 21220,
    "X-n157-k13": 16876,
    "X-n162-k11": 14138,
    "X-n167-k10": 20557,
    "X-n172-k51": 45607,
    "X-n176-k26": 47812,
    "X-n181-k23": 25569,
    "X-n186-k15": 24145,
    "X-n190-k8": 16980,
    "X-n195-k51": 44225,
    "X-n200-k36": 58578,
    "X-n204-k19": 19565,
    "X-n209-k16": 30656,
    "X-n214-k11": 10856,
    "X-n219-k73": 117595,
    "X-n223-k34": 40437,
    "X-n228-k23": 25742,
    "X-n233-k16": 19230,
    "X-n237-k14": 27042,
    "X-n242-k48": 82751,
    "X-n247-k50": 37274,
    "X-n251-k28": 38684,
    "X-n256-k16": 18839,
    "X-n261-k13": 26558,
    "X-n266-k58": 75478,
    "X-n270-k35": 35291,
    "X-n275-k28": 21245,
    "X-n280-k17": 33503,
    "X-n284-k15": 20226,
    "X-n289-k60": 95151,
    "X-n294-k50": 47161,
    "X-n298-k31": 34231,
    "X-n303-k21": 21736,
    "X-n308-k13": 25859,
    "X-n313-k71": 94043,
    "X-n317-k53": 78355,
    "X-n322-k28": 29834,
    "X-n327-k20": 27532,
    "X-n331-k15": 31102,
    "X-n336-k84": 139111,
    "X-n344-k43": 42050,
    "X-n351-k40": 25896,
    "X-n359-k29": 51505,
    "X-n367-k17": 22814,
    "X-n376-k94": 147713,
    "X-n384-k52": 65940,
    "X-n393-k38": 38260,
    "X-n401-k29": 66154,
    "X-n411-k19": 19712,
    "X-n420-k130": 107798,
    "X-n429-k61": 65449,
    "X-n439-k37": 36391,
    "X-n449-k29": 55233,
    "X-n459-k26": 24139,
    "X-n469-k138": 221824,
    "X-n480-k70": 89449,
    "X-n491-k59": 66483,
    "X-n502-k39": 69226,
    "X-n513-k21": 24201,
    "X-n524-k153": 154593,
    "X-n536-k96": 94846,
    "X-n548-k50": 86700,
    "X-n561-k42": 42717,
    "X-n573-k30": 50673,
    "X-n586-k159": 190316,
    "X-n599-k92": 108451,
    "X-n613-k62": 59535,
    "X-n627-k43": 62164,
    "X-n641-k35": 63684,
    "X-n655-k131": 106780,
    "X-n670-k130": 146332,
    "X-n685-k75": 68205,
    "X-n701-k44": 81923,
    "X-n716-k35": 43373,
    "X-n733-k159": 136187,
    "X-n749-k98": 77269,
    "X-n766-k71": 114417,
    "X-n783-k48": 72386,
    "X-n801-k40": 73311,
    "X-n819-k171": 158121,
    "X-n837-k142": 193737,
    "X-n856-k95": 88965,
    "X-n876-k59": 99299,
    "X-n895-k37": 53860,
    "X-n916-k207": 329179,
    "X-n936-k151": 132715,
    "X-n957-k87": 85465,
    "X-n979-k58": 118976,
    "X-n1001-k43": 72355,
}
