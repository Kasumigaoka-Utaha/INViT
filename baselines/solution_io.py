# this file implements several functions to write/read solutions to/from system files

import torch


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

    tours = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tour_storage], batch_first=True, padding_value=0)
    tour_lens = torch.tensor(tour_len_storage)
    time_consumptions = torch.tensor(ellapsed_time_storage)
    return tours, tour_lens, time_consumptions
