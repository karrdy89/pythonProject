import ray
from statics import Actors


def split_chunk(chunk: list[tuple], chunk_index: int, key_index: int, x_index: list[int], is_buffer_end: bool, act):
    split = []
    temp = []
    before_key = None
    for data in chunk:
        cur_key = data[key_index]
        for i in x_index:
            temp.append(data[i])
        if before_key != data[key_index]:
            split.append([cur_key, temp])
            temp = []
        before_key = data[key_index]
    split.append(chunk_index)
    split.append(is_buffer_end)
    result = ray.get(act.set_split.remote(data=split))
    if result != 0:
        act.fault_handle.remote(msg="failed to send split result")


def make_dataset(datas: list, labels: list[str], act):
    max_len = 0
    classes = {}
    for label in labels:
        classes[label] = 0
    dataset = []
    information = {}
    for data in datas:
        cust_id = data[0]
        features = data[1]
        matched_idx_before = 0
        matched = False
        for i, feature in enumerate(features):
            if matched:
                matched = False
                matched_idx_before += 1
                continue
            if feature in labels:
                matched_idx_current = i
                if matched_idx_current <= matched_idx_before + 1:
                    matched_idx_before = matched_idx_current
                else:
                    dataset.append([cust_id] + features[matched_idx_before:matched_idx_current] + [feature])
                    if max_len < (matched_idx_current - matched_idx_before):
                        max_len = matched_idx_current - matched_idx_before
                    classes[feature] += 1
                    matched_idx_before = matched_idx_current
                matched = True

    information["max_len"] = max_len
    information["classes"] = classes
    result = ray.get(act.set_dataset.remote(data=dataset, information=information))
    if result != 0:
        act.fault_handle.remote(msg="failed to send make dataset result")

