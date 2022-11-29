import ray


def split_chunk(chunk: list[tuple], chunk_index: int, key_index: int, x_index: list[int], act):
    split = []
    temp = []
    before_key = None
    try:
        for data in chunk:
            cur_key = data[key_index]
            for i in x_index:
                temp.append(data[i])
            if before_key != data[key_index]:
                split.append([cur_key, temp])
                temp = []
            before_key = data[key_index]
    except Exception as e:
        act.fault_handle.remote(msg="failed to split:" + e.__str__())
        raise(e.__str__())
    cases = len(split) - 2
    if cases < 0:
        cases = 0
    split.append(chunk_index)
    result = ray.get(act.set_split.remote(data=split, cases=cases))
    if result != 0:
        act.fault_handle.remote(msg="failed to send split result")


def make_dataset(datas: list, labels: list[str], len_limit: int, act):
    max_len = 0
    classes = {}
    for label in labels:
        classes[label] = 0
    dataset = []
    information = {}
    try:
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
                        tmp_features = features[matched_idx_before:matched_idx_current]
                        if len(tmp_features) >= len_limit:
                            tmp_features = tmp_features[-len_limit:]
                        dataset.append([cust_id] + tmp_features + [feature])
                        if max_len < (matched_idx_current - matched_idx_before):
                            max_len = matched_idx_current - matched_idx_before
                            if max_len >= len_limit:
                                max_len = len_limit
                        classes[feature] += 1
                        matched_idx_before = matched_idx_current
                    matched = True
    except Exception as e:
        act.fault_handle.remote(msg="failed to make_dataset:" + e.__str__())
        raise (e.__str__())
    information["max_len"] = max_len
    information["classes"] = classes
    result = ray.get(act.set_dataset.remote(data=dataset, information=information))
    if result != 0:
        act.fault_handle.remote(msg="failed to send make dataset result")
