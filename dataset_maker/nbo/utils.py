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
    split.append(chunk_index)
    result = ray.get(act.set_split.remote(data=split))
    if result != 0:
        act.fault_handle.remote(msg="failed to send split result")


def make_dataset(datas: list, labels: list[str], len_limit: int, label_ratio: dict, is_operation_end: bool, act):
    if is_operation_end:
        return
    max_len = 0
    classes = {}
    for label in labels:
        classes[label] = 0
    unk = None
    if "UNK" in labels:
        unk = "UNK"
    label_dataset = {}
    for label in labels:
        label_dataset[label] = []
    information = {}
    try:
        for data in datas:
            cust_id = data[0]
            features = data[1]
            len_features = len(features)
            matched_idx_before = 0
            unk_start_idx = 0
            matched = False
            if not labels:
                break
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
                        label_dataset[feature].append([cust_id] + tmp_features + [feature])
                        if max_len < (matched_idx_current - matched_idx_before):
                            max_len = matched_idx_current - matched_idx_before
                            if max_len >= len_limit:
                                max_len = len_limit
                        classes[feature] += 1
                        if classes[feature] >= label_ratio[feature]:
                            labels.remove(features)
                        matched_idx_before = matched_idx_current
                    matched = True
                else:
                    if unk in labels:
                        if len(label_dataset[unk]) > 25:
                            continue
                        if len_features-1 > i+2:
                            if not features[i+1] in labels:
                                if (i - unk_start_idx) < len_limit:
                                    label_dataset[unk].append([cust_id] + features[unk_start_idx:i+1] + [unk])
                                    classes[unk] += 1
                                    if classes[unk] >= label_ratio[unk]:
                                        labels.remove(unk)
                            else:
                                unk_start_idx = i+2
    except Exception as e:
        act.fault_handle.remote(msg="failed to make_dataset:" + e.__str__())
        raise (e.__str__())

    res = ray.get(act.is_committable.remote(labels_num=classes))
    information["classes"] = classes
    if not res:
        result = ray.get(act.set_dataset.remote(data=label_dataset, information=information, f_end=True))
        if result != 0:
            act.fault_handle.remote(msg="failed to send make dataset result")
    else:
        result = ray.get(act.set_dataset.remote(data=label_dataset, information=information))
        if result != 0:
            act.fault_handle.remote(msg="failed to send make dataset result")
