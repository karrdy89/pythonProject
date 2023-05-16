import ray
import random


def split_chunk(chunk: list[tuple], chunk_index: int, key_index: int, x_index: list[int], act):
    try:
        if len(chunk) == 0:
            return
        temp_feature = []
        split = []
        left_over = {}
        before_key = None
        for data in chunk:
            cur_key = data[key_index]
            if before_key is None or before_key == cur_key:
                for i in x_index:
                    temp_feature.append(data[i])
            elif before_key != cur_key:
                split.append([cur_key] + temp_feature)
                temp_feature = []
                for i in x_index:
                    temp_feature.append(data[i])
            before_key = cur_key
        if len(split) <= 2:
            left_over[chunk_index] = [split]
            act.set_left_over.remote(data=left_over)
        else:
            left_over[chunk_index] = [split[0], split[-1]]
            act.set_left_over.remote(data=left_over)
            act.set_split.remote(data=split)
    except Exception as e:
        act.fault_handle.remote(msg="failed to split:" + e.__str__())
        raise(e.__str__())


def make_dataset(datas: list, labels: list[str], len_limit: int, is_operation_end: bool, act):
    if is_operation_end:
        act.set_dataset.remote()
        return
    classes = {}
    label_dataset = {}
    for label in labels:
        classes[label] = 0
        label_dataset[label] = []
    try:
        for data in datas:
            cust_id = data[0]
            features = data[1:]
            for idx, feature in enumerate(features):
                if feature in labels:
                    if idx > 2:
                        if idx >= len_limit:
                            label_dataset[feature].append([cust_id] + features[idx-len_limit:idx] + [feature])
                        else:
                            label_dataset[feature].append([cust_id] + features[0:idx] + [feature])
                        classes[feature] += 1
            unk_sample_count = 0
            for idx, r_feature in enumerate(reversed(features)):
                if unk_sample_count >= 1:
                    continue
                if random.random() > 0.3:
                    continue
                if r_feature not in labels:
                    r_idx = len(features) - idx
                    if r_idx > 3:
                        if r_idx >= len_limit:
                            label_dataset["UNK"].append([cust_id] + features[r_idx - len_limit:r_idx] + ["UNK"])
                        else:
                            label_dataset["UNK"].append([cust_id] + features[0:r_idx] + ["UNK"])
                        classes["UNK"] += 1
                        unk_sample_count += 1
        committable = ray.get(act.get_committable.remote(classes=classes))
        for label, num in committable.items():
            if num > 0:
                label_dataset[label] = label_dataset[label][:num]
            else:
                label_dataset[label] = []
        act.set_dataset.remote(data=label_dataset, information=committable)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        act.fault_handle.remote(msg="failed to make_dataset:" + e.__str__())
        raise (e.__str__())

