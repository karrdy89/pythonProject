from datetime import datetime

import ray


def split_chunk(chunk: list[tuple], chunk_index: int, key_index: int, x_index: list[int], condition_index: int,
                condition: datetime, label_list: list, len_limit: int, is_leftover_handling: bool,  act):
    try:
        if len(chunk) == 0:
            return
        temp_feature = []

        classes = {}
        label_dataset = {}
        for label in label_list:
            classes[label] = 0
            label_dataset[label] = []

        if not is_leftover_handling:
            left_over = {}

            # trim first
            before_key = None
            trim_idx = 0
            for idx, data in enumerate(chunk):
                cur_key = data[key_index]
                if before_key is None or before_key == cur_key:
                    pass
                else:
                    trim_idx = idx
            if len(chunk[0:trim_idx]) > 1:
                left_over[chunk_index] = [chunk[0:trim_idx]]
            chunk = chunk[trim_idx:]

            # trim last
            before_key = None
            trim_idx = 0
            for idx, data in enumerate(reversed(chunk)):
                cur_key = data[key_index]
                if before_key is None or before_key == cur_key:
                    pass
                else:
                    trim_idx = idx
            if len(chunk[len(chunk)-trim_idx:]) > 1:
                left_over[chunk_index].append(chunk[len(chunk)-trim_idx:])
            chunk = chunk[:len(chunk)-trim_idx]
            act.set_left_over.remote(data=left_over)

        # process data
        start_idx = 0
        before_key = None
        for idx, data in enumerate(chunk):
            cur_key = data[key_index]
            if before_key is None or before_key == cur_key:
                if condition >= datetime.strptime(data[condition_index][:8], "%Y%m%d"):
                    label = data[x_index[0]]
                    if label in label_list:
                        matched_chunks = chunk[start_idx:idx+1]
                        matched_len = len(matched_chunks)
                        if matched_len > 1:
                            if matched_len > len_limit:
                                matched_chunks = chunk[idx - len_limit:idx+1]
                            for matched_data in matched_chunks:
                                temp_feature.append(matched_data[x_index[0]])
                            label_dataset[label].append([data[key_index]] + temp_feature)
                            classes[label] = classes[label] + 1
                            temp_feature = []
            else:
                for i1_idx, i1_data in enumerate(reversed(chunk[start_idx:idx])):
                    label = i1_data[x_index[0]]
                    if label not in label_list:
                        matched_chunks = chunk[start_idx:idx-i1_idx]
                        matched_len = len(matched_chunks)
                        if matched_len > 1:
                            if matched_len > len_limit:
                                matched_chunks = matched_chunks[matched_len - len_limit:]
                            for matched_data in matched_chunks:
                                temp_feature.append(matched_data[x_index[0]])
                            label_dataset["UNK"].append([data[key_index]] + temp_feature + ["UNK"])
                            classes["UNK"] = classes["UNK"] + 1
                            temp_feature = []
                            break
                start_idx = idx
            before_key = cur_key

        # commit
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
        act.fault_handle.remote(msg="failed to split:" + e.__str__())
        raise(e.__str__())
