import os
import pickle
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from time import sleep, time


class DataSet(object):
    char_dict = None
    tag_dict = None
    relation_dict = None
    tag_valid_combined = None
    tag_valid_combined_reversed = None
    _D = None
    _T1 = None
    _T2 = None
    _A = None
    stop_chars = set(' \n0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __init__(self, data_path,
                 shuffle=True,
                 test_size=43,
                 split_size=400,
                 max_rel_distance=200,
                 min_rel_distance=0,
                 max_split_rel_count=150,
                 all_regular_rel_count=779,
                 default_build_size=24,
                 capacity=150,
                 is_test_all_regular_rel=True,
                 worker_count=24,
                 enabled_pre_build=True,
                 seq_mask_rate=0.1,
                 is_train_random_pad_seq=True,
                 return_entity_detail_id=False):
        self.split_size = split_size
        self.max_split_rel_count = max_split_rel_count
        self.max_rel_distance = max_rel_distance  # max包含
        self.min_rel_distance = min_rel_distance  # min不包含
        self.all_regular_rel_count = all_regular_rel_count
        self.enabled_pre_build = enabled_pre_build
        self.is_test_all_regular_rel = is_test_all_regular_rel
        self.seq_mask_rate = seq_mask_rate
        self.is_train_random_pad_seq = is_train_random_pad_seq

        dict_path = os.path.dirname(os.path.dirname(os.path.realpath(data_path)))
        char_dict_file = os.path.join(dict_path, "f_used_chars.dic")
        tag_dict_file = os.path.join(dict_path, "f_used_tags.dic")
        rel_dict_file = os.path.join(dict_path, "f_used_rels.dic")

        if DataSet.char_dict is None:
            if os.path.isfile(char_dict_file):
                with open(char_dict_file, 'rb') as f_dict:
                    DataSet.char_dict = pickle.load(f_dict)
            else:
                DataSet.char_dict = {'__nil__': 0}
                for i in range(1, 16):
                    DataSet.char_dict['__tag' + str(i) + '__'] = i
        if DataSet.tag_dict is None:
            if os.path.isfile(tag_dict_file):
                with open(tag_dict_file, 'rb') as f_dict:
                    DataSet.tag_dict = pickle.load(f_dict)
            else:
                DataSet.tag_dict = {'O': 0}
        if DataSet.relation_dict is None:
            if os.path.isfile(rel_dict_file):
                with open(rel_dict_file, 'rb') as f_dict:
                    DataSet.relation_dict = pickle.load(f_dict)
            else:
                DataSet.relation_dict = {'none': 0}

        if DataSet.tag_valid_combined is None:
            DataSet.tag_valid_combined = {
                DataSet.convert_to_tag_index('Disease'): {
                    DataSet.convert_to_tag_index('Test'): "Test_Disease",
                    DataSet.convert_to_tag_index('Symptom'): "Symptom_Disease",
                    DataSet.convert_to_tag_index('Anatomy'): "Anatomy_Disease",
                    DataSet.convert_to_tag_index('Drug'): "Drug_Disease",
                    DataSet.convert_to_tag_index('Treatment'): "Treatment_Disease"
                },
                DataSet.convert_to_tag_index('Drug'): {
                    DataSet.convert_to_tag_index('SideEff'): "SideEff-Drug",
                    DataSet.convert_to_tag_index('Frequency'): "Frequency_Drug",
                    DataSet.convert_to_tag_index('Amount'): "Amount_Drug",
                    DataSet.convert_to_tag_index('Method'): "Method_Drug",
                    DataSet.convert_to_tag_index('Duration'): "Duration_Drug"
                }}

            DataSet.tag_valid_combined_reversed = {
                DataSet.convert_to_tag_index('Test'): {
                    DataSet.convert_to_tag_index('Disease'): "Test_Disease",
                },
                DataSet.convert_to_tag_index('Symptom'): {
                    DataSet.convert_to_tag_index('Disease'): "Symptom_Disease",
                },
                DataSet.convert_to_tag_index('Anatomy'): {
                    DataSet.convert_to_tag_index('Disease'): "Anatomy_Disease",
                },
                DataSet.convert_to_tag_index('Drug'): {
                    DataSet.convert_to_tag_index('Disease'): "Drug_Disease",
                },
                DataSet.convert_to_tag_index('Treatment'): {
                    DataSet.convert_to_tag_index('Disease'): "Treatment_Disease"
                },
                DataSet.convert_to_tag_index('SideEff'): {
                    DataSet.convert_to_tag_index('Drug'): "SideEff-Drug",
                },
                DataSet.convert_to_tag_index('Frequency'): {
                    DataSet.convert_to_tag_index('Drug'): "Frequency_Drug",
                },
                DataSet.convert_to_tag_index('Amount'): {
                    DataSet.convert_to_tag_index('Drug'): "Amount_Drug",
                },
                DataSet.convert_to_tag_index('Method'): {
                    DataSet.convert_to_tag_index('Drug'): "Method_Drug",
                },
                DataSet.convert_to_tag_index('Duration'): {
                    DataSet.convert_to_tag_index('Drug'): "Duration_Drug"
                }}

        DataSet._D = {DataSet.convert_to_tag_index('Disease')}
        DataSet._T1 = {DataSet.convert_to_tag_index('Test'),
                       DataSet.convert_to_tag_index('Symptom'),
                       DataSet.convert_to_tag_index('Anatomy'),
                       DataSet.convert_to_tag_index('Treatment')}
        DataSet._T2 = {DataSet.convert_to_tag_index('Drug')}
        DataSet._A = {DataSet.convert_to_tag_index('SideEff'),
                      DataSet.convert_to_tag_index('Frequency'),
                      DataSet.convert_to_tag_index('Amount'),
                      DataSet.convert_to_tag_index('Method'),
                      DataSet.convert_to_tag_index('Duration')}

        sample_files = []
        check_dirs = [data_path]
        while len(check_dirs) > 0:
            for root, dirs, files in os.walk(check_dirs.pop()):
                for filename in files:
                    if ".txt" == filename[-4:]:
                        file_path = os.path.join(root, filename)

                        ann_file = file_path[:-4] + ".ann"
                        if os.path.isfile(ann_file):
                            sample_files.append((file_path, 0, filename))
                        else:
                            sample_files.append((file_path, 1, filename))
                for d in dirs:
                    check_dirs.append(os.path.join(root, d))

        if len(sample_files) == 0:
            raise Exception('No data')

        sample_files = np.array(sample_files)
        if shuffle:
            perm2 = np.arange(len(sample_files))
            if not isinstance(shuffle, bool):
                np.random.seed(shuffle)
            np.random.shuffle(perm2)
            sample_files = sample_files[perm2]

        self._count = len(sample_files)
        train_files = sample_files[sample_files[:, 1] == '0']
        test_files = sample_files[sample_files[:, 1] == '1']

        if test_size <= 1:
            test_count = int(self._count * test_size)
        else:
            test_count = int(test_size)

        if test_count > len(test_files):
            split_len = test_count - len(test_files)
            if split_len >= len(train_files):
                test_files = np.concatenate([test_files, train_files], 0)
                self._test_count = self._count
            else:
                test_files = np.concatenate([test_files, train_files[:split_len]], 0)
                self._test_count = test_count
                train_files = train_files[split_len:]
        else:
            self._test_count = len(test_files)

        self._test_files = test_files
        self._train_count = self._count - self._test_count
        self._train_files = train_files

        self._d_train_samples = []
        max_seq_length = 0
        for file_path in train_files[:, 0]:
            seq, relations, entities, entity_positions, raw_relations = DataSet.load_file(
                file_path,
                return_entity_detail_id=return_entity_detail_id)
            if max_seq_length < len(seq):
                max_seq_length = len(seq)
            self._d_train_samples.append((seq, relations, entities, entity_positions, raw_relations))

        self._d_test_samples = []
        self._d_test_built_samples = [None for _ in range(self._test_count)]
        self._d_flod_test_built_samples = [None for _ in range(self._train_count)]
        for file_path in test_files[:, 0]:
            seq, relations, entities, entity_positions, raw_relations = DataSet.load_file(
                file_path,
                return_entity_detail_id=return_entity_detail_id)
            if max_seq_length < len(seq):
                max_seq_length = len(seq)
            self._d_test_samples.append((seq, relations, entities, entity_positions, raw_relations))

        with open(tag_dict_file, 'wb') as f_dict:
            pickle.dump(DataSet.tag_dict, f_dict)
        with open(char_dict_file, 'wb') as f_dict:
            pickle.dump(DataSet.char_dict, f_dict)
        with open(rel_dict_file, 'wb') as f_dict:
            pickle.dump(DataSet.relation_dict, f_dict)

        self.max_seq_length = max_seq_length
        self._train_offset = 0
        self._test_offset = 0

        self._cached_regular_combine = {}
        self._cached_free_combine = {}
        self._cached_all_combine = {}

    @staticmethod
    def convert_to_token_index(token):
        if token not in DataSet.char_dict:
            inx = len(DataSet.char_dict)

            DataSet.char_dict[token] = inx
        else:
            inx = DataSet.char_dict[token]

        return inx

    @staticmethod
    def get_token_index(label_word, label_class, return_entity_detail_id=False):
        if label_class > 0:
            if not return_entity_detail_id:
                return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
            else:
                if "Amount" in DataSet.tag_dict and label_class == DataSet.tag_dict["Amount"]:
                    return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
                elif "Test_Value" in DataSet.tag_dict and label_class == DataSet.tag_dict["Test_Value"]:
                    return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
                elif "Frequency" in DataSet.tag_dict and label_class == DataSet.tag_dict["Frequency"]:
                    return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
                elif "Duration" in DataSet.tag_dict and label_class == DataSet.tag_dict["Duration"]:
                    return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
                elif "Level" in DataSet.tag_dict and label_class == DataSet.tag_dict["Level"]:
                    return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
                elif "Operation" in DataSet.tag_dict and label_class == DataSet.tag_dict["Operation"]:
                    return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
                elif "Reason" in DataSet.tag_dict and label_class == DataSet.tag_dict["Operation"]:
                    return DataSet.convert_to_token_index('__tag' + str(label_class) + '__')
                else:
                    return DataSet.convert_to_token_index(label_word)
        else:
            return DataSet.convert_to_token_index(label_word)

    @staticmethod
    def convert_to_seq_indices(chars, labels, return_entity_detail_id=False):
        # labels 0 为实体类型，1 为实体ID
        indices = []
        last_entity_id = 0
        label_start = -1
        entity_positions = {}
        for i, c in enumerate(chars):
            if labels[i, 1] != last_entity_id and last_entity_id > 0:
                label_word = chars[label_start:i]
                label_word = label_word.replace('\n', '')
                entity_positions[labels[i - 1, 1]] = len(indices)
                indices.append((DataSet.get_token_index(label_word, labels[i - 1, 0],
                                                        return_entity_detail_id=return_entity_detail_id),
                                labels[i - 1, 0],
                                labels[i - 1, 1]))
                label_start = -1
                last_entity_id = 0

            if labels[i, 1] == 0:
                if c not in DataSet.stop_chars:
                    indices.append((DataSet.convert_to_token_index(c), 0, 0))
            elif last_entity_id == 0:
                label_start = i
                last_entity_id = labels[i, 1]

        if last_entity_id > 0:
            label_word = chars[label_start:]
            label_word = label_word.replace('\n', '')
            entity_positions[labels[-1, 1]] = len(indices)
            indices.append((DataSet.get_token_index(label_word, labels[-1, 0],
                                                    return_entity_detail_id=return_entity_detail_id),
                            labels[-1, 0],
                            labels[-1, 1]))

        # 返回序列： 第0列 为字或实体内容索引，第1列 为实体类型，第2列 为实体ID
        return indices, entity_positions

    @staticmethod
    def convert_to_tag_index(tag):
        if tag in DataSet.tag_dict:
            return DataSet.tag_dict[tag]
        else:
            inx = len(DataSet.tag_dict)
            DataSet.tag_dict[tag] = inx
            return inx

    @staticmethod
    def convert_to_relation_index(relation):
        if relation in DataSet.relation_dict:
            return DataSet.relation_dict[relation]
        else:
            inx = len(DataSet.relation_dict)
            DataSet.relation_dict[relation] = inx
            return inx

    @staticmethod
    def convert_to_entity_index(entity_id, entity_dict, is_create=True):
        if entity_id in entity_dict:
            return entity_dict[entity_id]
        elif is_create:
            inx = len(entity_dict)
            entity_dict[entity_id] = inx
            return inx
        else:
            return -1

    @staticmethod
    def tag_class_count():
        return len(DataSet.tag_dict)

    @staticmethod
    def load_file(filename, return_entity_detail_id=False):
        ann_file = filename[:-4] + ".ann"
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        labels = np.zeros((len(text), 2), dtype=np.int32)
        relations = []
        raw_relations = []
        entities = {"O": 0}
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                l_arr = line.split('\t')
                if line[:1] == "T":
                    e_arr = l_arr[1].split(' ')
                    tag = DataSet.convert_to_tag_index(e_arr[0])

                    s = int(e_arr[1])
                    e = int(e_arr[-1])
                    if np.sum(labels[s:e, 0]) == 0:
                        # 防止多重标注情况
                        labels[s:e, 0] = tag
                        labels[s:e, 1] = DataSet.convert_to_entity_index(l_arr[0], entities)
                elif line[:1] == "R":
                    r_arr = l_arr[1].split(' ')

                    if r_arr[1][:5] != "Arg1:":
                        continue
                    elif r_arr[2][:5] != "Arg2:":
                        continue
                    else:
                        e1_id = r_arr[1][5:]
                        e2_id = r_arr[2][5:].rstrip()

                        raw_relations.append((r_arr[0], e1_id, e2_id))

                        e1_inx = DataSet.convert_to_entity_index(e1_id, entities, False)
                        e2_inx = DataSet.convert_to_entity_index(e2_id, entities, False)

                        if e1_inx < 0:
                            continue

                        if e2_inx < 0:
                            continue

                        relations.append((DataSet.convert_to_relation_index(r_arr[0]), e1_inx, e2_inx))

        seq, entity_positions = DataSet.convert_to_seq_indices(text, labels,
                                                               return_entity_detail_id=return_entity_detail_id)

        # relation中加上实体token在序列中的位置
        for i, r in enumerate(relations):
            e1_pos = entity_positions[r[1]]
            e2_pos = entity_positions[r[2]]
            e1_tag = seq[e1_pos][1]
            e2_tag = seq[e2_pos][1]
            relations[i] = (r[0], e1_pos, e2_pos, r[1], r[2],
                            DataSet.convert_to_relation_index(DataSet.tag_valid_combined[e2_tag][e1_tag])
                            if e2_tag in DataSet.tag_valid_combined and e1_tag in DataSet.tag_valid_combined[e2_tag]
                            else 0)

        return seq, relations, entities, entity_positions, raw_relations

    def build_combine_regular(self, entities, positive_relations=None, skip_positive_relations=False):
        # entities  (token id，token type，entity id, entity pos)
        # positive_relations  (entity1 id, entity2 id) set
        # 与build_combine相比，优化在不需要生成全组合，只要生成有效的组合

        if positive_relations is None:
            positive_relations = {}

        # 每种关系的mask
        vp = dict([(c[0], np.array([p[1] in c[1] for p in entities]))
                   for c in DataSet.tag_valid_combined.items()])

        ret = []
        for p in entities:
            if p[1] in DataSet.tag_valid_combined:
                vi = (entities[:, 3] <= p[3] + self.max_rel_distance) & \
                     (entities[:, 3] >= p[3] - self.max_rel_distance) & \
                     (abs(entities[:, 3] - p[3]) > self.min_rel_distance) & \
                     (vp[p[1]])
                if not skip_positive_relations:
                    ret.extend([(positive_relations[(tp[2], p[2])] if (tp[2], p[2]) in positive_relations else 0,
                                 tp[3], p[3],
                                 DataSet.convert_to_relation_index(DataSet.tag_valid_combined[p[1]][tp[1]]),
                                 tp[2], p[2])
                                for tp in entities[vi]])
                else:
                    ret.extend([(0,
                                 tp[3], p[3],
                                 DataSet.convert_to_relation_index(DataSet.tag_valid_combined[p[1]][tp[1]]),
                                 tp[2], p[2])
                                for tp in entities[vi]
                                if (tp[2], p[2]) not in positive_relations])

        # 组合 (real relation type, entity1 index, entity2 index, regular relation type, entity1 id, entity2 id)
        return ret

    def build_combine_strong_regular(self, entities, positive_relations=None, skip_positive_relations=False):
        # entities  (token id，token type，entity id, entity pos)
        # positive_relations  (entity1 id, entity2 id) set
        # 与build_combine相比，优化在不需要生成全组合，只要生成有效的组合

        if positive_relations is None:
            positive_relations = {}

        ret = []
        sorted_entities = sorted(entities, key=lambda x: x[3])
        for i, p in enumerate(sorted_entities):
            if p[1] in DataSet.tag_valid_combined:
                for tt in DataSet.tag_valid_combined[p[1]]:
                    # 向前寻找，找到目标类型之后第一次遇到当前类型结束
                    flag = False
                    for j in range(0, i):
                        te = sorted_entities[i - j - 1]
                        if te[3] >= p[3] - self.min_rel_distance:
                            continue
                        if te[3] < p[3] - self.max_rel_distance:
                            # 超过设置范围的忽略
                            break
                        if te[1] == tt:
                            flag = True
                            # 第一类字典，目标是前面一个实体
                            ret.append((positive_relations[(te[2], p[2])] if (te[2], p[2]) in positive_relations else 0,
                                        te[3], p[3],
                                        DataSet.convert_to_relation_index(DataSet.tag_valid_combined[p[1]][te[1]]),
                                        te[2], p[2]))
                        elif te[1] == p[1] and flag:
                            break
            if p[1] in DataSet.tag_valid_combined_reversed:
                for tt in DataSet.tag_valid_combined_reversed[p[1]]:
                    # 向前寻找，找到目标类型之后第一次遇到当前类型结束
                    flag = False
                    for j in range(0, i):
                        te = sorted_entities[i - j - 1]
                        if te[3] >= p[3] - self.min_rel_distance:
                            continue
                        if te[3] < p[3] - self.max_rel_distance:
                            # 超过设置范围的忽略
                            break
                        if te[1] == tt:
                            flag = True
                            # 第二类字典，目标是后面的实体
                            ret.append((positive_relations[(p[2], te[2])] if (p[2], te[2]) in positive_relations else 0,
                                        p[3], te[3],
                                        DataSet.convert_to_relation_index(DataSet.tag_valid_combined[te[1]][p[1]]),
                                        p[2], te[2]))
                        elif te[1] == p[1] and flag:
                            break

        if skip_positive_relations:
            ret = list(filter(lambda x: x[0] == 0, ret))

        return ret

    @staticmethod
    def shuffle_np(arr):
        perm = np.arange(len(arr))
        np.random.shuffle(perm)
        return arr[perm]
