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
    dd = np.array([0.166082884, 0.07415115, 0.085323476, 0.06540342, 0.060209436, 0.045123433, 0.04024525, 0.033430686,
                   0.030436955, 0.024878187, 0.024244158, 0.020113539, 0.018041933, 0.017044035, 0.015117045,
                   0.015047088, 0.01263797, 0.012110642, 0.011160911, 0.010675937, 0.010081212, 0.010102517,
                   0.008975307, 0.008483644, 0.008236351, 0.007756464, 0.007269872, 0.006969876, 0.006758432,
                   0.006499717, 0.006075802, 0.005566457, 0.005731588, 0.005309538, 0.005172165, 0.005144202,
                   0.004724465, 0.005149776, 0.004620759, 0.004442354, 0.004455768, 0.003908822, 0.003784738,
                   0.003987622, 0.003852502, 0.003612988, 0.003583327, 0.003540516, 0.003405853,
                   0.003332433, 0.002910762, 0.002795673, 0.002990742, 0.002723203, 0.002951466, 0.002587495,
                   0.002768828, 0.002723538, 0.002421426, 0.002215874, 0.002463461, 0.001993889, 0.00217896, 0.00216905,
                   0.002338657, 0.002119497, 0.001928417, 0.002004743, 0.00179522, 0.001905897])

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

        np.random.seed()  # 随机种子
        self._cached_data = [None for _ in range(capacity)]
        self._capacity = capacity
        self._cache_offset = 0
        self._cache_num = 0
        self._default_build_size = default_build_size
        self._mutex = threading.Lock()
        self._mutex_read = threading.Lock()
        self._closed = False
        if worker_count > 1:
            self._executor = ThreadPoolExecutor(worker_count)
        else:
            self._executor = None
        if enabled_pre_build:
            self._pre_build_thread = threading.Thread(target=self.pre_build, daemon=True)
            self._pre_build_thread.start()

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

    @property
    def train_count(self):
        return self._train_count

    @property
    def is_train_end(self):
        return self.train_position == 0

    @property
    def test_count(self):
        return self._test_count

    @property
    def is_test_end(self):
        return self._test_offset >= self._test_count

    def reset_train(self):
        with self._mutex_read:
            if self._train_offset <= self._cache_num:
                # 缓存中数据是包含0的，把next读取开始位置指向0处
                self._cache_num = self._train_offset
            else:
                self._train_offset = 0
                self._cache_num = 0
                self._cache_offset = 0

    def reset_test(self):
        self._test_offset = 0

    @property
    def train_position(self):
        if self._train_count > 0:
            ret = self._train_offset - self._cache_num
            while ret < 0:
                ret += self._train_count
        else:
            ret = 0
        return ret

    def skip_train(self, count):
        with self._mutex_read:
            self._train_offset = (self._train_offset + count) % self._train_count

            if count < self._cache_num:
                self._cache_num -= count
            else:
                self._train_offset = (self.train_position + count) % self._count
                self._cache_num = 0
                self._cache_offset = 0

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

    def build_samples(self, sample, doc_id,
                      pad=None,
                      is_all_regular_combine=False,
                      is_strong_regular=False,
                      stride_rate=2,
                      mask_rate=0.0):
        d_seq = []
        d_seq_len = []
        d_rel = []
        d_rel_len = []
        d_doc = []

        # sample结构
        # 0: 序列  (token id，token type，entity id)
        seq = sample[0]
        # 1: 关系  (relation type, entity1 pos, entity2 pos, entity1 id, entity2 id, regular relation type)
        relations = sample[1]
        # 2: 实体字典  {text: id}
        # 3. 实体位置  {id: pos}

        step = self.split_size // stride_rate
        # 随机产生一个pad
        if pad is None:
            if self.is_train_random_pad_seq:
                pad = np.random.randint(0, step)
                use_cached_combine = True
            else:
                pad = 0
                use_cached_combine = True
        else:
            use_cached_combine = True
        pos_relations = dict([((r[3], r[4]), r[0]) for r in relations])

        # 对序列进行分割
        splited_seq = [seq[:j + self.split_size] if j < 0 else seq[j:j + self.split_size]
                       for j in range(-pad, len(seq) - step, step)]
        for j in range(len(splited_seq)):
            seq = np.array(splited_seq[j])
            if mask_rate > 0:
                # 按mask_rate随机选择一些位置替换为mask_token
                # 不能把实体mask掉了，因为生成组合会用到
                seq[(np.random.random((len(seq),)) < mask_rate) & (seq[:, 1] == 0)] = \
                    (DataSet.get_token_index("__mask__", 0), 0, 0)

            # 加上位置编号
            seq = np.concatenate([seq, np.arange(0, len(seq), dtype=np.int32).reshape(-1, 1)], axis=-1)
            entities_s = seq[seq[:, 1] > 0, :]

            if len(entities_s) <= 1:
                # 无法形成组合
                continue

            if not is_all_regular_combine:
                # 在正样本数量小于最大样本数量一半的情况下：
                # 如果正样本数量大于等于3，采样的规则负样本最多为正样本数量2倍，自由负样本为正样本数量的1/3
                # 如果正样本数量小于等于3，采样的规则负样本最多为6，自由负样本1个
                # 如果总数量超过最大样本数量，正样本保持固定，规则和自由负样本按比率缩减
                # 如果采样回来的规则负样本不足正样本的两倍，全体规则负样本都使用，使用自由负样本补充1/3

                # 获取在这个segment中的正样本和规则负样本
                # input: 关系  (relation type, entity1 pos, entity2 pos, entity1 id, entity2 id)
                # (real relation type, entity1 index, entity2 index, regular relation type, entity1 id, entity2 id)
                regular_combine_cache_key = (doc_id, pad, j)

                if not use_cached_combine:
                    if is_strong_regular:
                        srr = self.build_combine_strong_regular(entities_s, pos_relations, False)
                    else:
                        srr = self.build_combine_regular(entities_s, pos_relations, False)
                    srr = np.array(srr, dtype=np.int32)
                else:
                    if regular_combine_cache_key not in self._cached_regular_combine:
                        if is_strong_regular:
                            srr = self.build_combine_strong_regular(entities_s, pos_relations, False)
                        else:
                            srr = self.build_combine_regular(entities_s, pos_relations, False)
                        srr = np.array(srr, dtype=np.int32)
                        self._cached_regular_combine[regular_combine_cache_key] = srr
                    else:
                        srr = self._cached_regular_combine[regular_combine_cache_key]

                if len(srr) == 0:
                    # 一个规则样本都没有，跳过
                    continue

                # 正样本第一级
                s1 = srr[srr[:, 0] > 0]
                pos_cnt = len(s1)
                if pos_cnt <= 3:
                    neg_regular_cnt = 6
                elif pos_cnt <= self.max_split_rel_count // 3:
                    neg_regular_cnt = pos_cnt * 2
                else:
                    neg_regular_cnt = self.max_split_rel_count - pos_cnt

                # 符合规则的负样本第二级
                s2 = srr[srr[:, 0] == 0]
                if len(s2) > neg_regular_cnt:
                    w = DataSet.dd[np.abs(s2[:, 1] - s2[:, 2])-1]
                    w += np.random.random((len(s2),)) * (np.random.random() * 0.1 + 0.02)
                    item_indices = list(np.concatenate([w.reshape(-1, 1),
                                                        np.arange(len(s2)).reshape(-1, 1)], axis=1))
                    item_indices.sort(key=lambda x: x[0], reverse=True)
                    s2 = s2[[int(inx[1]) for inx in item_indices][:neg_regular_cnt]]

                rel = np.concatenate([s1, s2], axis=0)

                rel_padded = np.zeros((self.max_split_rel_count, 6), dtype=np.int32)
                rel_padded[:len(rel), :] = rel[:, :6]
                d_rel.append(rel_padded)
                d_rel_len.append(len(rel))
            else:
                if is_strong_regular:
                    rel = np.array(self.build_combine_strong_regular(entities_s, positive_relations=pos_relations))
                else:
                    rel = np.array(self.build_combine_regular(entities_s, positive_relations=pos_relations))
                if len(rel) > self.all_regular_rel_count:
                    print("Warning: rel count %d for split %d in doc %d is exceeded max rel count" % (
                    len(rel), j, doc_id))
                    rel = rel[:self.all_regular_rel_count]
                elif len(rel) == 0:
                    continue

                rel_padded = np.zeros((self.all_regular_rel_count, 6), dtype=np.int32)
                rel_padded[:len(rel), :] = rel[:, :6]
                d_rel.append(rel_padded)
                d_rel_len.append(len(rel))

            seq_padded = np.zeros((self.split_size, 2), dtype=np.int32)
            seq_padded[:len(seq), :] = seq[:, :2]
            d_seq.append(seq_padded)
            d_seq_len.append(len(seq))

            d_doc.append(doc_id)

        return {
            'seq': d_seq,
            'seq_len': d_seq_len,
            'rel': d_rel,
            'rel_len': d_rel_len,
            'doc': d_doc
        }

    def pre_build(self):
        while not self._closed:
            self.build(self._default_build_size)
            sleep(0.02)

    def build_train_worker(self, cache_index, train_index):
        self._cached_data[cache_index] = self.build_samples(
            self._d_train_samples[train_index],
            -train_index - 1,
            mask_rate=self.seq_mask_rate,
            is_strong_regular=True)

    def build(self, size):
        with self._mutex:
            if size > self._capacity - self._cache_num:
                size = self._capacity - self._cache_num
            if size > 0:
                if self._executor is not None:
                    all_task = []
                    for i in range(size):
                        all_task.append(
                            self._executor.submit(
                                self.build_train_worker,
                                (self._cache_offset + i) % self._capacity,
                                (self._train_offset + i) % self._train_count
                            )
                        )
                    wait(all_task, return_when=ALL_COMPLETED)
                else:
                    for i in range(size):
                        cache_index = (self._cache_offset + i) % self._capacity
                        train_index = (self._train_offset + i) % self._train_count
                        self._cached_data[cache_index] = self.build_samples(
                            self._d_train_samples[train_index], -train_index - 1,
                            mask_rate=self.seq_mask_rate,
                            is_strong_regular=True)

                with self._mutex_read:
                    self._cache_num += size
                    self._cache_offset = (self._cache_offset + size) % self._capacity
                    self._train_offset = (self._train_offset + size) % self._train_count

    def next_train(self, batch_size, silenced=True):
        d_seq = []
        d_seq_len = []
        d_rel = []
        d_rel_len = []
        d_doc = []

        while not self._closed:
            with self._mutex_read:
                if batch_size > self._train_count - self.train_position:
                    batch_size = self._train_count - self.train_position

                if self._cache_num >= batch_size:
                    for i in range(batch_size):
                        cache_index = (self._cache_offset - self._cache_num + self._capacity + i) % self._capacity
                        cache_data = self._cached_data[cache_index]
                        d_seq.extend(cache_data['seq'])
                        d_seq_len.extend(cache_data['seq_len'])
                        d_rel.extend(cache_data['rel'])
                        d_rel_len.extend(cache_data['rel_len'])
                        d_doc.extend(cache_data['doc'])

                    self._cache_num -= batch_size
                    break

            if self.enabled_pre_build and not silenced:
                st = time()
                self.build(batch_size)
                print('wait build data', time() - st)
            else:
                self.build(batch_size)

        return np.array(d_seq), np.array(d_seq_len), \
               np.array(d_rel), np.array(d_rel_len), np.array(d_doc), len(d_seq)

    def next_flod_test(self, i, k, train_batch_size):
        d_seq = []
        d_seq_len = []
        d_rel = []
        d_rel_len = []
        d_doc = []
        d_entities = []

        for test_index in range(self._train_count):
            if (test_index // train_batch_size) % k == i:
                if self._d_flod_test_built_samples[test_index] is None:
                    self._d_flod_test_built_samples[test_index] = self.build_samples(
                        self._d_train_samples[test_index],
                        test_index,
                        pad=0,
                        is_all_regular_combine=self.is_test_all_regular_rel,
                        is_strong_regular=True
                    )

                cache_data = self._d_flod_test_built_samples[test_index]
                d_entities.append(self._d_train_samples[test_index][2])
                d_seq.extend(cache_data['seq'])
                d_seq_len.extend(cache_data['seq_len'])
                d_rel.extend(cache_data['rel'])
                d_rel_len.extend(cache_data['rel_len'])
                d_doc.extend(cache_data['doc'])

        return np.array(d_seq), np.array(d_seq_len), \
               np.array(d_rel), np.array(d_rel_len), np.array(d_doc), d_entities, len(d_seq)

    def next_test(self, batch_size):
        if self._test_offset >= self._test_count:
            self.reset_test()

        if batch_size > self._test_count - self._test_offset:
            batch_size = self._test_count - self._test_offset

        d_seq = []
        d_seq_len = []
        d_rel = []
        d_rel_len = []
        d_doc = []
        d_entities = []

        for i in range(batch_size):
            test_index = (self._test_offset + i) % self._test_count
            if self._d_test_built_samples[test_index] is None:
                self._d_test_built_samples[test_index] = self.build_samples(
                    self._d_test_samples[test_index],
                    test_index,
                    pad=0,
                    is_all_regular_combine=self.is_test_all_regular_rel,
                    is_strong_regular=True
                )

            cache_data = self._d_test_built_samples[test_index]
            d_entities.append(self._d_test_samples[test_index][2])
            d_seq.extend(cache_data['seq'])
            d_seq_len.extend(cache_data['seq_len'])
            d_rel.extend(cache_data['rel'])
            d_rel_len.extend(cache_data['rel_len'])
            d_doc.extend(cache_data['doc'])

        self._test_offset += batch_size

        return np.array(d_seq), np.array(d_seq_len), \
               np.array(d_rel), np.array(d_rel_len), np.array(d_doc), d_entities, len(d_seq)

    def next_exec(self, batch_size):
        if self._test_offset >= self._test_count:
            self.reset_test()

        if batch_size > self._test_count - self._test_offset:
            batch_size = self._test_count - self._test_offset

        d_seq = []
        d_seq_len = []
        d_rel = []
        d_rel_len = []
        d_doc = []
        d_entities = []

        for i in range(batch_size):
            test_index = (self._test_offset + i) % self._test_count
            s = self._d_test_samples[test_index]
            cache_data = self.build_samples(s, test_index, pad=0,
                                            is_all_regular_combine=True,
                                            is_strong_regular=True)

            d_entities.append(s[2])
            d_seq.extend(cache_data['seq'])
            d_seq_len.extend(cache_data['seq_len'])
            d_rel.extend(cache_data['rel'])
            d_rel_len.extend(cache_data['rel_len'])
            d_doc.extend(cache_data['doc'])

        self._test_offset += batch_size

        return np.array(d_seq), np.array(d_seq_len), \
               np.array(d_rel), np.array(d_rel_len), np.array(d_doc), d_entities, len(d_seq)

    @property
    def test_filenames(self):
        return self._test_files[:, 2]

    @property
    def train_filenames(self):
        return self._train_files[:, 2]

    @property
    def max_train_seq_len(self):
        return self.split_size

    @property
    def max_test_seq_len(self):
        return self.split_size

    @property
    def max_train_rel_count(self):
        return self.max_split_rel_count

    @property
    def max_test_rel_count(self):
        if self.is_test_all_regular_rel:
            return self.all_regular_rel_count
        else:
            return self.max_split_rel_count

    def get_raw_relations_of_test(self):
        return [(self._test_files[i, 2], self._d_test_samples[i][4], i) for i in range(self._test_count)]

    def get_raw_relations_of_flod_k(self, i, k, train_batch_size):
        return [(self._train_files[j, 2], self._d_train_samples[j][4], j)
                for j in range(self._train_count)
                if (j // train_batch_size) % k == i]
