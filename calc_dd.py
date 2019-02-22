import os
import numpy as np
from collections import Counter
from dataset_for_calc import DataSet


def calc():
    data_path = 'Demo/DataSets'
    ds = DataSet(os.path.join(data_path, 'ruijin_round2_train/ruijin_round2_train'),
                 test_size=0,
                 seq_mask_rate=0.05,
                 min_rel_distance=0,
                 max_rel_distance=70,
                 max_split_rel_count=139,
                 all_regular_rel_count=260,
                 split_size=140,
                 worker_count=4,
                 capacity=240,
                 return_entity_detail_id=False)

    # 收集训练集所有文章中符合规则的实体对列表
    # 每项是个6元组
    # 6元组第一个元素是真实关系类别，如果为0，代表是负样本
    # 第二第三个元素是两个实体在序列中的位置
    all_srr = []
    for i in range(ds._train_count):
        sample = ds._d_train_samples[i]
        seq = sample[0]
        seq = np.concatenate([seq, np.arange(0, len(seq), dtype=np.int32).reshape(-1, 1)], axis=-1)
        pos_relations = dict([((r[3], r[4]), r[0]) for r in sample[1]])
        entities_s = seq[seq[:, 1] > 0, :]
        srr = ds.build_combine_strong_regular(entities_s, pos_relations, False)
        all_srr.append(srr)

    all_srr = np.concatenate(all_srr, axis=0)

    # 统计正样本各种距离上的数量
    cp = Counter([abs(r[1] - r[2]) for r in all_srr[all_srr[:, 0] > 0]])
    # 统计负样本各种距离上的数量
    cn = Counter([abs(r[1] - r[2]) for r in all_srr[all_srr[:, 0] == 0]])

    # 计算正样本与负样本数量的比例
    # 正样本越多负样本越少此值越高，负样本采样的概率越高，从而达到一定程度上平衡负样本的目的
    # 这个只是随意加的补偿量，所以也没有准确计算能够补偿到什么标准
    # 只用到前70个数据
    r = np.zeros(70, dtype=np.float32)
    for i in range(70):
        k = i + 1
        if k not in cp:
            r[i] = 0
        elif k not in cn:
            r[i] = 0
        else:
            r[i] = cp[k] / cn[k]

    # 归一化
    r = r / np.sum(r)

    # 打印出来就是dataset_ss中的那个dd
    print(r)


if __name__ == "__main__":
    calc()
