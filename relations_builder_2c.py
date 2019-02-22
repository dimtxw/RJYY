import os


def selu(x, min_value=-2, max_value=2):
    return min(max(x, min_value), max_value)


def build_relations_from_data(doc, filenames, entities, rel, rsl, pred_score, relation_labels, is_selu_score=False):
    # doc是N个int的列表，总共有B个不同的值
    # filenames是M个string的列表
    # entities是B个string => int的字典，对于每个不同的doc值，有一个字典。字典内容是：entity code text => entity id
    # rel是C*L个(relation type, entity1 index, entity2 index, relation type, entity1 id, entity2 id)的列表，shape=(C,L,6)
    # rsl是C个int列表，代表rel中实际有效的数据个数。如rsl[i]=10，那么rel[i, 0:10, :]是有效的数据，其他都是补零对齐的
    # pred_score是C*L个float，代表每个关系的预测得分，同样有效与否由rsl指示，shape=(C,L)
    # relation_lables是一个int => string字典， 内容是relation id(label) => relation label text

    results = []
    relations = {}
    for i in range(len(rel)):
        if i > 0 and doc[i] != doc[i - 1]:
            results.append((filenames[doc[i - 1]][:-4] + ".ann",
                            entities[len(results)],
                            relations,
                            doc[i - 1]))

            relations = {}

        for j in range(rsl[i]):
            if rel[i, j, 3] > 0:
                r_key = tuple(rel[i, j, 4:])
                if is_selu_score:
                    score = selu(pred_score[i, j])
                else:
                    score = pred_score[i, j]
                if r_key in relations:
                    relations[r_key]['score'] += score
                    relations[r_key]['count'] += 1
                else:
                    relations[r_key] = {'score': score,
                                        'relation_label': relation_labels[rel[i, j, 3]],
                                        'distance': abs(rel[i, j, 1] - rel[i, j, 2]),
                                        'count': 1}

    results.append((filenames[doc[-1]][:-4] + ".ann",
                    entities[-1],
                    relations,
                    doc[-1]))

    # 返回(filename, entities, relations, doc_id)四元组列表，每个四元组是一个文件的信息
    # filename是ann文件名
    # entities是这个文件中的实体字典：entity code text => entity id
    # relations是预测得到的关系，是一个字典： (entity1 id, entity2 id) =>
    #           {score: 预测分数, relation_label: 关系标签文本, distance: 实体的距离, count: 预测次数}
    # doc_id是这个文件在数据集中的id(index)
    return results


def merge_relations(relations):
    # relations是多个build_relations_from_data返回的四元组列表
    ret_dict = {}
    for r in relations:
        doc_id = r[3]
        doc_relations = r[2]
        if doc_id not in ret_dict:
            ret_dict[doc_id] = r
        else:
            for k in doc_relations:
                if k in ret_dict[doc_id][2]:
                    ret_dict[doc_id][2][k]['score'] += doc_relations[k]['score']
                    ret_dict[doc_id][2][k]['count'] += doc_relations[k]['count']
                else:
                    ret_dict[doc_id][2][k] = doc_relations[k]
    return list(ret_dict.values())


def build_raw_relations(relations, threshold_fn=None):
    # 生成原始关系表述（把ID/index换成文本）
    # relations是build_relations_from_data返回的四元组列表

    results = []
    for i, item in enumerate(relations):
        raw_relations = []
        reversed_entities = dict(map(lambda t: (t[1], t[0]), item[1].items()))
        for (k, v) in item[2].items():
            if threshold_fn is None:
                threshold = 0
            else:
                threshold = threshold_fn(v['distance'])
            if v['score'] / v['count'] > threshold:
                raw_relations.append((v['relation_label'], reversed_entities[k[0]], reversed_entities[k[1]]))
        results.append((item[0], raw_relations, item[3]))

    # 返回(filename, raw_relations, doc_id)三元组列表，每个三元组是一个文件的信息
    # filename是ann文件名
    # raw_relations是一个(relation label text, entity1 code text, entity2 code text)三元组列表
    # doc_id是这个文件在数据集中的id(index)
    return results


def to_files(raw_relations, output_dir):
    # 生成结果文件
    # raw_relations是build_raw_relations返回的三元组列表

    for item in raw_relations:
        with open(os.path.join(output_dir, item[0]), 'w', encoding='utf-8') as f:
            for i, r in enumerate(item[1]):
                f.write("R{0}\t{1} Arg1:{2} Arg2:{3}\n".format(i+1, r[0], r[1], r[2]))


def compare_result(pred_raw_relations, fact_raw_relations):
    # 比较结果
    # pred_raw_relations/fact_raw_relations都是一个(filename, raw_relations, doc_id)三元组列表
    # raw_relations是一个(relation label text, entity1 code text, entity2 code text)三元组列表

    pred_raw_relations_dict = dict(map(lambda t: (t[2], t[1]), pred_raw_relations))
    fact_raw_relations_dict = dict(map(lambda t: (t[2], t[1]), fact_raw_relations))

    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    total_f1, total_p, total_r = 0.0, 0.0, 0.0
    total_c = 0

    for doc_id, pr in pred_raw_relations_dict.items():
        total_c += 1
        if doc_id not in fact_raw_relations_dict:
            results[doc_id] = {'F1': 0.0, 'P': 0.0, 'R': 0.0, 'TP': [], 'FP': pr, 'FN': []}
            total_fp += len(pr)
        else:
            fr = fact_raw_relations_dict[doc_id]

            # 各种样本列表
            list_tp = []
            list_fp = []
            list_fn = []

            # 分别为预测关系与实际关系构建字典：relation label text => (entity1 code text, entity2 code text) set
            d_pr = {}
            for r in pr:
                k = r[0]
                if k not in d_pr:
                    d_pr[k] = set()
                d_pr[k].add((r[1], r[2]))
            d_fr = {}
            for r in fr:
                k = r[0]
                if k not in d_fr:
                    d_fr[k] = set()
                d_fr[k].add((r[1], r[2]))

            k_pr = set(d_pr.keys())
            k_fr = set(d_fr.keys())

            # 两个集合key的交集
            for k in (k_pr & k_fr):
                set_tp = d_pr[k] & d_fr[k]   # 真阳性集合
                set_fp = d_pr[k] - d_fr[k]   # 假阳性集合
                set_fn = d_fr[k] - d_pr[k]   # 假阴性集合

                list_tp.extend([(k, p[0], p[1]) for p in set_tp])
                list_fp.extend([(k, p[0], p[1]) for p in set_fp])
                list_fn.extend([(k, p[0], p[1]) for p in set_fn])

            # 预测集合中有，实际集合中没有，全部是假阳性
            for k in (k_pr - k_fr):
                list_fp.extend([(k, p[0], p[1]) for p in d_pr[k]])

            # 实际集合中有，预测集合中没有，全部是假阴性
            for k in (k_fr - k_pr):
                list_fn.extend([(k, p[0], p[1]) for p in d_fr[k]])

            c_tp = len(list_tp)
            c_fp = len(list_fp)
            c_fn = len(list_fn)

            m_f1 = 0.0 if c_tp == 0 else 2 * c_tp / (2 * c_tp + c_fp + c_fn)
            m_p = 0.0 if c_tp == 0 else c_tp / (c_tp + c_fp)
            m_r = 0.0 if c_tp == 0 else c_tp / (c_tp + c_fn)

            results[doc_id] = {'F1': m_f1,
                               'P': m_p,
                               'R': m_r,
                               'TP': list_tp, 'FP': list_fp, 'FN': list_fn}
            total_tp += c_tp
            total_fp += c_fp
            total_fn += c_fn
            total_f1 += m_f1
            total_p += m_p
            total_r += m_r

    return {
        'micro': {
            'F1': 0.0 if total_tp == 0 else 2 * total_tp / (2 * total_tp + total_fp + total_fn),
            'P': 0.0 if total_tp == 0 else total_tp / (total_tp + total_fp),
            'R': 0.0 if total_tp == 0 else total_tp / (total_tp + total_fn)
        },
        'macro': {
            'F1': 0.0 if total_c == 0 else total_f1 / total_c,
            'P': 0.0 if total_c == 0 else total_p / total_c,
            'R': 0.0 if total_c == 0 else total_r / total_c
        },
        'detail': results
    }
