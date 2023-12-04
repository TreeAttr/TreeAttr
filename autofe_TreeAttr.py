import time
import random
import numpy as np
import torch
import os
from pyspark.sql import SparkSession

from core.dataset import Datset
from core.space import SearchSpace
from core.evaluation import FeaSetEvaluation
from core.model import Score_Imp_Path_Pair_Model


class TreeAttr(object):
    def __init__(self, autofe_config, alg_config, dataset_func, space_func, evaluation_fun):
        # 获取执行相关配置参数
        self.autofe_rounds = autofe_config['autofe_rounds']  # 算法最大迭代轮数
        self.autofe_time_limits = autofe_config['autofe_time_limits']  # 算法最大执行时长（s）
        self.autofe_select_num = autofe_config['autofe_select_num']  # 最佳特征搜索个数
        self.autofe_topk = autofe_config['autofe_topk']  # 记录topK方案
        self.autofe_task_name = autofe_config['autofe_task_name']  # 任务名称
        self.autofe_xgb_metric = autofe_config['autofe_xgb_metric']  # 任务评估指标
        self.autofe_log_file_name = autofe_config['autofe_log_file_name']  # log文件名称
        self.autofe_config = autofe_config

        # 获取算法相关配置参数
        self.treeattr_generation_method = alg_config['treeattr_generation_method']
        self.treeattr_selection_method = alg_config['treeattr_selection_method']
        self.treeattr_autofe_generation_num = alg_config['treeattr_autofe_generation_num']
        self.treeattr_init_rounds = alg_config['treeattr_init_rounds']
        self.treeattr_embed_dim = alg_config['treeattr_embed_dim']
        self.treeattr_num_heads = alg_config['treeattr_num_heads']
        self.treeattr_batch_size = alg_config['treeattr_batch_size']
        self.treeattr_training_epoch = alg_config['treeattr_training_epoch']
        self.treeattr_neg_num = alg_config['treeattr_neg_num']
        self.alg_config = alg_config
        self.spark = SparkSession.builder.getOrCreate()

        # 根据任务确定数据集、特征子集评估函数、搜索空间函数
        self.dataset_func = dataset_func(self.autofe_task_name, self.spark)
        self.data_config = self.dataset_func.get_data_config(public_regen=True)
        self.trainval_tables = self.dataset_func.get_trainval_tables(self.data_config)
        self.test_tables = self.dataset_func.get_test_tables(self.data_config)

        self.space_func = space_func()
        self.feat_meaning_types, self.feat_available_generation_rules, self.generation_rules, self.label_feature_same_columns = self.space_func.run(
            self.data_config, self.trainval_tables)
        print("feat_meaning_types", self.feat_meaning_types)
        print("feat_available_generation_rules", self.feat_available_generation_rules)
        print("generation_rules", self.generation_rules)
        print("label_feature_same_columns", self.label_feature_same_columns)

        self.autofe_task_details = {
            'data_config': self.data_config,
            'trainval_tables': self.trainval_tables,
            'test_tables': self.test_tables,
            'label_feature_same_columns': self.label_feature_same_columns
        }
        self.evaluation_fun = evaluation_fun(self.autofe_task_details, self.autofe_xgb_metric, self.spark)

        # 记录最佳方案
        self.top_results = []  # item: [index, valid_score, test_score, valid_score_dict, test_score_dict, feature_generation_selection_scheme, select_columns]
        self.start_time = time.time()
        return

    def get_index_dict(self):
        # 对特征、类型、规则、方法、天数信息进行编码
        # index信息生成
        type_index_dict, i = {'None': 0}, 1
        feat_index_dict, j = {'None': 0}, 1
        for type_name in self.feat_meaning_types.keys():
            type_index_dict[type_name] = i
            i = i + 1
            for feat in self.feat_meaning_types[type_name]:
                feat_index_dict[feat] = j
                j = j + 1
        rule_index_dict, k = {'None': 0}, 1
        meth_index_dict, m = {'None': 0}, 1
        day_index_dict, n = {'None': 0}, 1
        for rule_name in self.generation_rules.keys():
            rule_index_dict[rule_name] = k
            k = k + 1
            if 'method' in list(self.generation_rules[rule_name].keys()):
                for method in self.generation_rules[rule_name]['method']:
                    if method not in list(meth_index_dict.keys()):
                        meth_index_dict[method] = m
                        m = m + 1
            if 'parts' in list(self.generation_rules[rule_name].keys()):
                for day in self.generation_rules[rule_name]['parts']:
                    if str(day) not in list(day_index_dict.keys()):
                        day_index_dict[str(day)] = n
                        n = n + 1
            if 'friend_feat_method' in list(self.generation_rules[rule_name].keys()):
                for method in self.generation_rules[rule_name]['friend_feat_method']:
                    if method not in list(meth_index_dict.keys()):
                        meth_index_dict[method] = m
                        m = m + 1
        return type_index_dict, feat_index_dict, rule_index_dict, meth_index_dict, day_index_dict

    def get_scheme_index(self, feature_generation_selection_scheme):
        # 将特征生成选择方案转换为nn模型可识别的index类型
        feature_generation_selection_scheme_index = []
        for scheme in feature_generation_selection_scheme:
            column_name, generation_rule, generation_rule_params = scheme
            method, days, friend_feat_method = generation_rule_params
            feat_index = self.feat_index_dict[str(column_name)]
            for type_name in list(self.feat_meaning_types.keys()):
                if column_name in self.feat_meaning_types[type_name]:
                    type_index = self.type_index_dict[type_name]
                    break
            rule_index = self.rule_index_dict[str(generation_rule)]
            meth_index = self.meth_index_dict[str(method)]
            day_index = self.day_index_dict[str(days)]
            friend_meth_index = self.meth_index_dict[str(friend_feat_method)]
            feature_generation_selection_scheme_index.append(
                [feat_index, type_index, rule_index, meth_index, day_index, friend_meth_index])
        return feature_generation_selection_scheme_index

    def get_random_scheme_index_key(self, key_features, select_num):
        # 随机生成一个特定规模的特征生成选择方案，并将该方案转换为nn模型可识别的index类型
        rule_params = ['method', 'parts', 'friend_feat_method']
        meaning_types = list(self.feat_meaning_types.keys())
        feature_generation_selection_scheme = []
        while len(feature_generation_selection_scheme) < select_num:
            if random.random() <= 0.3 and len(key_features) > 0:
                column_name = random.sample(key_features, 1)[0]
            else:
                meaning_type = random.sample(meaning_types, 1)[0]
                column_name = random.sample(self.feat_meaning_types[meaning_type], 1)[0]
            generation_rule = random.sample(self.feat_available_generation_rules[column_name], 1)[0]
            generation_rule_params = []
            for param in rule_params:
                if param in list(self.generation_rules[generation_rule].keys()):
                    param_value = random.sample(self.generation_rules[generation_rule][param], 1)[0]
                else:
                    param_value = None
                generation_rule_params.append(param_value)
            if [column_name, generation_rule, generation_rule_params] not in feature_generation_selection_scheme:
                feature_generation_selection_scheme.append(list([column_name, generation_rule, generation_rule_params]))
        feature_generation_selection_scheme_index = self.get_scheme_index(feature_generation_selection_scheme)
        return feature_generation_selection_scheme, feature_generation_selection_scheme_index

    def get_random_scheme_index(self, select_num):
        # 随机生成一个特定规模的特征生成选择方案
        rule_params = ['method', 'parts', 'friend_feat_method']
        meaning_types = list(self.feat_meaning_types.keys())
        feature_generation_selection_scheme = []
        while len(feature_generation_selection_scheme) < select_num:
            meaning_type = random.sample(meaning_types, 1)[0]
            column_name = random.sample(self.feat_meaning_types[meaning_type], 1)[0]
            generation_rule = random.sample(self.feat_available_generation_rules[column_name], 1)[0]
            generation_rule_params = []
            for param in rule_params:
                if param in list(self.generation_rules[generation_rule].keys()):
                    param_value = random.sample(self.generation_rules[generation_rule][param], 1)[0]
                else:
                    param_value = None
                generation_rule_params.append(param_value)
            if [column_name, generation_rule, generation_rule_params] not in feature_generation_selection_scheme:
                feature_generation_selection_scheme.append(list([column_name, generation_rule, generation_rule_params]))
        feature_generation_selection_scheme_index = self.get_scheme_index(feature_generation_selection_scheme)
        return feature_generation_selection_scheme, feature_generation_selection_scheme_index

    def run(self):
        # 1--autofe算法初始化
        key_features, key_features_scheme, top3_key_features_scheme = [], [], []
        score_imp_data, tree_path_data, pair_data = {}, [], {}
        type_index_dict, feat_index_dict, rule_index_dict, meth_index_dict, day_index_dict = self.get_index_dict()
        self.type_index_dict, self.feat_index_dict, self.rule_index_dict, self.meth_index_dict, self.day_index_dict = type_index_dict, feat_index_dict, rule_index_dict, meth_index_dict, day_index_dict
        n_feature, n_type, n_rule, n_method, n_day = len(feat_index_dict), len(type_index_dict), len(
            rule_index_dict), len(meth_index_dict), len(day_index_dict)
        print(n_feature, n_type, n_rule, n_method, n_day)
        model = Score_Imp_Path_Pair_Model(n_feature, n_type, n_rule, n_method, n_day, self.treeattr_embed_dim,
                                          self.treeattr_num_heads)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
        loss_fn = torch.nn.MSELoss(reduce=True)
        rule_params = ['method', 'parts', 'friend_feat_method']

        # 2--autofe算法执行
        for i in range(self.autofe_rounds):
            if time.time() - self.start_time >= self.autofe_time_limits:
                break
            print("\n@ iteration" + str(i))
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("\n@ iteration" + str(i) + "\n")
            # 2.1--最佳特征生成选择方案构造
            if i < self.treeattr_init_rounds:
                # 2.1.1--随机生成选择阶段
                feature_generation_selection_scheme, feature_generation_selection_scheme_index = self.get_random_scheme_index(
                    self.autofe_select_num)
            else:
                # 2.1.2--智能生成选择阶段：nn分析器分析，最佳特征生成选择方案构造
                feature_generation_selection_scheme = []
                rate = 1.0 / len(self.treeattr_selection_method)
                total_rate = 0
                if "best_scheme_sample" in self.treeattr_selection_method:
                    if random.random() <= 0.8:
                        feature_generation_selection_scheme = key_features_scheme[
                                                              :max(int(self.autofe_select_num * rate), 1)]
                    else:
                        feature_generation_selection_scheme, _ = self.get_random_scheme_index_key(key_features, max(int(
                            self.autofe_select_num * rate), 1))
                    total_rate += rate
                if "score_filter" in self.treeattr_selection_method:
                    if self.treeattr_generation_method == "random":
                        random_features, random_feature_indexes = self.get_random_scheme_index_key(key_features,
                                                                                                   self.treeattr_autofe_generation_num)
                    else:
                        random_features, random_feature_indexes = [], []

                    feature_info = random_feature_indexes
                    res = model(feature_info, flag="score")
                    for j in range(len(res)):
                        if len(feature_generation_selection_scheme) < int(self.autofe_select_num * (total_rate + rate)):
                            if random_features[res[j][0]] not in feature_generation_selection_scheme:
                                feature_generation_selection_scheme.append(random_features[res[j][0]])
                        else:
                            break
                    total_rate += rate
                if "imp_filter" in self.treeattr_selection_method:
                    if self.treeattr_generation_method == "random":
                        random_features, random_feature_indexes = self.get_random_scheme_index_key(key_features,
                                                                                                   self.treeattr_autofe_generation_num)
                    else:
                        random_features, random_feature_indexes = [], []

                    feature_info = random_feature_indexes
                    res = model(feature_info, flag="imp")
                    for j in range(len(res)):
                        if len(feature_generation_selection_scheme) < int(self.autofe_select_num * (total_rate + rate)):
                            if random_features[res[j][0]] not in feature_generation_selection_scheme:
                                feature_generation_selection_scheme.append(random_features[res[j][0]])
                    total_rate += rate
                if "tree_path" in self.treeattr_selection_method:
                    path_indexes = [[[0, 0, 0, 0, 0, 0]] for j in range(self.treeattr_autofe_generation_num)]
                    path_schemes = [["None"] for j in range(self.treeattr_autofe_generation_num)]
                    for j in range(50):
                        if self.treeattr_generation_method == "random":
                            random_features, random_feature_indexes = self.get_random_scheme_index_key(key_features,
                                                                                                       self.treeattr_autofe_generation_num)
                        else:
                            random_features, random_feature_indexes = [], []
                        path_info = []
                        for k in range(len(random_feature_indexes)):
                            path = list(path_indexes[k]) + list([random_feature_indexes[k]])
                            path_info.append(list(path))
                        res = model(None, path_info=path_info, flag="tree_path")
                        top10_index = [res[k][0] for k in range(10)]
                        for k in range(len(path_indexes)):
                            path_indexes[k].append(random_feature_indexes[random.sample(top10_index, 1)[0]])
                            path_schemes[k].append(random_features[random.sample(top10_index, 1)[0]])
                    for j in range(5):
                        for k in range(1, len(path_schemes[j])):
                            scheme = path_schemes[j][k]
                            if len(feature_generation_selection_scheme) < int(
                                    self.autofe_select_num * (total_rate + rate)):
                                if scheme not in feature_generation_selection_scheme:
                                    feature_generation_selection_scheme.append(scheme)
                    total_rate += rate
                if "pair_path" in self.treeattr_selection_method:
                    if self.treeattr_generation_method == "random":
                        random_features = random.choices(top3_key_features_scheme,
                                                         k=self.treeattr_autofe_generation_num * 10)
                        random_pairs, random_pair_indexes = self.get_random_scheme_index_key(key_features,
                                                                                             self.treeattr_autofe_generation_num * 10)
                    else:
                        random_features = []
                        random_pairs, random_pair_indexes = [], []

                    feature_info = self.get_scheme_index(random_features)
                    pair_info = random_pair_indexes
                    res = model(feature_info, pair_info=pair_info, flag="pair")
                    for j in range(len(res)):
                        if len(feature_generation_selection_scheme) < int(self.autofe_select_num * (total_rate + rate)):
                            if random_pairs[res[j][0]] not in feature_generation_selection_scheme:
                                feature_generation_selection_scheme.append(random_pairs[res[j][0]])
                    total_rate += rate

                # 补充
                while len(feature_generation_selection_scheme) < self.autofe_select_num:
                    meaning_type = random.sample(list(self.feat_meaning_types.keys()), 1)[0]
                    column_name = random.sample(self.feat_meaning_types[meaning_type], 1)[0]
                    generation_rule = random.sample(self.feat_available_generation_rules[column_name], 1)[0]
                    generation_rule_params = []
                    for param in rule_params:
                        if param in list(self.generation_rules[generation_rule].keys()):
                            param_value = random.sample(self.generation_rules[generation_rule][param], 1)[0]
                        else:
                            param_value = None
                        generation_rule_params.append(param_value)
                    if [column_name, generation_rule,
                        generation_rule_params] not in feature_generation_selection_scheme:
                        feature_generation_selection_scheme.append(
                            list([column_name, generation_rule, generation_rule_params]))
            print("* feature_generation_selection_scheme: " + str(feature_generation_selection_scheme))
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("* feature_generation_selection_scheme: " + str(feature_generation_selection_scheme) + "\n")

            # 2.2--特征生成选择方案评估
            # 验证集和测试集性能
            scheme_valid_score, scheme_valid_score_dict, scheme_test_score, scheme_test_score_dict, select_columns, extra_details = self.evaluation_fun.run(
                feature_generation_selection_scheme)
            print("* select_columns: " + str(select_columns))
            print("* scheme_valid_score: " + str(scheme_valid_score))
            print("* scheme_valid_score_dict: " + str(scheme_valid_score_dict))
            print("* scheme_test_score: " + str(scheme_test_score))
            print("# scheme_test_score_dict: " + str(scheme_test_score_dict) + ", duration: " + str(
                (time.time() - self.start_time) / 60) + "min")
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("* select_columns: " + str(select_columns) + "\n")
                f.write("* scheme_valid_score: " + str(scheme_valid_score) + "\n")
                f.write("* scheme_valid_score_dict: " + str(scheme_valid_score_dict) + "\n")
                f.write("* scheme_test_score: " + str(scheme_test_score) + "\n")
                f.write("# scheme_test_score_dict: " + str(scheme_test_score_dict) + ", duration: " + str(
                    (time.time() - self.start_time) / 60) + "min\n")
            if self.autofe_task_name[:10] == "public_ts_":
                scheme_valid_score = scheme_test_score
            # 更新topk结果
            scheme_info = {
                'scheme_index': i,
                'scheme_valid_score': scheme_valid_score,
                'scheme_valid_score_dict': scheme_valid_score_dict,
                'scheme_test_score': scheme_test_score,
                'scheme_test_score_dict': scheme_test_score_dict,
                'feature_generation_selection_scheme': list(feature_generation_selection_scheme),
                'select_columns': list(select_columns)
            }
            self.update_topK(scheme_info)

            # 2.3--特征生成选择方案评估结果分析
            get_column_names_rule_details = extra_details['get_column_names_rule_details']
            scheme_model = extra_details['scheme_model']
            xgb_booster = scheme_model.get_booster()

            fscore = xgb_booster.get_fscore()
            select_columns_dict = {"f" + str(j): select_columns[j] for j in range(len(select_columns))}
            fscore_rank = [[select_columns_dict[key], value] for (key, value) in fscore.items()]
            fscore_rank = sorted(fscore_rank, key=lambda x: x[1], reverse=True)
            # 更新key_features，key_features_scheme，top3_key_features_scheme
            if self.top_results[0][0] == i:
                key_features = list(
                    set([get_column_names_rule_details[fscore_rank[j][0]][0] for j in range(len(fscore_rank))]))
                key_features_scheme = [get_column_names_rule_details[fscore_rank[j][0]][:3] for j in
                                       range(len(fscore_rank))]
                for j in range(len(fscore_rank)):
                    scheme = get_column_names_rule_details[fscore_rank[j][0]][:3]
                    if scheme not in top3_key_features_scheme:
                        top3_key_features_scheme.append(scheme)
                top3_key_features_scheme = top3_key_features_scheme[-100:]
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("-- fscore_rank: " + str(fscore_rank) + "\n")
            # 更新score_imp_data
            # feature: {'infos': [(score, imp)], 'score_max': None, 'score_avg': None, 'imp_max': None, 'imp_avg': None, 'scheme': None, 'scheme_index': None, 'infos_num': 0}
            used_columns = list(fscore.keys())
            for j in range(len(select_columns)):
                if "f" + str(j) not in used_columns:
                    fscore_rank.append([select_columns_dict["f" + str(j)], 0])
            for item in fscore_rank:
                column, value = item
                value = value / 10 if value > 0 else -1
                valid_auc = scheme_valid_score if value > 0 else 0
                test_auc = scheme_test_score if value > 0 else 0
                if column not in score_imp_data.keys():
                    score_imp_data[column] = {
                        'infos': [],
                        'score_max': 0,
                        'score_avg': 0,
                        'imp_max': 0,
                        'imp_avg': 0,
                        'scheme': list(get_column_names_rule_details[column][:3]),
                        'scheme_index': list(self.get_scheme_index([get_column_names_rule_details[column][:3]]))[0],
                        'info_num': 0
                    }
                score_imp_data[column]['infos'].append([valid_auc, test_auc, value])
                score_imp_data[column]['score_max'] = max(score_imp_data[column]['score_max'], valid_auc)
                score_imp_data[column]['score_avg'] = (score_imp_data[column]['score_avg'] * score_imp_data[column][
                    'info_num'] + valid_auc) / (score_imp_data[column]['info_num'] + 1)
                score_imp_data[column]['imp_max'] = max(score_imp_data[column]['imp_max'], value)
                score_imp_data[column]['imp_avg'] = (score_imp_data[column]['imp_avg'] * score_imp_data[column][
                    'info_num'] + value) / (score_imp_data[column]['info_num'] + 1)
                score_imp_data[column]['info_num'] = score_imp_data[column]['info_num'] + 1
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("-- score_imp_data: " + str(len(score_imp_data)) + ", " + str(
                    [[key, value] for key, value in score_imp_data.items()][-5:]) + "\n")
            # 更新tree_path_data
            trees_dataframe = xgb_booster.trees_to_dataframe()
            column_name_seq = ["None"]
            column_scheme_seq = [["None", "None", ["None", "None", "None"]]]
            column_scheme_index_seq = [[0, 0, 0, 0, 0, 0]]
            for index, row in trees_dataframe.iterrows():
                if row['Feature'] != 'Leaf':
                    column_name = select_columns_dict[row['Feature']]
                    if column_name not in column_name_seq:
                        column_name_seq.append(column_name)
                        column_scheme_seq.append(get_column_names_rule_details[column_name][:3])
                        column_scheme_index_seq.append(
                            list(self.get_scheme_index([get_column_names_rule_details[column_name][:3]]))[0])
            tree_path_data.append(list(
                [column_name_seq, column_scheme_seq, column_scheme_index_seq, scheme_valid_score, scheme_test_score]))
            if i == 0:
                base_rl_score = scheme_valid_score
            # 更新pair_data
            node_name_dict = {}
            pair_node_list = []
            for index, row in trees_dataframe.iterrows():
                node_name = row['ID']
                if row['Feature'] != 'Leaf':
                    column_name = select_columns_dict[row['Feature']]
                    node_name_dict[node_name] = column_name
                    yes_node_name = row['Yes']
                    no_node_name = row['No']
                    pair_node_list.append([node_name, yes_node_name, no_node_name])
            for pair_info in pair_node_list:
                node_name, yes_node_name, no_node_name = pair_info
                try:
                    column_name = node_name_dict[node_name]
                    yes_column_name = node_name_dict[yes_node_name]
                    no_column_name = node_name_dict[no_node_name]
                except:
                    continue
                for columny in [yes_column_name, no_column_name]:
                    pair_name = str([column_name, columny])
                    if pair_name not in pair_data.keys():
                        pair_data[pair_name] = {
                            'infos': [],
                            'score_max': 0,
                            'score_avg': 0,
                            'schemex': list(get_column_names_rule_details[column_name][:3]),
                            'schemex_index':
                                list(self.get_scheme_index([get_column_names_rule_details[column_name][:3]]))[0],
                            'schemey': list(get_column_names_rule_details[columny][:3]),
                            'schemey_index': list(self.get_scheme_index([get_column_names_rule_details[columny][:3]]))[
                                0],
                            'info_num': 0
                        }
                    pair_data[pair_name]['infos'].append([scheme_valid_score, scheme_test_score])
                    pair_data[pair_name]['score_max'] = max(pair_data[pair_name]['score_max'], scheme_valid_score)
                    pair_data[pair_name]['score_avg'] = (pair_data[pair_name]['score_avg'] * pair_data[pair_name][
                        'info_num'] + scheme_valid_score) / (pair_data[pair_name]['info_num'] + 1)
                    pair_data[pair_name]['info_num'] = pair_data[pair_name]['info_num'] + 1
                neg_column_names = []
                while len(neg_column_names) < self.treeattr_neg_num:
                    neg_column_name = random.sample(select_columns, 1)[0]
                    if neg_column_name not in neg_column_names and neg_column_name not in [yes_column_name,
                                                                                           no_column_name]:
                        neg_column_names.append(neg_column_name)
                for columny in neg_column_names:
                    pair_name = str([column_name, columny])
                    if pair_name not in pair_data.keys():
                        pair_data[pair_name] = {
                            'infos': [],
                            'score_max': 0,
                            'score_avg': 0,
                            'schemex': list(get_column_names_rule_details[column_name][:3]),
                            'schemex_index':
                                list(self.get_scheme_index([get_column_names_rule_details[column_name][:3]]))[0],
                            'schemey': list(get_column_names_rule_details[columny][:3]),
                            'schemey_index': list(self.get_scheme_index([get_column_names_rule_details[columny][:3]]))[
                                0],
                            'info_num': 0
                        }
                    pair_data[pair_name]['infos'].append([0, 0])
                    pair_data[pair_name]['score_avg'] = (pair_data[pair_name]['score_avg'] * pair_data[pair_name][
                        'info_num'] + 0) / (pair_data[pair_name]['info_num'] + 1)
                    pair_data[pair_name]['info_num'] = pair_data[pair_name]['info_num'] + 1
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("-- pair_data: " + str(len(pair_data)) + ", " + str(
                    [[key, value] for key, value in pair_data.items()][-5:]) + "\n")

            # 2.4--nn分析器训练
            # score_imp model training
            score_imp_data_train = [[score_imp_data[column]['scheme_index'],
                                     [score_imp_data[column]['score_max'], score_imp_data[column]['score_avg'],
                                      score_imp_data[column]['imp_max'], score_imp_data[column]['imp_avg']]] for column
                                    in score_imp_data.keys()]
            random.shuffle(score_imp_data_train)
            for e in range(self.treeattr_training_epoch):
                p, start, end = 0, 0, 0
                loss_all = 0
                while end < len(score_imp_data_train):
                    start = p * self.treeattr_batch_size
                    end = min((p + 1) * self.treeattr_batch_size, len(score_imp_data_train))
                    p = p + 1

                    train_x = [score_imp_data_train[j][0] for j in range(start, end)]
                    train_y = torch.tensor(np.array([score_imp_data_train[j][1] for j in range(start, end)])).float()
                    train_y_pred = model(train_x, flag="score_imp_predict")

                    optimizer.zero_grad()
                    loss = loss_fn(train_y_pred, train_y)
                    loss_all += loss.item()
                    loss.backward()
                    optimizer.step()
                print("score_imp model", e, loss_all)
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("-- score_imp model loss: " + str(loss_all) + ", " + str(self.treeattr_training_epoch) + "\n")
            # tree_path model training
            tree_path_data_train = [[item[2], item[3]] for item in tree_path_data]
            random.shuffle(tree_path_data_train)
            for e in range(self.treeattr_training_epoch):
                p, start, end = 0, 0, 0
                loss_all = 0
                while end < len(tree_path_data_train):
                    start = p * 4
                    end = min((p + 1) * 4, len(tree_path_data_train))
                    p = p + 1

                    train_x = [tree_path_data_train[j][0] for j in range(start, end)]
                    train_y = torch.tensor(np.array([tree_path_data_train[j][1] for j in range(start, end)])).float()
                    train_y_pred = model(None, path_info=train_x, flag="tree_path_predict")
                    optimizer.zero_grad()
                    loss = 0
                    for j in range(train_y.shape[0]):
                        loss = loss + (base_rl_score - train_y[j]) * torch.log(train_y_pred[j])
                    loss_all += loss.item()
                    loss.backward()
                    optimizer.step()
                print("tree_path model", e, loss_all)
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("-- tree_path model loss: " + str(loss_all) + ", " + str(self.treeattr_training_epoch) + "\n")
            # pair model training
            pair_data_train = [[pair_data[pair]['schemex_index'], pair_data[pair]['schemey_index'],
                                [pair_data[pair]['score_max'], pair_data[pair]['score_avg']]] for pair in
                               pair_data.keys()]
            random.shuffle(pair_data_train)
            for e in range(self.treeattr_training_epoch):
                p, start, end = 0, 0, 0
                loss_all = 0
                while end < len(pair_data_train):
                    start = p * self.treeattr_batch_size
                    end = min((p + 1) * self.treeattr_batch_size, len(pair_data_train))
                    p = p + 1

                    train_x1 = [pair_data_train[j][0] for j in range(start, end)]
                    train_x2 = [pair_data_train[j][1] for j in range(start, end)]
                    train_y = torch.tensor(np.array([pair_data_train[j][2] for j in range(start, end)])).float()
                    train_y_pred = model(train_x1, pair_info=train_x2, flag="pair_predict")

                    optimizer.zero_grad()
                    loss = loss_fn(train_y_pred, train_y)
                    loss_all += loss.item()
                    loss.backward()
                    optimizer.step()
                print("pair model", e, loss_all)
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("-- pair model loss: " + str(loss_all) + ", " + str(self.treeattr_training_epoch) + "\n")

        # 从topk结果中选择最佳结果，作为算法最终输出
        best_scheme = self.get_best_scheme()
        return best_scheme

    def update_topK(self, scheme_info):
        # 更新topk结果
        scheme_index = scheme_info['scheme_index']
        scheme_valid_score = scheme_info['scheme_valid_score']
        scheme_valid_score_dict = scheme_info['scheme_valid_score_dict']
        scheme_test_score = scheme_info['scheme_test_score']
        scheme_test_score_dict = scheme_info['scheme_test_score_dict']
        feature_generation_selection_scheme = scheme_info['feature_generation_selection_scheme']
        select_columns = scheme_info['select_columns']

        self.top_results.append(list(
            [scheme_index, scheme_valid_score, scheme_test_score, scheme_valid_score_dict, scheme_test_score_dict,
             feature_generation_selection_scheme, select_columns]))
        self.top_results = sorted(self.top_results, key=lambda x: x[1], reverse=True)[:self.autofe_topk]

        # 最新结果输出
        print("! top_results", [result[:3] for result in self.top_results])
        print("!!! best_result", self.top_results[0][0], self.top_results[0][1], self.top_results[0][2])
        print("!!! best_result_dict(valid)", self.top_results[0][3])
        print("!!! best_result_dict(test)", self.top_results[0][4])
        print("!!! best feature_generation_selection_scheme", self.top_results[0][5])
        duration = (time.time() - self.start_time) / 60.0
        print("!!! best select_columns", self.top_results[0][6], 'duration', duration)
        with open(self.autofe_log_file_name, "a+") as f:
            f.write("! top_results: " + str([result[:3] for result in self.top_results]) + "\n")
            f.write("!!! best_result: " + str(self.top_results[0][:3]) + "\n")
            f.write("!!! best_result_dict(valid): " + str(self.top_results[0][3]) + "\n")
            f.write("!!! best_result_dict(test): " + str(self.top_results[0][4]) + "\n")
            f.write("!!! feature_generation_selection_scheme: " + str(self.top_results[0][5]) + "\n")
            f.write("!!! select_columns: " + str(self.top_results[0][6]) + ", duration: " + str(duration) + "\n")
        return

    def get_best_scheme(self):
        # 从topk结果中选择最佳结果，作为算法最终输出
        best_result = sorted(self.top_results, key=lambda x: x[2], reverse=True)[0]
        print("best_result", best_result[:5])
        best_feature_generation_selection_scheme, best_select_columns = best_result[5], best_result[6]
        best_scheme = {
            'best_feature_generation_selection_scheme': best_feature_generation_selection_scheme,
            'best_select_columns': best_select_columns,
            'autofe_config': self.autofe_config,
            'alg_config': self.alg_config,
            'top_results': self.top_results
        }
        print("best_scheme", best_scheme)
        return best_scheme


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if not os.path.exists("outputs/"):
        os.makedirs("outputs/")

    autofe_config = {
        'autofe_rounds': 200,
        'autofe_time_limits': 3600 * 5,  # business-8, public-ts-1, public-business-5
        'autofe_select_num': 10,
        'autofe_topk': 5,
        'autofe_task_name': 'public_user_KKBox',
        'autofe_xgb_metric': ['Acc'],
    }
    autofe_config['autofe_log_file_name'] = "outputs/autofe_TreeAttr_" + str(autofe_config['autofe_task_name']) + ".txt"  # 算法搜索过程结果存储路径（databricks端）
    alg_config = {
        'treeattr_generation_method': 'random',
        'treeattr_selection_method': ["best_scheme_sample", "imp_filter", "tree_path", "pair_path",
                                      "random_generation"],
        'treeattr_autofe_generation_num': 1000,
        'treeattr_init_rounds': 1,
        'treeattr_embed_dim': 32,
        'treeattr_num_heads': 4,
        'treeattr_batch_size': 64,
        'treeattr_training_epoch': 10,
        'treeattr_neg_num': 2,
    }
    obj = TreeAttr(autofe_config, alg_config, Datset, SearchSpace, FeaSetEvaluation)
    best_scheme = obj.run()
