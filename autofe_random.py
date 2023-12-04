import time
import random
import os
from pyspark.sql import SparkSession

from core.dataset import Datset
from core.space import SearchSpace
from core.evaluation import FeaSetEvaluation


class Random(object):
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

    def get_random_scheme(self):
        # 随机生成一个特定规模的特征生成选择方案
        rule_params = ['method', 'parts', 'friend_feat_method']
        meaning_types = list(self.feat_meaning_types.keys())
        feature_generation_selection_scheme = []
        while len(feature_generation_selection_scheme) < self.autofe_select_num:
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
        return feature_generation_selection_scheme

    def run(self):
        for i in range(self.autofe_rounds):
            if time.time() - self.start_time >= self.autofe_time_limits:
                break
            # 随机生成一组特征子集
            feature_generation_selection_scheme = self.get_random_scheme()
            print("\n@ iteration" + str(i))
            print("* feature_generation_selection_scheme: " + str(feature_generation_selection_scheme))
            with open(self.autofe_log_file_name, "a+") as f:
                f.write("\n@ iteration" + str(i) + "\n")
                f.write("* feature_generation_selection_scheme: " + str(feature_generation_selection_scheme) + "\n")
            # 评估特征子集
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


if __name__ == '__main__':
    if not os.path.exists("outputs/"):
        os.makedirs("outputs/")

    autofe_config = {
        'autofe_rounds': 200,
        'autofe_time_limits': 3600*5, # business-8, public-ts-1, public-business-5
        'autofe_select_num': 150,
        'autofe_topk': 5,
        'autofe_task_name': 'public_user_KKBox',
        'autofe_xgb_metric': ['Acc'],
    }
    autofe_config['autofe_log_file_name'] = "outputs/autofe_Random_" + str(autofe_config['autofe_task_name']) + ".txt" # 算法搜索过程结果存储路径（databricks端）
    alg_config = {}
    obj = Random(autofe_config, alg_config, Datset, SearchSpace, FeaSetEvaluation)
    best_scheme = obj.run()