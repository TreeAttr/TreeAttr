import time
import random
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np


# 根据任务确定特征子集评估函数
class FeaSetEvaluation(object):
    def __init__(self, autofe_task_details, xgb_metric, spark):
        self.data_config = autofe_task_details['data_config']
        self.trainval_tables = autofe_task_details['trainval_tables']
        self.test_tables = autofe_task_details['test_tables']
        self.label_feature_same_columns = autofe_task_details['label_feature_same_columns']
        self.xgb_metric = xgb_metric
        self.spark = spark
        return

    def generate_features(self, sample_infos, feature_generation_selection_scheme=[], na=0):
        # 基础特征表格时间分区为天级
        timestamp_style = 'yyyyMMdd'
        gap_addition = 60 * 60 * 60

        # 生成适用于databricks端的时间处理函数（时间加减）
        def get_databricks_time_sql(time_column, add_value, timestamp_style, gap_addition):
            sql = "INT(from_unixtime(unix_timestamp(STRING(" + str(time_column) + "), '" + str(
                timestamp_style) + "') + (" + str(add_value) + "*" + str(gap_addition) + "), '" + str(
                timestamp_style) + "'))"
            return sql

        # 融合新特征，并清理数据
        def merge_new_columns(df_gen, gen_column_names, newly_add_column_names):
            # 融合新特征
            try:
                df_gen.createTempView("df_gen")
            except:
                self.spark.catalog.dropTempView("df_gen")
                df_gen.createTempView("df_gen")

            generation_sql = "select "
            choices = []
            for item in gen_column_names:
                generation_sql = generation_sql + "t1." + item + " as " + item + ", "
            for item in newly_add_column_names:
                generation_sql = generation_sql + "t2." + item + " as " + item + ", "
                gen_column_names.append(item)
            generation_sql = generation_sql[:-2] + " from df_gen_all t1 left join df_gen t2 on "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + "=t2." + str(item) + " and "
            generation_sql = generation_sql[:-5]
            df_gen_all = self.spark.sql(generation_sql).fillna(na)
            # print('*', 'generation_sql', generation_sql)

            # 清理数据
            self.spark.catalog.dropTempView("df_gen")
            df_gen = None
            self.spark.catalog.dropTempView("df_gen_all")
            df_gen_all.createTempView("df_gen_all")
            return df_gen_all, df_gen

        # 整理所需生成特征的信息
        feat_available_generation_rules_dict = {}
        for index in range(len(feature_generation_selection_scheme)):
            generation_item = feature_generation_selection_scheme[index]
            column_name, generation_rule, generation_rule_params = generation_item
            if generation_rule not in feat_available_generation_rules_dict.keys():
                feat_available_generation_rules_dict[generation_rule] = []
            feat_available_generation_rules_dict[generation_rule].append([column_name, generation_rule_params, index])
        used_generation_rules = list(feat_available_generation_rules_dict.keys())

        # 开始根据特征生成规则，生成新的特征实验数据
        gen_column_names = []
        generation_sql = "select "
        for item in self.data_config['instance_id_columns']:
            generation_sql = generation_sql + str(item) + ", "
            gen_column_names.append(item)
        if self.data_config['label_partition_column'] not in self.data_config['instance_id_columns']:
            generation_sql = generation_sql + str(self.data_config['label_partition_column']) + ", "
            gen_column_names.append(self.data_config['label_partition_column'])
        if 'row_num' not in gen_column_names:
            generation_sql = generation_sql + "FLOAT(rand_num) as rand_num, INT(row_num) as row_num, INT(" + str(
                self.data_config['label_column']) + ") as label from " + sample_infos['label_table_name'] + \
                             sample_infos['flag']
            gen_column_names = gen_column_names + ['rand_num', 'row_num', 'label']
        else:
            generation_sql = generation_sql + "FLOAT(rand_num) as rand_num, INT(" + str(
                self.data_config['label_column']) + ") as label from " + sample_infos['label_table_name'] + \
                             sample_infos['flag']
            gen_column_names = gen_column_names + ['rand_num', 'label']
        initial_gen_column_names = list(gen_column_names)
        print('initial_gen_column_names', initial_gen_column_names)
        df_gen_all = self.spark.sql(generation_sql)
        try:
            df_gen_all.createTempView("df_gen_all")
        except:
            self.spark.catalog.dropTempView("df_gen_all")
            df_gen_all.createTempView("df_gen_all")
        get_column_names_rule_details = {}

        start_time = time.time()

        # 生成数值特征的全部聚合型新特征 -- 0
        if 'num_agg_all_gen' in used_generation_rules:
            # 生成全部聚合特征p
            newly_add_column_names = []
            generation_sql = "select "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + " as " + str(item) + ", "
            generation_sql_features = "select "
            for item in self.data_config['user_id_columns']:
                generation_sql_features = generation_sql_features + str(item) + ", "
            for item in [self.data_config['feature_partition_column']]:
                generation_sql_features = generation_sql_features + str(item) + ", " + "INT(" + str(
                    item) + ") as " + str(item) + "_databricks, "
            for column_name, generation_rule_params, index in feat_available_generation_rules_dict['num_agg_all_gen']:
                method = generation_rule_params[0]
                new_column_name = column_name + "_" + method
                if "FLOAT(" + column_name + ")" not in generation_sql_features:
                    generation_sql_features = generation_sql_features + "FLOAT(" + column_name + ") as " + column_name + ", "
                # 计算特征非零数量
                if method == 'count':
                    generation_sql = generation_sql + "sum(if(t2." + str(
                        self.data_config['feature_partition_column']) + "_databricks <= t1." + str(
                        self.data_config['user_feature_partition_range_columns'][
                            1]) + "_databricks and t2." + column_name + " > 0, 1, 0 ))" + " as " + new_column_name + ", "
                if method == 'strmax':
                    generation_sql = generation_sql + "max(if(t2." + str(
                        self.data_config['feature_partition_column']) + "_databricks <= t1." + str(
                        self.data_config['user_feature_partition_range_columns'][
                            1]) + "_databricks and length(t2." + column_name + ")<10, 0, size(split(t2." + column_name + ", ','))))" + " as " + new_column_name + ", "

                    # 计算特征聚合特征
                elif method in ['min', 'max', 'sum', 'avg', 'std', 'stddev']:  # @hok added
                    generation_sql = generation_sql + method + "(if(t2." + str(
                        self.data_config['feature_partition_column']) + "_databricks <= t1." + str(
                        self.data_config['user_feature_partition_range_columns'][
                            1]) + "_databricks, t2." + column_name + ", 0))" + " as " + new_column_name + ", "
                newly_add_column_names.append(new_column_name)
                get_column_names_rule_details[new_column_name] = [column_name, "num_agg_all_gen",
                                                                  generation_rule_params, index]
            generation_sql = generation_sql[:-2] + " from (select *, "
            for item in self.data_config['user_feature_partition_range_columns']:
                if item != None:
                    generation_sql = generation_sql + "INT(" + str(item) + ") as " + str(item) + "_databricks, "
            generation_sql = generation_sql[:-2] + " from " + sample_infos['label_table_name'] + sample_infos[
                'flag'] + ") t1 left join (" + generation_sql_features[:-2] + " from " + sample_infos[
                                 'feature_table_name'] + sample_infos['flag'] + ") t2 on "
            for item in self.data_config['user_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + "=t2." + str(item) + " and "
            generation_sql = generation_sql[:-5] + " where t1." + \
                             self.data_config['user_feature_partition_range_columns'][0] + "_databricks <= t2." + \
                             self.data_config['feature_partition_column'] + "_databricks group by "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + ", "
            generation_sql = generation_sql[:-2]
            df_gen = self.spark.sql(generation_sql).fillna(na)
            # print('0', 'generation_sql', (time.time()-start_time)/60.0, generation_sql)

            # 添加聚合特征至df_gen_all，并清理数据
            df_gen_all, df_gen = merge_new_columns(df_gen, gen_column_names, newly_add_column_names)
            print('0', 'ok', (time.time() - start_time) / 60.0)

        # 生成数值特征的聚合型新特征 -- 1
        if 'num_agg_gen' in used_generation_rules:
            # 生成聚合特征
            newly_add_column_names = []
            generation_sql = "select "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + " as " + str(item) + ", "
            generation_sql_features = "select "
            for item in self.data_config['user_id_columns']:
                generation_sql_features = generation_sql_features + str(item) + ", "
            for item in [self.data_config['feature_partition_column']]:
                generation_sql_features = generation_sql_features + str(item) + ", " + "INT(" + str(
                    item) + ") as " + str(item) + "_databricks, "
            for column_name, generation_rule_params, index in feat_available_generation_rules_dict['num_agg_gen']:
                method = generation_rule_params[0]
                parts = generation_rule_params[1]
                new_column_name = column_name + "_" + str(parts) + "p_" + method
                if "FLOAT(" + column_name + ")" not in generation_sql_features:
                    generation_sql_features = generation_sql_features + "FLOAT(" + column_name + ") as " + column_name + ", "
                # 计算特征非零数量
                if method == 'count':
                    if self.data_config['user_feature_partition_range_columns'][0] != None and self.data_config[
                        'user_feature_agg_start_direction'] == 'left':
                        generation_sql = generation_sql + "sum(if(t2." + str(
                            self.data_config['feature_partition_column']) + "_databricks < " + get_databricks_time_sql(
                            "t1." + str(self.data_config['user_feature_partition_range_columns'][0]) + "_databricks",
                            parts, timestamp_style, gap_addition) + " and t2." + str(
                            self.data_config['feature_partition_column']) + "_databricks >= t1." + str(
                            self.data_config['user_feature_partition_range_columns'][
                                0]) + "_databricks and t2." + column_name + " > 0, 1, 0 ))" + " as " + new_column_name + ", "  ### <= to <
                    else:
                        generation_sql = generation_sql + "sum(if(t2." + str(
                            self.data_config['feature_partition_column']) + "_databricks > " + get_databricks_time_sql(
                            "t1." + str(self.data_config['user_feature_partition_range_columns'][1]) + "_databricks",
                            (-1) * parts, timestamp_style,
                            gap_addition) + " and t2." + column_name + " > 0, 1, 0 ))" + " as " + new_column_name + ", "  ### >= to >

                # 计算特征聚合特征
                elif method in ['min', 'max', 'sum', 'avg', 'std']:
                    if self.data_config['user_feature_partition_range_columns'][0] != None and self.data_config[
                        'user_feature_agg_start_direction'] == 'left':
                        generation_sql = generation_sql + method + "(if(t2." + str(
                            self.data_config['feature_partition_column']) + "_databricks < " + get_databricks_time_sql(
                            "t1." + str(self.data_config['user_feature_partition_range_columns'][0]) + "_databricks",
                            parts, timestamp_style, gap_addition) + " and t2." + str(
                            self.data_config['feature_partition_column']) + "_databricks >= t1." + str(
                            self.data_config['user_feature_partition_range_columns'][
                                0]) + "_databricks, t2." + column_name + ", 0))" + " as " + new_column_name + ", "  ### <= to <
                    else:
                        generation_sql = generation_sql + method + "(if(t2." + str(
                            self.data_config['feature_partition_column']) + "_databricks > " + get_databricks_time_sql(
                            "t1." + str(self.data_config['user_feature_partition_range_columns'][1]) + "_databricks",
                            (-1) * parts, timestamp_style,
                            gap_addition) + ", t2." + column_name + ", 0))" + " as " + new_column_name + ", "  ### >= to >

                newly_add_column_names.append(new_column_name)
                get_column_names_rule_details[new_column_name] = [column_name, "num_agg_gen", generation_rule_params,
                                                                  index]
            generation_sql = generation_sql[:-2] + " from (select *, "
            for item in self.data_config['user_feature_partition_range_columns']:
                if item != None:
                    generation_sql = generation_sql + "INT(" + str(item) + ") as " + str(item) + "_databricks, "
            generation_sql = generation_sql[:-2] + " from " + sample_infos['label_table_name'] + sample_infos[
                'flag'] + ") t1 left join (" + generation_sql_features[:-2] + " from " + sample_infos[
                                 'feature_table_name'] + sample_infos['flag'] + ") t2 on "
            for item in self.data_config['user_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + "=t2." + str(item) + " and "
            generation_sql = generation_sql[:-5] + " where t1." + str(
                self.data_config['user_feature_partition_range_columns'][1]) + "_databricks >= t2." + str(
                self.data_config['feature_partition_column']) + "_databricks group by "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + ", "
            generation_sql = generation_sql[:-2]
            df_gen = self.spark.sql(generation_sql).fillna(na)
            # print('4', 'generation_sql', (time.time()-start_time)/60.0, generation_sql)

            # 添加聚合特征至df_gen_all，并清理数据
            df_gen_all, df_gen = merge_new_columns(df_gen, gen_column_names, newly_add_column_names)
            print('5', 'ok', (time.time() - start_time) / 60.0)

        # 生成数值特征的时序型新特征 -- 2
        if 'num_ts_gen' in used_generation_rules:
            max_parts, newly_add_column_names = 0, []
            generation_sql = "select "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + str(item) + "_t1 as " + str(item) + ", "
            generation_sql_features = "select "
            for item in self.data_config['user_id_columns']:
                generation_sql_features = generation_sql_features + str(item) + ", "
            for item in [self.data_config['feature_partition_column']]:
                generation_sql_features = generation_sql_features + str(item) + ", " + "INT(" + str(
                    item) + ") as " + str(item) + "_databricks, "
            for column_name, generation_rule_params, index in feat_available_generation_rules_dict['num_ts_gen']:
                method = generation_rule_params[0]
                parts = generation_rule_params[1]
                max_parts = max(max_parts, parts)
                new_column_name = column_name + "_" + str(parts) + "ts_" + method
                if "FLOAT(" + column_name + ")" not in generation_sql_features:
                    generation_sql_features = generation_sql_features + "FLOAT(" + column_name + ") as " + column_name + ", "
                if method in ['min', 'max', 'sum', 'avg', 'std']:
                    generation_sql = generation_sql + method + "(if(rn <= " + str(
                        parts) + ", " + column_name + ", 0 ))" + " as " + new_column_name + ", "
                elif method == 'max_min':
                    generation_sql = generation_sql + "nvl(max(if(rn <= " + str(
                        parts) + ", " + column_name + ", 0 )),0)-nvl(min(if(rn <= " + str(
                        parts) + ", " + column_name + ", 0 )),0)" + " as " + new_column_name + ", "
                elif method == 'max_day':
                    generation_sql = generation_sql + "nvl(max(if(rn <= " + str(
                        parts) + ", " + column_name + ", 0 )),0)-nvl(max(if(rn = 1, " + column_name + ", 0 )),0)" + " as " + new_column_name + ", "
                elif method == 'avg_day':
                    generation_sql = generation_sql + "nvl(avg(if(rn <= " + str(
                        parts) + ", " + column_name + ", 0 )),0)-nvl(max(if(rn = 1, " + column_name + ", 0 )),0)" + " as " + new_column_name + ", "
                elif method == 'day_div_avg':
                    generation_sql = generation_sql + "nvl(max(if(rn = 1, " + column_name + ", 0 )),0)/nvl(avg(if(rn <= " + str(
                        parts) + ", " + column_name + ", 0 )),0.01)" + " as " + new_column_name + ", "
                newly_add_column_names.append(new_column_name)
                get_column_names_rule_details[new_column_name] = [column_name, "num_ts_gen", generation_rule_params,
                                                                  index]
            generation_sql = generation_sql[:-2] + " from (select "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + " as " + str(item) + "_t1, "
            generation_sql = generation_sql + "t2.*, row_number() over (partition by "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + ", "
            generation_sql = generation_sql[:-2] + " order by t2." + self.data_config[
                'feature_partition_column'] + " desc) as rn from (" + sample_infos['label_table_name'] + sample_infos[
                                 'flag'] + ") t1 left join (" + generation_sql_features[:-2] + " from " + sample_infos[
                                 'feature_table_name'] + sample_infos['flag'] + ") t2 on "
            for item in self.data_config['user_id_columns']:
                generation_sql = generation_sql + "t1." + str(item) + "=t2." + str(item) + " and "
            generation_sql = generation_sql[:-5] + " where t2." + self.data_config[
                'feature_partition_column'] + " <= t1." + self.data_config['user_feature_partition_range_columns'][1]
            if self.data_config['user_feature_partition_range_columns'][0] != None:
                generation_sql = generation_sql + " and t2." + self.data_config[
                    'feature_partition_column'] + " >= t1." + self.data_config['user_feature_partition_range_columns'][
                                     0]  # !!! range changed!!!#
            generation_sql = generation_sql + ") where rn <= " + str(max_parts) + " group by "
            for item in self.data_config['instance_id_columns']:
                generation_sql = generation_sql + str(item) + "_t1, "
            generation_sql = generation_sql[:-2]
            df_gen = self.spark.sql(generation_sql).fillna(na)
            # print('6', 'generation_sql', (time.time()-start_time)/60.0, generation_sql)

            # 添加聚合特征至df_gen_all，并清理数据
            df_gen_all, df_gen = merge_new_columns(df_gen, gen_column_names, newly_add_column_names)
            print('7', 'ok', (time.time() - start_time) / 60.0)

        return df_gen_all, gen_column_names, get_column_names_rule_details

    def get_metric_score(self, metric, real_y, y_pred, y_pred_proba):
        # xgb评估指标
        # 0-1 classification task: auc, F1-score, topK-precision;
        # 0-N classification task: auc, F1-score;
        # Regression task: topK-precision, topK-mse;
        if metric == "auc":
            if self.data_config['autofe_task_type'] == '01-Class':  # 0-1 classification task
                score = metrics.roc_auc_score(real_y, y_pred_proba[:, 1])
            elif self.data_config['autofe_task_type'] == 'Multi-Class':  # 0-N classification task
                real_y_one_hot = label_binarize(real_y, classes=self.labels)
                score = metrics.roc_auc_score(real_y_one_hot, y_pred_proba, average='macro')
            else:
                score = 0
        elif metric == "F1-score":
            if self.data_config['autofe_task_type'] == '01-Class':  # 0-1 classification task
                score = metrics.f1_score(real_y, y_pred)
            elif self.data_config['autofe_task_type'] == 'Multi-Class':  # 0-N classification task
                score = metrics.f1_score(real_y, y_pred, average='macro')
            else:
                score = 0
        elif metric == "Acc":
            if self.data_config['autofe_task_type'] == '01-Class':  # 0-1 classification task
                acc = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == real_y[i]:
                        acc = acc + 1
                score = acc * 1.0 / len(y_pred)
            elif self.data_config['autofe_task_type'] == 'Multi-Class':  # 0-N classification task
                acc = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == real_y[i]:
                        acc = acc + 1
                score = acc * 1.0 / len(y_pred)
            else:
                score = 0
        else:
            score = 0
        return score

    def scheme_training(self, df_gen_all, gen_column_names):
        try:
            df_gen_all.createTempView("df_gen_all")
        except:
            self.spark.catalog.dropTempView("df_gen_all")
            df_gen_all.createTempView("df_gen_all")

        tmp = []
        for item in self.data_config['instance_id_columns']:
            tmp.append(item)
        if self.data_config['label_partition_column'] not in self.data_config['instance_id_columns']:
            tmp.append(self.data_config['label_partition_column'])
        if 'row_num' not in tmp:
            tmp = tmp + ['rand_num', 'row_num', 'label']
        else:
            tmp = tmp + ['rand_num', 'label']
        initial_gen_column_names = list(tmp)

        generation_sql = "select INT(label) as label, "
        select_columns = list(gen_column_names[len(initial_gen_column_names):])
        random.shuffle(select_columns)
        for column_name in select_columns:  # {label,} generated features ......
            generation_sql = generation_sql + column_name + ", "
        sql_autofe_train = generation_sql[:-2] + " from df_gen_all where rand_num>0.2 order by row_num desc"
        sql_autofe_valid = generation_sql[:-2] + " from df_gen_all where rand_num<=0.2 order by row_num desc"

        # 数据转换
        xy_train_df = self.spark.sql(sql_autofe_train).fillna(0)
        xy_train = xy_train_df.toPandas().to_numpy()
        xy_valid = self.spark.sql(sql_autofe_valid).fillna(0).toPandas().to_numpy()
        print("shape", xy_train.shape, xy_valid.shape)

        x_train, y_train = xy_train[:, 1:], xy_train[:, 0]
        x_valid, y_valid = xy_valid[:, 1:], xy_valid[:, 0]

        if self.data_config['autofe_task_type'] == '01-Class':  # 0-1 classification task
            objective = "binary:logistic"
        elif self.data_config['autofe_task_type'] == 'Multi-Class':  # 0-N classification task
            objective = "multi:softmax"
        elif self.data_config['autofe_task_type'] == 'Regression':  # Regression task
            objective = "reg:linear"

        scheme_model = xgb.XGBClassifier(
            objective=objective,
            nthread=24,
            n_estimators=self.data_config['xgb_n_estimators'],
            max_depth=6,
            booster='gbtree',
            gamma=0,
            reg_lambda=2,
            subsample=self.data_config['xgb_subsample'],
            colsample_bytree=self.data_config['xgb_colsample_bytree'],
            min_child_weight=1,
            eta=0.05,
            scale_pos_weight=2,
            max_leaves=2000,
            missing=-999,
            seed=2023
        )
        scheme_model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=10
        )

        # 验证集性能测试
        y_pred_valid = scheme_model.predict(x_valid)
        if self.data_config['autofe_task_type'] == '01-Class':  # 0-1 classification task
            y_pred_valid_proba = scheme_model.predict_proba(x_valid)
        elif self.data_config['autofe_task_type'] == 'Multi-Class':  # 0-N classification task
            y_pred_valid_proba = scheme_model.predict_proba(x_valid)
        else:
            y_pred_valid_proba = None
        y_valid = y_valid.astype(np.float32)

        labels = []
        for item in y_valid:
            if item not in labels:
                labels.append(item)
        self.labels = labels

        score_all, score_all_dict = 0, {}
        for metric in self.xgb_metric:
            score = self.get_metric_score(metric, y_valid, y_pred_valid, y_pred_valid_proba)
            score_all = score_all + score
            score_all_dict[metric] = score
            print(metric, score, score_all)

        # 数据清理
        xy_train, xy_valid = None, None
        x_train, y_train = None, None
        x_valid, y_valid = None, None
        df_gen_all = None
        self.spark.catalog.dropTempView("df_gen_all")
        return df_gen_all, score_all, score_all_dict, scheme_model, select_columns

    def scheme_evaluation(self, df_gen_all, gen_column_names_test, select_columns, scheme_model):
        try:
            df_gen_all.createTempView("df_gen_all")
        except:
            self.spark.catalog.dropTempView("df_gen_all")
            df_gen_all.createTempView("df_gen_all")

        generation_sql = "select INT(label) as label, "
        for column_name in select_columns:  # {label,} generated features ......
            if column_name not in gen_column_names_test:
                generation_sql = generation_sql + "0 as " + column_name + ", "
            else:
                generation_sql = generation_sql + column_name + ", "
        sql_comparison_test = generation_sql[:-2] + " from df_gen_all order by row_num desc"

        # 数据加载
        xy_test_df = self.spark.sql(sql_comparison_test).fillna(0)
        xy_test = xy_test_df.toPandas().to_numpy()
        print(xy_test.shape)

        x_test, y_test = xy_test[:, 1:], xy_test[:, 0]

        # 测试集性能检验
        y_pred = scheme_model.predict(x_test)
        if self.data_config['autofe_task_type'] == '01-Class':  # 0-1 classification task
            y_pred_proba = scheme_model.predict_proba(x_test)
        elif self.data_config['autofe_task_type'] == 'Multi-Class':  # 0-N classification task
            y_pred_proba = scheme_model.predict_proba(x_test)
        else:
            y_pred_proba = None
        y_test = y_test.astype(np.float32)

        score_all, score_all_dict = 0, {}
        for metric in self.xgb_metric:
            score = self.get_metric_score(metric, y_test, y_pred, y_pred_proba)
            score_all = score_all + score
            score_all_dict[metric] = score
            print(metric, score, score_all)

        # 数据清理
        xy_test = None
        x_test, y_test = None, None
        df_gen_all = None
        self.spark.catalog.dropTempView("df_gen_all")
        return df_gen_all, score_all, score_all_dict

    def create_tempt_view(self, autofe_sample_infos):
        if autofe_sample_infos['flag'] == '_trainval':
            df_label, df_feature, df_graph = self.trainval_tables
        else:
            df_label, df_feature, df_graph = self.test_tables

        try:
            df_label.createTempView(autofe_sample_infos['label_table_name'] + autofe_sample_infos['flag'])
        except:
            self.spark.catalog.dropTempView(autofe_sample_infos['label_table_name'] + autofe_sample_infos['flag'])
            df_label.createTempView(autofe_sample_infos['label_table_name'] + autofe_sample_infos['flag'])

        try:
            df_feature.createTempView(autofe_sample_infos['feature_table_name'] + autofe_sample_infos['flag'])
        except:
            self.spark.catalog.dropTempView(autofe_sample_infos['feature_table_name'] + autofe_sample_infos['flag'])
            df_feature.createTempView(autofe_sample_infos['feature_table_name'] + autofe_sample_infos['flag'])

        if df_graph != None:
            try:
                df_graph.createTempView(autofe_sample_infos['graph_table_name'] + autofe_sample_infos['flag'])
            except:
                self.spark.catalog.dropTempView(autofe_sample_infos['graph_table_name'] + autofe_sample_infos['flag'])
                df_graph.createTempView(autofe_sample_infos['graph_table_name'] + autofe_sample_infos['flag'])
        return

    def get_train_xy(self, feature_generation_selection_scheme):
        # 生成特征表格
        autofe_sample_infos = {
            'label_table_name': 'cntest_autofeL2_label_table',
            'feature_table_name': 'cntest_autofeL2_feature_table',
            'graph_table_name': 'cntest_autofeL2_graph_table',
            'flag': '_trainval',
        }
        self.create_tempt_view(autofe_sample_infos)
        df_gen_all, gen_column_names, get_column_names_rule_details = self.generate_features(autofe_sample_infos,
                                                                                             feature_generation_selection_scheme,
                                                                                             0)

        # 获取train_xy
        try:
            df_gen_all.createTempView("df_gen_all")
        except:
            self.spark.catalog.dropTempView("df_gen_all")
            df_gen_all.createTempView("df_gen_all")

        tmp = []
        for item in self.data_config['instance_id_columns']:
            tmp.append(item)
        if self.data_config['label_partition_column'] not in self.data_config['instance_id_columns']:
            tmp.append(self.data_config['label_partition_column'])
        if 'row_num' not in tmp:
            tmp = tmp + ['rand_num', 'row_num', 'label']
        else:
            tmp = tmp + ['rand_num', 'label']
        initial_gen_column_names = list(tmp)

        generation_sql = "select INT(label) as label, "
        select_columns = list(gen_column_names[len(initial_gen_column_names):])
        feature_indexes = []
        random.shuffle(select_columns)
        for column_name in select_columns:  # {label,} generated features ......
            generation_sql = generation_sql + column_name + ", "
            feature_indexes.append(get_column_names_rule_details[column_name][-1])
        sql_autofe_train = generation_sql[:-2] + " from df_gen_all order by row_num desc"

        # 数据转换
        xy_train_df = self.spark.sql(sql_autofe_train).fillna(0)
        xy_train = xy_train_df.toPandas().to_numpy()
        print("shape", xy_train.shape)

        x_train, y_train = xy_train[:, 1:], xy_train[:, 0]
        return x_train, y_train, feature_indexes

    def run(self, feature_generation_selection_scheme):
        # 计算validation性能
        autofe_sample_infos = {
            'label_table_name': 'cntest_autofeL2_label_table',
            'feature_table_name': 'cntest_autofeL2_feature_table',
            'graph_table_name': 'cntest_autofeL2_graph_table',
            'flag': '_trainval',
        }
        self.create_tempt_view(autofe_sample_infos)
        df_gen_all, gen_column_names, get_column_names_rule_details = self.generate_features(autofe_sample_infos,
                                                                                             feature_generation_selection_scheme,
                                                                                             0)
        df_gen_all, scheme_valid_score, scheme_valid_score_dict, scheme_model, select_columns = self.scheme_training(
            df_gen_all, gen_column_names)

        # 计算test性能
        comparison_sample_infos = {
            'label_table_name': 'cntest_autofeL2_label_table',
            'feature_table_name': 'cntest_autofeL2_feature_table',
            'graph_table_name': 'cntest_autofeL2_graph_table',
            'flag': '_test',
        }
        self.create_tempt_view(comparison_sample_infos)
        df_gen_all, gen_column_names_test, _ = self.generate_features(comparison_sample_infos,
                                                                      feature_generation_selection_scheme, 0)
        df_gen_all, scheme_test_score, scheme_test_score_dict = self.scheme_evaluation(df_gen_all,
                                                                                       gen_column_names_test,
                                                                                       select_columns, scheme_model)
        # 输出特征名称及模型细节
        extra_details = {
            'scheme_model': scheme_model,
            'get_column_names_rule_details': get_column_names_rule_details
        }
        return scheme_valid_score, scheme_valid_score_dict, scheme_test_score, scheme_test_score_dict, select_columns, extra_details
