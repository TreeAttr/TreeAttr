import copy
import pandas as pd
import numpy as np
import os
import json


# 根据任务名称获取数据细节，以及训练集、验证集、测试集数据
class Datset(object):
    def __init__(self, autofe_task_name, spark):
        self.autofe_task_name = autofe_task_name
        self.spark = spark
        return

    def public_shared_config(self):
        shared_config = {
            'label_table_used_source': 'public',
            'label_table': '',
            'label_partition_column': 'row_num',
            'label_column': 'label',  # 标签名称
            'instance_id_columns': ['ID', 'first_timestep', 'last_timestep'],  # 实例唯一识别列
            'user_id_columns': ['ID'],  # 用户唯一识别列
            'user_feature_partition_range_columns': ['first_timestep', 'last_timestep'],  # 用户特征表分区值需要在两个column范围内
            'label_feature_columns': [],  # 标签表格中可以当做特征的column
            'user_feature_agg_start_direction': 'right',
            # 用户特征起始点如果为right表示将user_feature_partition_range_columns[-1]作为特征起始点分区，choices: left, right (default)
            # 用户最后一次登录时间与user_feature_partition_range_columns[1]的时间间隔column，没有则为None
            'label_train_partition_range': ['', ''],  # 训练集分区范围
            'label_test_partition_range': ['', ''],  # 测试集分区范围

            'feature_table_used_source': 'public',  # 算法使用的特征表格来源，choices: parquet, dbfs, csv
            'feature_table': '',  # databricks DBFS路径/parquet路径（推荐）
            'feature_partition_column': 'timestep',  # 分区名称
            'feature_train_partition_range': ['', ''],  # 训练集分区范围
            'feature_test_partition_range': ['', ''],  # 测试集分区范围
            'feature_max_agg_length': -1,  # 用户最多聚合的特征长度

            'autofe_task_type': 'Multi-Class',
            # 预测任务类型，（1）0-1 classification task: 01-Class;（2）0-N classification task: Multi-Class;（3）Regression task: Regression
            'xgb_n_estimators': 10,  # 算法中xgb训练轮数
            'xgb_subsample': 1,  # 算法中xgb sub采样比例
            'xgb_colsample_bytree': 0.7  # 算法中xgb col采样比例
        }
        return shared_config

    def get_data_config(self, public_regen=False):
        if self.autofe_task_name == "public_user_KKBox":
            shared_config = self.public_shared_config()
            data_config = copy.deepcopy(shared_config)

            data_config['label_table'] = data_config['label_table'] + "_KKBox"
            data_config['label_train_partition_range'] = [1, 500000]
            data_config['label_test_partition_range'] = [500001, 970960]
            data_config['feature_table'] = data_config['feature_table'] + "_KKBox"
            data_config['feature_train_partition_range'] = ['20150101', '20170331']
            data_config['feature_test_partition_range'] = ['20150101', '20170331']
            data_config['feature_max_agg_length'] = 819
            data_config['autofe_task_type'] = '01-Class'

            path = os.path.dirname(os.path.abspath(__file__)) + "/data/KKBox"
            self.df_autofe_label_all, self.df_autofe_feature_all = self.read_dataset_KKBox(path)
        return data_config

    def read_dataset_KKBox(self, path):
        # Step 1: transactions_v2.csv
        # msno, transaction_date, payment_method_id|payment_plan_days|plan_list_price|actual_amount_paid|is_auto_renew|is_cancel, left_days=membership_expire_date-transaction_date (f1-f7)
        df_trans = self.spark.read.csv(path + '/transactions_v2.csv', header="true", inferSchema="true")
        try:
            df_trans.createTempView('df_trans')
        except:
            self.spark.catalog.dropTempView('df_trans')
            df_trans.createTempView('df_trans')

        generation_sql = "select msno as ID, transaction_date as timestep, payment_method_id as f1, payment_plan_days as f2, plan_list_price as f3, actual_amount_paid as f4, is_auto_renew as f5, is_cancel as f6, INT((unix_timestamp(STRING(membership_expire_date), 'yyyyMMdd')-unix_timestamp(STRING(transaction_date), 'yyyyMMdd'))/3600/24) as f7 from df_trans"
        df_trans = self.spark.sql(generation_sql).fillna(0)
        try:
            df_trans.createTempView('df_trans_all')
        except:
            self.spark.catalog.dropTempView('df_trans_all')
            df_trans.createTempView('df_trans_all')

        # Step 2: user_logs_v2_split1.csv-user_logs_v2_split5.csv
        # msno, date, num_25|num_50|num_75|num_985|num_100|num_unq|total_secs (f8-f14)
        file_names = ['user_logs_v2_split1.csv', 'user_logs_v2_split2.csv', 'user_logs_v2_split3.csv',
                      'user_logs_v2_split4.csv', 'user_logs_v2_split5.csv']
        df_names = []
        for file_name in file_names:
            print(file_name)
            # df_log = pd.read_csv(path + '/' + file_name)
            df_log = self.spark.read.csv(path + '/' + file_name, header="true", inferSchema="true")
            # df_log = self.spark.createDataFrame(df_log)
            try:
                df_log.createTempView(file_name[:-4])
            except:
                self.spark.catalog.dropTempView(file_name[:-4])
                df_log.createTempView(file_name[:-4])

            generation_sql = "select msno as ID, date as timestep, num_25 as f8, num_50 as f9, num_75 as f10, num_985 as f11, num_100 as f12, num_unq as f13, total_secs as f14 from " + file_name[
                                                                                                                                                                                         :-4]
            df_log = self.spark.sql(generation_sql).fillna(0)
            try:
                df_log.createTempView(file_name[:-4])
            except:
                self.spark.catalog.dropTempView(file_name[:-4])
                df_log.createTempView(file_name[:-4])
            df_names.append(file_name[:-4])

        generation_sql = "select * from ("
        for df_name in df_names:
            generation_sql = generation_sql + "select * from " + df_name + " UNION ALL "
        generation_sql = generation_sql[:-11] + ")"
        df_log = self.spark.sql(generation_sql).fillna(0)
        try:
            df_log.createTempView('df_log_all')
        except:
            self.spark.catalog.dropTempView('df_log_all')
            df_log.createTempView('df_log_all')

        # Step 3: members_v3_split1.csv
        # msno, city|bd|registered_via
        # df_memb = pd.read_csv(path + '/members_v3_split1.csv')
        df_memb = self.spark.read.csv(path + '/members_v3_split1.csv', header="true", inferSchema="true")
        # df_memb = self.spark.createDataFrame(df_memb)
        try:
            df_memb.createTempView('df_memb')
        except:
            self.spark.catalog.dropTempView('df_memb')
            df_memb.createTempView('df_memb')

        generation_sql = "select msno as ID, city as f15, bd as f16, registered_via as f17 from df_memb"
        df_memb = self.spark.sql(generation_sql).fillna(0)
        try:
            df_memb.createTempView('df_memb_all')
        except:
            self.spark.catalog.dropTempView('df_memb_all')
            df_memb.createTempView('df_memb_all')

        # Merge all features
        # 生成feature表格
        # feature_names = ['ID', 'timestep'] + ['f' + str(i+1) for i in range(17)]
        generation_sql = "select t1.ID as ID, timestep, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17 from (select if(a.ID is null, b.ID, a.ID) as ID, if(a.timestep is null, b.timestep, a.timestep) as timestep, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 from(select * from df_log_all) a full join (select * from df_trans_all) b on a.ID=b.ID and a.timestep=b.timestep) t1 left join (select * from df_memb_all) t2 on t1.ID=t2.ID"
        df_autofe_feature = self.spark.sql(generation_sql).fillna(0)
        print(df_autofe_feature.show())
        try:
            df_autofe_feature.createTempView('df_autofe_feature')
        except:
            self.spark.catalog.dropTempView('df_autofe_feature')
            df_autofe_feature.createTempView('df_autofe_feature')

        # Step 4: train_v2.csv
        # msno, is_churn
        # df_train = pd.read_csv(path + '/train_v2.csv')
        df_train = self.spark.read.csv(path + '/train_v2.csv', header="true", inferSchema="true")
        # df_train = self.spark.createDataFrame(df_train)
        try:
            df_train.createTempView('df_train')
        except:
            self.spark.catalog.dropTempView('df_train')
            df_train.createTempView('df_train')

        # 生成label表格 (ID, label, first_timestep, last_timestep, rand_num, row_num)
        label_names = ['ID', 'label', 'first_timestep', 'last_timestep', 'rand_num']
        generation_sql = "select "
        for name in label_names:
            generation_sql = generation_sql + name + ", "
        generation_sql = generation_sql + "row_number() over (order by rand_num asc) as row_num from(select t1.*, first_timestep, last_timestep, rand() as rand_num from (select msno as ID, is_churn as label from df_train) t1 left join (select ID, min(timestep) as first_timestep, max(timestep) as last_timestep from df_autofe_feature group by ID) t2 on t1.ID = t2.ID)"
        df_autofe_label = self.spark.sql(generation_sql).fillna(0)
        print(df_autofe_label.show())
        return df_autofe_label, df_autofe_feature

    def get_trainval_tables(self, data_config):
        df_autofe_label, df_autofe_feature, df_autofe_graph = None, None, None

        # 公开用户数据集
        if self.autofe_task_name in ["public_user_KKBox"]:
            try:
                self.df_autofe_label_all.createTempView('df_autofe_label_all')
            except:
                self.spark.catalog.dropTempView('df_autofe_label_all')
                self.df_autofe_label_all.createTempView('df_autofe_label_all')
            try:
                self.df_autofe_feature_all.createTempView('df_autofe_feature_all')
            except:
                self.spark.catalog.dropTempView('df_autofe_feature_all')
                self.df_autofe_feature_all.createTempView('df_autofe_feature_all')
            generation_sql = "select * from df_autofe_label_all where row_num between " + str(
                data_config['label_train_partition_range'][0]) + " and " + str(
                data_config['label_train_partition_range'][1])
            df_autofe_label = self.spark.sql(generation_sql).fillna(0)
            generation_sql = "select * from df_autofe_feature_all where timestep between " + str(
                data_config['feature_train_partition_range'][0]) + " and " + str(
                data_config['feature_train_partition_range'][1])
            df_autofe_feature = self.spark.sql(generation_sql).fillna(0)

        df_trainval_label, df_trainval_feature, df_trainval_graph = df_autofe_label, df_autofe_feature, df_autofe_graph
        return df_trainval_label, df_trainval_feature, df_trainval_graph

    def get_test_tables(self, data_config):
        df_comparison_label, df_comparison_feature, df_comparison_graph = None, None, None

        # 公开用户数据集
        if self.autofe_task_name in ["public_user_KKBox"]:
            try:
                self.df_autofe_label_all.createTempView('df_autofe_label_all')
            except:
                self.spark.catalog.dropTempView('df_autofe_label_all')
                self.df_autofe_label_all.createTempView('df_autofe_label_all')
            try:
                self.df_autofe_feature_all.createTempView('df_autofe_feature_all')
            except:
                self.spark.catalog.dropTempView('df_autofe_feature_all')
                self.df_autofe_feature_all.createTempView('df_autofe_feature_all')
            generation_sql = "select * from df_autofe_label_all where row_num between " + str(
                data_config['label_test_partition_range'][0]) + " and " + str(
                data_config['label_test_partition_range'][1])
            df_comparison_label = self.spark.sql(generation_sql).fillna(0)
            generation_sql = "select * from df_autofe_feature_all where timestep between " + str(
                data_config['feature_test_partition_range'][0]) + " and " + str(
                data_config['feature_test_partition_range'][1])
            df_comparison_feature = self.spark.sql(generation_sql).fillna(0)

        df_test_label, df_test_feature, df_test_graph = df_comparison_label, df_comparison_feature, df_comparison_graph
        return df_test_label, df_test_feature, df_test_graph
