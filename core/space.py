# 根据任务确定搜索空间函数
class SearchSpace(object):
    def __init__(self):
        return

    def search_space_construction(self, data_config):
        # 确定配置参数
        df_autofe_label = self.df_autofe_label
        df_autofe_feature = self.df_autofe_feature
        instance_id_columns = data_config['instance_id_columns']
        user_id_columns = data_config['user_id_columns']
        feature_partition_column = data_config['feature_partition_column']
        label_feature_columns = data_config['label_feature_columns']
        feature_max_agg_length = data_config['feature_max_agg_length']
        label_table_used_source = data_config['label_table_used_source']

        # 明确各特征的含义类型
        feat_meaning_types = {
            'all_feats': []
        }
        all_feature_columns = list(df_autofe_feature.columns)
        for feat in all_feature_columns:
            if feat in instance_id_columns or feat in user_id_columns or feat == feature_partition_column:
                continue
            feat_meaning_types['all_feats'].append(feat)

        # 明确autofe算法可支持的特征生成方式
        if feature_max_agg_length < 1000:
            parts_list = [10, 20, 30]
            gap = 10
        else:
            parts_list = [100, 200, 300]
            gap = 100
        if feature_max_agg_length < parts_list[-1]:
            new_parts_list = []
            for item in parts_list:
                if item <= feature_max_agg_length:
                    new_parts_list.append(item)
            parts_list = list(new_parts_list)
        else:
            while feature_max_agg_length >= parts_list[-1]:
                parts_list.append(parts_list[-1] + gap)
            parts_list = list(parts_list[:-1])
        if label_table_used_source == 'public':
            generation_rules = {
                'num_agg_all_gen': {
                    'description': "generate all aggregation features for users",
                    'method': ['min', 'max', 'sum', 'avg', 'std', 'count', 'stddev'],
                    'gen_name_rules': '{feat}_{method}'
                },
                'num_agg_gen': {
                    'description': "generate aggregation features for users",
                    'method': ['min', 'max', 'sum', 'avg', 'std', 'count'],
                    'parts': list(parts_list),
                    'gen_name_rules': '{feat}_{parts}p_{method}'
                },
                'num_ts_gen': {
                    'description': "generate temporal features for users",
                    'method': ['min', 'max', 'sum', 'avg', 'std', 'max_min', 'max_day', 'avg_day', 'day_div_avg'],
                    'parts': list(parts_list),
                    'gen_name_rules': '{feat}_{parts}ts_{method}'
                }
            }

        # 获取单日特征表格中各特征的可选特征生成方式
        string_class_columns = []
        int_class_columns = []
        skip_columns = []
        label_feature_same_columns = []

        for (feat_name, feat_type) in df_autofe_feature.dtypes:
            if feat_name in instance_id_columns or feat_name in user_id_columns or feat_name == feature_partition_column:
                skip_columns.append(feat_name)
                continue
            if feat_type == 'string' or feat_type == "STRING":
                string_class_columns.append(feat_name)
            if feat_name in df_autofe_label.columns:
                label_feature_same_columns.append(feat_name)
        for (feat_name, feat_type) in df_autofe_label.dtypes:
            if feat_name in label_feature_columns:
                if feat_type in ['string', 'STRING'] and feat_name not in string_class_columns:
                    string_class_columns.append(feat_name)
                elif feat_type in ['BIGINT', 'bigint', 'INT', 'int'] and feat_name not in int_class_columns:
                    int_class_columns.append(feat_name)
                if feat_name in df_autofe_feature.columns and feat_name not in label_feature_same_columns:
                    label_feature_same_columns.append(feat_name)
        print("string_class_columns", string_class_columns)
        print("int_class_columns", int_class_columns)
        print("skip_columns", skip_columns)
        print("label_feature_same_columns", label_feature_same_columns)

        feat_available_generation_rules = {}
        for meaning_type in feat_meaning_types.keys():
            for column_name in feat_meaning_types[meaning_type]:
                if column_name in skip_columns:
                    continue
                feat_available_generation_rules[column_name] = ['num_agg_all_gen', 'num_agg_gen', 'num_ts_gen']
        return feat_meaning_types, feat_available_generation_rules, generation_rules, label_feature_same_columns

    def run(self, data_config, trainval_tables):
        self.df_autofe_label, self.df_autofe_feature, _ = trainval_tables
        return self.search_space_construction(data_config)
