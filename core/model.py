# 实现新的autofe算法 -- 算法中的辅助函数及类
import torch.nn as nn
import torch

# 核心分析模型，判断特定特征的重要性、特征对的重要性
class Score_Imp_Path_Pair_Model(nn.Module):
    def __init__(self, n_feature, n_type, n_rule, n_method, n_day, embed_dim=32, num_heads=4):
        super(Score_Imp_Path_Pair_Model, self).__init__()
        # feature: [feature, (type, rule, method, days, friend_feat_method)]
        self.feat_emb = nn.Embedding(n_feature, embed_dim)
        self.type_emb = nn.Embedding(n_type, embed_dim)
        self.rule_emb = nn.Embedding(n_rule, embed_dim)
        self.meth_emb = nn.Embedding(n_method, embed_dim)
        self.day_emb = nn.Embedding(n_day, embed_dim)
        self.feat_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # score_imp model
        self.score_imp_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.ReLU(),
            nn.Linear(embed_dim, 4) # score_max, score_avg, imp_max, imp_avg
        )
        # tree_path model
        self.path_rnn = nn.GRU(embed_dim, embed_dim, 2, batch_first=True)
        self.path_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid() # possibility
        )
        self.embed_dim = embed_dim
        # pair model
        self.pair_out = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim, bias=True),
            nn.ReLU(),
            nn.Linear(embed_dim, 2) # score_max, score_avg
        )
        return

    def forward(self, feature_info, path_info=None, pair_info=None, flag="score_imp"):
        # print("feature_info", feature_info)
        if feature_info != None:
            # BxE
            feature1 = self.feat_emb(torch.LongTensor([item[0] for item in feature_info]))
            type1 = self.type_emb(torch.LongTensor([item[1] for item in feature_info]))
            rule1 = self.rule_emb(torch.LongTensor([item[2] for item in feature_info]))
            method1 = self.meth_emb(torch.LongTensor([item[3] for item in feature_info]))
            day1 = self.day_emb(torch.LongTensor([item[4] for item in feature_info]))
            friend_method1 = self.meth_emb(torch.LongTensor([item[5] for item in feature_info]))

            q = feature1.unsqueeze(1) # Bx1xE
            kv = torch.stack([type1, rule1, method1, day1, friend_method1], dim=1) # BxNxE
            enhanced_feature1 = self.feat_attn(q, kv, kv)[0].squeeze(1) # BxE
            # print("ok1", q.shape, kv.shape, enhanced_feature1.shape)

        if flag in ["score_imp", "score", "imp", "score_imp_predict"]:
            score_imp = self.score_imp_out(enhanced_feature1) # Bx4
            # print("ok2", score_imp.shape, type(score_imp))
            if flag == "score_imp_predict":
                return score_imp
            res = []
            for i in range(score_imp.shape[0]):
                res.append([i, feature_info[i], score_imp[i][0], score_imp[i][1], score_imp[i][2], score_imp[i][3]])
            if flag == "score":
                res = sorted(res, key=lambda x:x[2]+x[3], reverse=True)
            elif flag == "imp":
                res = sorted(res, key=lambda x:x[4]+x[5], reverse=True)
            elif flag == "score_imp":
                res = sorted(res, key=lambda x:x[3]*x[5]+x[3], reverse=True)
        elif flag in ["tree_path", "tree_path_predict"]:
            max_seq = 0
            for i in range(len(path_info)):
                max_seq = max(max_seq, len(path_info[i]))
            feat_sum = len(path_info)*max_seq
            # Bxmax_seqxE->(B*max_seq)xE
            features = self.feat_emb(torch.LongTensor([[item[0] for item in seq]+[0]*(max_seq-len(seq)) for seq in path_info])).reshape(feat_sum, -1)
            types = self.type_emb(torch.LongTensor([[item[1] for item in seq]+[0]*(max_seq-len(seq)) for seq in path_info])).reshape(feat_sum, -1)
            rules = self.rule_emb(torch.LongTensor([[item[2] for item in seq]+[0]*(max_seq-len(seq)) for seq in path_info])).reshape(feat_sum, -1)
            methods = self.meth_emb(torch.LongTensor([[item[3] for item in seq]+[0]*(max_seq-len(seq)) for seq in path_info])).reshape(feat_sum, -1)
            days = self.day_emb(torch.LongTensor([[item[4] for item in seq]+[0]*(max_seq-len(seq)) for seq in path_info])).reshape(feat_sum, -1)
            friend_methods = self.meth_emb(torch.LongTensor([[item[5] for item in seq]+[0]*(max_seq-len(seq)) for seq in path_info])).reshape(feat_sum, -1)

            qs = features.unsqueeze(1) # (B*max_seq)x1xE
            kvs = torch.stack([types, rules, methods, days, friend_methods], dim=1) # (B*max_seq)xNxE
            enhanced_features = self.feat_attn(qs, kvs, kvs)[0] # (B*max_seq)x1xE
            enhanced_features = enhanced_features.reshape(len(path_info), max_seq, -1) # Bxmax_seqxE

            h0 = torch.zeros(2, len(path_info), self.embed_dim)
            rnn_out, hn = self.path_rnn(enhanced_features, h0) # Bxmax_seqxE, 2xBxE
            tree_path_score = self.path_out(rnn_out) # Bxmax_seqx1
            tree_path_score = torch.sum(tree_path_score, dim=1).reshape(len(path_info)) # B
            if flag == "tree_path_predict":
                return tree_path_score
            res = []
            for i in range(tree_path_score.shape[0]):
                res.append([i, path_info[i], tree_path_score[i].item()])
            res = sorted(res, key=lambda x:x[2], reverse=True)
        elif flag in ["pair", "pair_predict"]:
            feature2 = self.feat_emb(torch.LongTensor([item[0] for item in pair_info]))
            type2 = self.type_emb(torch.LongTensor([item[1] for item in pair_info]))
            rule2 = self.rule_emb(torch.LongTensor([item[2] for item in pair_info]))
            method2 = self.meth_emb(torch.LongTensor([item[3] for item in pair_info]))
            day2 = self.day_emb(torch.LongTensor([item[4] for item in pair_info]))
            friend_method2 = self.meth_emb(torch.LongTensor([item[5] for item in pair_info]))

            q2 = feature2.unsqueeze(1) # Bx1xE
            kv2 = torch.stack([type2, rule2, method2, day2, friend_method2], dim=1) # BxNxE
            enhanced_feature2 = self.feat_attn(q2, kv2, kv2)[0].squeeze(1) # BxE
            # print("ok3", q2.shape, kv2.shape, enhanced_feature2.shape)

            enhanced_feature12 = torch.cat([enhanced_feature1, enhanced_feature2], dim=1) # Bx2E
            pair_score = self.pair_out(enhanced_feature12) # Bx2
            # print("ok4", pair_score.shape, type(pair_score))
            if flag == "pair_predict":
                return pair_score
            res = []
            for i in range(pair_score.shape[0]):
                res.append([i, [feature_info[i], pair_info[i]], pair_score[i][0], pair_score[i][1]])
            res = sorted(res, key=lambda x:x[2]+x[3], reverse=True)
        return res
