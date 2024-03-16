import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand

residue2idx = pickle.load(open('./dataset/meta_data/residue2idx.pkl', 'rb'))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]

        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]

        # expand_as类似于expand，只是目标规格是x.shape
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]

        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)

        # layerNorm
        embedding = self.norm(embedding)
        return embedding

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 多头注意力模块
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        # 全连接模块
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        # transformer encoder
        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        print('config.max_len:')
        print(config.max_len)
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = len(residue2idx)
        device = torch.device("cuda")
        ways = 2
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 定义重复的模块

        # textcnn
        dim_cnn_out = 128
        filter_num = config.num_filter
        filter_sizes = [int(fsz) for fsz in config.filter_sizes.split(',')]
        vocab_size = len(residue2idx)
        embedding_dim = config.dim_embedding_cnn

        self.embedding_cnn = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.linear_cnn = nn.Linear(len(filter_sizes) * filter_num, dim_cnn_out)

        self.fc_task = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.classifier = nn.Linear(d_model // 2, ways)

    def forward(self, input_ids, label_input=False, label=None, epoch_num=None):

        if label_input:
            sum_encoder = input_ids
            X_tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30).fit_transform(
                sum_encoder.cpu().detach().numpy())
            cmap = ListedColormap(['#00beca', '#f87671'])
            plt.subplots()
            plt.title("T-SNE")
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=cmap, c=label, s=2)
            plt.savefig(f'./epoch_scatter_3D/test_{str(epoch_num)}_start.png')

        output_encoder = self.embedding(input_ids)

        if label_input:
            sum_encoder = torch.sum(output_encoder, dim=-2)
            X_tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30).fit_transform(
                sum_encoder.cpu().detach().numpy())
            cmap = ListedColormap(['#00beca', '#f87671'])
            plt.subplots()
            plt.title("T-SNE")
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=cmap, c=label, s=2)
            plt.savefig(f'./epoch_scatter_3D/test_{str(epoch_num)}_embedding.png')

        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]

        for layer in self.layers:
            output_encoder = layer(output_encoder, enc_self_attn_mask)


        output_encoder = output_encoder[:, 0, :]
        if label_input:
            sum_encoder = output_encoder
            X_tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30).fit_transform(
                sum_encoder.cpu().detach().numpy())
            cmap = ListedColormap(['#00beca', '#f87671'])
            plt.subplots()
            plt.title("T-SNE")
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=cmap, c=label, s=2)
            plt.savefig(f'./epoch_scatter_3D/test_{str(epoch_num)}_CLS.png')

        feature_cnn = self.embedding_cnn(input_ids)
        feature_cnn = feature_cnn.view(feature_cnn.size(0), 1, feature_cnn.size(1), self.config.dim_embedding)
        feature_cnn = [F.relu(conv(feature_cnn)) for conv in self.convs]
        feature_cnn = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in
                       feature_cnn]
        feature_cnn = [x_item.view(x_item.size(0), -1) for x_item in feature_cnn]
        feature_cnn = torch.cat(feature_cnn, 1)
        feature_cnn = self.linear_cnn(feature_cnn)
        if label_input:
            sum_encoder = feature_cnn
            X_tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30).fit_transform(
                sum_encoder.cpu().detach().numpy())
            cmap = ListedColormap(['#00beca', '#f87671'])
            plt.subplots()
            plt.title("T-SNE")
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=cmap, c=label, s=2)
            plt.savefig(f'./epoch_scatter_3D/test_{str(epoch_num)}_CNN.png')

        feature_new = torch.cat((output_encoder, feature_cnn), dim=1)

        if label_input:
            sum_encoder = feature_new
            X_tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30).fit_transform(
                sum_encoder.cpu().detach().numpy())
            cmap = ListedColormap(['#00beca', '#f87671'])
            plt.subplots()
            plt.title("T-SNE")
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=cmap, c=label, s=2)

            plt.savefig(f'./epoch_scatter_3D/test_{str(epoch_num)}_feature_new_256.png')

        feature_new = self.fc_task(feature_new)

        if label_input:
            sum_encoder = feature_new
            X_tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30).fit_transform(
                sum_encoder.cpu().detach().numpy())
            cmap = ListedColormap(['#00beca', '#f87671'])
            plt.subplots()

            plt.title("T-SNE")
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=cmap, c=label, s=2)

            plt.savefig(f'./epoch_scatter_3D/test_{str(epoch_num)}_feature_new_64.png')
        logits_clsf = self.classifier(feature_new)

        if label_input:
            sum_encoder = logits_clsf

            X_tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=30).fit_transform(
                sum_encoder.cpu().detach().numpy())
            cmap = ListedColormap(['#00beca', '#f87671'])
            plt.subplots()
            plt.title("T-SNE")
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=cmap, c=label, s=2)
            plt.savefig(f'./epoch_scatter_3D/test_{str(epoch_num)}_result.png')
        embeddings = feature_new.view(feature_new.size(0), -1)

        return logits_clsf, embeddings

class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer, self).__init__()
        self.config = config

        # transformer encoder
        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = len(residue2idx)
        device = torch.device("cuda")
        ways = 2
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 定义重复的模块

        # 分类
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.classifier = nn.Linear(d_model // 2, ways)

    def forward(self, input_ids, label_input=False, label=None, epoch_num=None):
        # 输入x的形状应该是 [batch_size, sequence_length]
        output_encoder = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)

        for layer in self.layers:
            output_encoder = layer(output_encoder, enc_self_attn_mask)

        output_encoder = output_encoder[:, 0, :]

        # 经过全连接层进行分类任务
        output_encoder = self.fc_task(output_encoder)
        output = self.classifier(output_encoder)

        return output,output_encoder.view(output_encoder.size(0),-1)


class CnnModel(nn.Module):
    def __init__(self,config):
        super(CnnModel, self).__init__()
        self.config = config
        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        ways=2

        dim_cnn_out = 128
        filter_num = config.num_filter
        filter_sizes = [int(fsz) for fsz in config.filter_sizes.split(',')]
        vocab_size = len(residue2idx)
        embedding_dim = config.dim_embedding_cnn

        self.embedding_cnn = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.linear_cnn = nn.Linear(len(filter_sizes) * filter_num, dim_cnn_out)

        # 分类
        self.fc_task = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),  # 调整输入维度
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),  # 调整输出维度
        )
        self.classifier = nn.Linear(embedding_dim // 4, ways)  # 调整输出维度

    def forward(self, input_ids, label_input=False, label=None, epoch_num=None):
        feature_cnn = self.embedding_cnn(input_ids)
        feature_cnn = feature_cnn.view(feature_cnn.size(0), 1, feature_cnn.size(1), self.config.dim_embedding)
        feature_cnn = [F.relu(conv(feature_cnn)) for conv in self.convs]
        feature_cnn = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in
                       feature_cnn]
        feature_cnn = [x_item.view(x_item.size(0), -1) for x_item in feature_cnn]
        feature_cnn = torch.cat(feature_cnn, 1)
        feature_cnn = self.linear_cnn(feature_cnn)

        feature_new = self.fc_task(feature_cnn)
        logits_clsf = self.classifier(feature_new)
        embeddings = feature_new.view(feature_new.size(0), -1)
        return logits_clsf, embeddings


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        # 多头注意力是同时计算的，一次tensor乘法即可，这里是将多头注意力进行切分
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        # context: [batch_size, n_head, seq_len, d_v], attn: [batch_size, n_head, seq_len, seq_len]
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]

        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))