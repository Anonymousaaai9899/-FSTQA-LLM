import numpy as np
from numpy import *
import scipy.sparse as sp
from tqdm import tqdm,trange
import pandas as pd
import random

import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.optim as optim
import torch.nn as nn

import transformers
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


AA = np.load('normal_adjacent.npy')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    """compute L=D^-0.5 * (A+I) * D^-0.5"""
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj += sp.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    norm_adj = d_hat.dot(adj).dot(d_hat)
    return sparse_mx_to_torch_sparse_tensor(norm_adj)

def normalize(mx):
    """"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


adj0 = sp.coo_matrix(AA, dtype=np.float32)
adj0 = adj0 + adj0.T.multiply(adj0.T > adj0) - adj0.multiply(adj0.T > adj0)
adj0 = sparse_mx_to_torch_sparse_tensor((adj0 + sp.eye(adj0.shape[0])))



import datetime
 
def get_weekday(date):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # 将输入的日期字符串转换为datetimemd
    date_obj = datetime.datetime.strptime(date, '%m/%d/%Y')
    
    # 获取该日期所在的星期几（0表示星期一）
    weekday_index = date_obj.weekday()
    
    return weekdays[weekday_index]

import holidays

us_holidays = holidays.US()
def holiday_(date):
    if date in us_holidays:
        return ', holiday'
    else:
        return ''

A_com = np.load(r'normal_adjacent.npy')
connections = []
for i in range(78):
    for j in range(i, 78):
        if A_com[i][j] == 1:
            connections.append('(r'+str(i)+', r' + str(j)+')')
            
xx = ''
for i in range(len(connections)):
    xx = xx + connections[i]+','

regions  = 'r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37, r39, r39, r40, r41, r42, r43, r44, r45 r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77'


text_s = []
for i in tqdm(range(len(taxi_tensor_in))):
    text_s.append('At ' + text1[i]\
            +', there were' \
            + ' ' + str(list(map(int, taxi_tensor_in[i])))[1:-1] \
            + ' taxis visiting Community regions ' + regions +'.'\
            + ' The spatial sequence has a minimum of ' \
            + str(int(taxi_tensor_in[i].min())) \
            + ' at region ' + 'r'+ str(list(taxi_tensor_in[i]).index(min(taxi_tensor_in[i])))\
            +'; a second minimum of ' + str(int(sorted(taxi_tensor_in[i], reverse=False)[2]))\
            + ' at region ' + 'r'+ str(list(taxi_tensor_in[i]).index(int(sorted(taxi_tensor_in[i], reverse=False)[2])))\
            +'; a maximum of ' + str(int(taxi_tensor_in[i].max())) \
            +' at region ' + 'r'+ str(list(taxi_tensor_in[i]).index(max(taxi_tensor_in[i])))\
            +'; a second maximum of ' + str(int(sorted(taxi_tensor_in[i], reverse=False)[-2])) \
            +' at region ' + 'r'+ str(list(taxi_tensor_in[i]).index(sorted(taxi_tensor_in[i], reverse=False)[-2]))\
            + '; and a mean of '+str(round(taxi_tensor_in[i].mean(),2))\
            +'. There are spatial correlations between regions. Adjacent regions may affect each other.'\
            +'The regions in parentheses below are adjacent:' + xx[:-1]+'.')
    
def text_T(T,t,R):
    label = 'From ' + text1[T] + ', to ' + text1[T+t-1]\
            +', there were ' + str(list(map(int, taxi_tensor_in[T:T+t,R])))[1:-1]\
            + ' taxis visiting Community region ' + 'r'+str(R)+'.'\
            + ' The time interval is 1 hour.'\
            + ' The temporal sequence has a minimum of '\
            + str(int(taxi_tensor_in[T:T+t,R].min()))\
            + ' at ' + text1[T+list(taxi_tensor_in[T:T+t,R]).index(taxi_tensor_in[T:T+t,R].min())] \
            +'; a second minimum of ' + str(int(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[2]))\
            + ' at '+ text1[T+(list(taxi_tensor_in[T:T+t,R]).index(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[2]))]\
            +'; a maximum of ' + str(int(taxi_tensor_in[T:T+t,R].max()))\
            + ' at ' + text1[T+list(taxi_tensor_in[T:T+t,R]).index(taxi_tensor_in[T:T+t,R].max())] \
            +'; a second maximum of ' + str(int(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[-2])) \
            +' at '+ text1[T+(list(taxi_tensor_in[T:T+t,R]).index(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[-2]))]\
            + '; and a mean of '+str(round(taxi_tensor_in[T:T+t,R].mean(),2))+'.'
    return label



class GraphConvolution(nn.Module):
  
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_):
        support = torch.matmul(input, self.weight)
        output = torch.spmm(adj_, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc10 = GraphConvolution(nfeat, nhid)
        self.gc11 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(normalized_shape = 16, eps = 1e-6)
        self.dropout_layer = nn.Dropout(p = dropout)
        self.Embedding = Embedding()

    def forward(self, x, adj):
        x1 = self.Embedding(x.reshape([78,1]))
        x2 = self.gc11(F.dropout(F.relu(self.gc10(x1, adj)), self.dropout, training = self.training), adj)
        return self.dropout_layer(self.layer_norm(x2))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(in_features = 1, out_features = 16, bias=True)

    def forward(self, X_feature):
        X = self.linear(X_feature)
        return X


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class transformer_t(nn.Module):

    def __init__(self, d_model = 16, n_heads = 4, dropout = 0.4):
        super(transformer_t, self).__init__()

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, dropout = 0.4),
                                             num_layers = 1,
                                             norm = nn.LayerNorm(normalized_shape = d_model, eps = 1e-6))
        self.positional_encoding = PositionalEmbedding(d_model)
        self.predictor = nn.Linear(d_model, 8)
        self.dropout = nn.Dropout(dropout)
        self.Embedding = Embedding()


    def forward(self, src):
        src = src.reshape([8,1])
        x1 = self.Embedding(src.reshape([8,1]))
        out = self.predictor(self.encoder(x1 + self.positional_encoding(src)))
        return self.dropout(out)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(8)
        attn = nn.Softmax(dim=-1)(scores)
        prob = torch.matmul(attn, V)
        return prob

class MultiHeadAttention(nn.Module):
    def __init__(self, dk):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = 2
        self.W_Q = nn.Linear(dk, 8 * 2, bias=False)
        self.W_K = nn.Linear(4096, 8 * 2, bias=False)
        self.W_V = nn.Linear(4096, 8 * 2, bias=False)
        self.fc = nn.Linear(8 * 2, 4096, bias=False)  # ff 全连接
        self.layer_norm = nn.LayerNorm(16)  # normal 归一化
        self.ScaledDotProductAttention = ScaledDotProductAttention()

    def forward(self, input_Q, input_K, input_V):
        # input_Q：1*4*6，每批1句 * 每句4个词 * 每词6长度编码
        # residual 先临时保存下：原始值，后面做残差连接加法
        residual, batch = input_Q, input_Q.size(0)

        # 乘上 W 矩阵。注：W 就是要训练的参数
        # 注意：维度从2维变成3维，增加 head 维度，也是一次性并行计算
        Q = self.W_Q(input_Q)  # 乘以 W(6*6) 变为 1*4*6
        Q = Q.view(batch, -1, 2, 8).transpose(1, 2)  # 切开为2个Head 变为 1*2*4*3 1批 2个Head 4词 3编码
        K = self.W_K(input_K).view(batch, -1, 2, 8).transpose(1, 2)
        V = self.W_V(input_V).view(batch, -1, 2, 8).transpose(1, 2)

        # 返回1*2*4*3，2个头，4*3为带上关注关系的4词
        prob = self.ScaledDotProductAttention(Q, K, V)

        # 把2头重新拼接起来，变为 1*4*6
        prob = prob.transpose(1, 2).contiguous()
        prob = prob.view(batch, -1, 2 * 8).contiguous()

        # 全连接层：对多头注意力的输出进行线性变换，从而更好地提取信息
        output = self.fc(prob)

        # 残差连接 & 归一化
        # res = self.layer_norm(residual + output) # return 1*4*6
        return output


class Model(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Model, self).__init__()

        self.llama_config = LlamaConfig.from_pretrained(r'Llama-2-7b-hf/')
        self.llama_config.num_hidden_layers = 1
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.llama = LlamaModel.from_pretrained(r'Llama-2-7b-hf/', use_safetensors=True)

#         self.llama = torch.quantization.quantize_dynamic(LlamaModel.from_pretrained("/home/yzhang82/Llama-2-7b-hf",
#                                                                 use_safetensors=True), {torch.nn.Linear}, dtype=torch.qint8)
        self.tokenizer = LlamaTokenizer.from_pretrained(r'Llama-2-7b-hf/')
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llama.parameters():
            param.requires_grad = False
        

        self.dropout = nn.Dropout(0.2)

        self.MultiHeadAttention0 = MultiHeadAttention(1248)
        self.MultiHeadAttention1 = MultiHeadAttention(64)
        self.model_graphlearning = GCN(nfeat = 16, nhid = 32, nclass = 16, dropout = 0.4)
        self.tt_trans = transformer_t(dropout = 0.4)
        self.Softmaxlayer = torch.nn.Softmax(dim = 1)
        self.output_projection0 = nn.Linear(4096, 16, bias = True)
#         self.output_projection1 = nn.Linear(80, 1, bias = True)
        self.output_projection1 = nn.Linear(16*50, 1, bias = True)

    def text_T(self, T, t, R):
        label = 'From ' + text1[T] + ', to ' + text1[T+t-1]\
                +', there were ' + str(list(map(int, taxi_tensor_in[T:T+t,R])))[1:-1]\
                + ' taxis visiting Community region ' + 'r'+str(R)+'.'\
                + ' The time interval is 1 hour.'\
                + ' The temporal sequence has a minimum of '\
                + str(int(taxi_tensor_in[T:T+t,R].min()))\
                + ' at ' + text1[T+list(taxi_tensor_in[T:T+t,R]).index(taxi_tensor_in[T:T+t,R].min())] \
                +'; a second minimum of ' + str(int(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[2]))\
                + ' at '+ text1[T+(list(taxi_tensor_in[T:T+t,R]).index(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[2]))]\
                +'; a maximum of ' + str(int(taxi_tensor_in[T:T+t,R].max()))\
                + ' at ' + text1[T+list(taxi_tensor_in[T:T+t,R]).index(taxi_tensor_in[T:T+t,R].max())] \
                +'; a second maximum of ' + str(int(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[-2])) \
                +' at '+ text1[T+(list(taxi_tensor_in[T:T+t,R]).index(sorted(taxi_tensor_in[T:T+t,R], reverse=False)[-2]))]\
                + '; and a mean of '+str(taxi_tensor_in[T:T+t,R].mean())+'.'
        return label

    def forward(self, x_input, x_s_text, adj, T, t, question):
        token_whole = torch.empty((0)).cuda()
        for i in range(t):
            prompt = self.tokenizer(x_s_text[T+i], return_tensors = "pt", padding = True, truncation = True, 
                                    max_length = 2500).input_ids.cuda()
            token_whole = torch.concat([token_whole, 
        self.MultiHeadAttention0(self.model_graphlearning(x_input[T+i], adj).reshape([1, 1,-1]), 
        self.llama.get_input_embeddings()(prompt)[0],
        self.llama.get_input_embeddings()(prompt)[0])], dim=1)
            
        # token_t = torch.empty((x_input.shape[1]))
        for j in range(78):
            prompt = self.tokenizer(self.text_T(T,t,j), return_tensors = "pt", padding = True, truncation = True, 
                                    max_length = 2500).input_ids.cuda()
            token_whole = torch.concat([token_whole, 
                                        self.MultiHeadAttention1(self.tt_trans(x_input[T:T+t, j]).reshape([1, 1, -1]), 
                                                                 self.llama.get_input_embeddings()(prompt)[0],
                                                                 self.llama.get_input_embeddings()(prompt)[0])],
                                       dim=1)
        select_question = self.tokenizer(question, return_tensors = "pt", padding = True, truncation = True, 
                                         max_length = 100).input_ids.cuda()
        token_whole = torch.concat([token_whole, self.llama.get_input_embeddings()(select_question)], dim = 1)
        dec_out = self.llama(inputs_embeds = token_whole).last_hidden_state[:, -50: , :]
        dec_out =  self.output_projection1((self.output_projection0(dec_out[0]).T).reshape([1,16*50]))
        
        return dec_out[0]

def questions_(t0, t1, t2):
    questions = []
    questions.append('Given the historical traffic flows for 78 Community regions from ' + text1[t0] + ' to ' +  text1[t1] +'.' \
                     + ' Your task is to predict the total traffic flows at ' + text1[t2] + '.')
    return questions

def answers_(t2):
    answers = sum_list[t2]
    return answers



model = Model(nfeat = 16, nhid = 32, nclass = 16, dropout = 0.4)
taxi_tensor_in = np.load('taxi_tensor_in.npy')
sum_list = np.sum(taxi_tensor_in, axis=1)
model = model.cuda()
taxi_tensor_in = torch.FloatTensor(taxi_tensor_in).cuda()
adj0 = adj0.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001, weight_decay = 5e-4)
criterion0 = torch.nn.L1Loss()
criterion1 = torch.nn.MSELoss()
def train(epoch):
    model.train()
    optimizer.zero_grad()
    lossall = []
    loss_train = 0
    for i in random.sample(range(0, N), 8):
        question = questions_(i, i+7, i+8)
        answer = torch.tensor(answers_(i+8)).reshape([1])
        output = model(taxi_tensor_in, text_s, adj0, i, 8, question[0]).cpu()
        loss = criterion0(output, answer)
        loss_train = loss_train + loss
        lossall.append(float(loss.cpu()))
    loss_train.backward()
    optimizer.step()

    return  mean(lossall)

torch.backends.cudnn.enabled = False
loss0 = []
for epoch in tqdm(range(3000)):
    loss0.append(float(train(epoch)))


def test0(k):
    model.eval()
    criterion0 = torch.nn.L1Loss()
    criterion1 = torch.nn.MSELoss()
    loss_test0 = 0
    loss_test1 = 0
    with torch.no_grad():
        for i in tqdm(range(N,M)):
            question = questions_(i, i+7, i+8+k)
            answer = torch.tensor(answers_(i+8+k)).reshape([1])
            output = model(taxi_tensor_in, text_s, adj0, i, 8, question).cpu()
            
            loss0 = criterion0(output, answer)
            loss_test0 = loss_test0 + loss0
            loss1 = criterion1(output, answer)
            loss_test1 = loss_test1 + loss1
    return loss_test0/len(range(N,M)), loss_test1/len(range(N,M))






