
# Cell

import torch
from torch import nn
import torch.nn.functional as F

class Extract_Variable_Dependecnce(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.MHA = nn.MultiheadAttention( embed_dim = d_model, num_heads = 4,kdim=d_model,vdim= d_model)
    def forward(self,x_external,x_target):
        '''
        x_external: B M-1 d_model
        x_target: B 1 d_model
        out: B 1 d_model
        '''
        output,_ = self.MHA(x_target,x_external,x_external)
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_size, d_model, num_layers=2):
        super(LSTMModel, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, d_model, num_layers, batch_first=True)

    def forward(self, x):
        '''
        x: B M patch_num patch_len
        out: B M d_model
        '''
        # 因为LSTM的输入只能是三维的，所以先把x做个reshape，变成三维，输出时候再reshape回来
        B,M,L,C =x.shape
        x = x.reshape(B*M,L,C) # B M patch_num patch_len -> B*M patch_num patch_len
        h0 = torch.zeros(self.num_layers, x.size(0), self.d_model).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.d_model).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out 是lstm最后一层所有时间步的hidden，取最后一个时间步做时序表征
        out = out[:,-1,:] # B*M patch_num d_model -> B*M d_model
        # reshape，把变量维度拿出来
        out = out.reshape(B,M,self.d_model)
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.patch_num = int(self.seq_len//self.pred_len)
        self.d_model = configs.d_model
        # lstm时序特征提取
        self.lstm = LSTMModel(input_size=self.patch_len,d_model=self.d_model)
        # 目标变量对外部变量的关系提取
        self.extract_variable_dependence = Extract_Variable_Dependecnce(d_model=self.d_model)
    def forward(self, x):           
        '''
        x: [Batch, Sequence Length, Multi-Variables]
        out: [Batch, Prediction Length, Single-Variable]
        '''
        
        B, L, M = x.shape
        # 1. 对x进行patch，[B,L,M]->[B,M,patch_num,patch_len]，多了一个通道维度，方便LSTM学习 
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = x.reshape(B,M,self.patch_num,self.patch_len) # 通过patch增加通道维度： [B, M ,Sequence Length] -> [B, M, patch_num, patch_len]
        
        # 2. 使用共享的 LSTM 对 内部变量 和 外部变量 做时序特征提取
        lstm_out = self.lstm(x) # [B,M,d_model]
        x_external = lstm_out[:,:-1,:] # 前M-1个变量是外部变量 [B,M-1,d_model]
        x_target = lstm_out[:,-1:,:] # 第m个变量是目标变量 [B,1,d_model]
        
        # 3. 使用交叉注意力，提取目标变量对外部变量的依赖
        out = self.extract_variable_dependence(x_external,x_target)
        return out 