
# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np 
from models.Variable_Token import CrossAtt
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel,LlamaForCausalLM
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
class Variable_Embedder(nn.Module):
    def __init__(self,patch_len,seq_len,d_model):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len # 
        self.patch_num = int(seq_len//patch_len)
        self.position_embedding = nn.Parameter(torch.randn(self.patch_num, self.patch_len))
        self.MHA = nn.MultiheadAttention( embed_dim = patch_len, num_heads = 4,kdim=patch_len,vdim= patch_len)
        self.fc = nn.Linear(patch_len,d_model)
    def forward(self,x):
        '''
        x: B M L
        out: B M D
        '''
        b,m,l = x.shape
        x = x.reshape(b*m,self.patch_num,self.patch_len)#bm q l
        pe = self.position_embedding.unsqueeze(0).expand(b*m, -1, -1)
        x_with_pe = x+pe
        output,_ = self.MHA(x_with_pe,x_with_pe,x_with_pe)# q,k,v  # b*m p_n p_l
        output = self.fc(output[:,-1,:]) #b*m d_model
        output = output.reshape(b,m,-1) # b m d_model
        return output
    
class Sequential_Detokenizer(nn.Module):
    def __init__(self, d_model, seq_len,pred_len,dropout):
        super().__init__()
        self.MHA = nn.MultiheadAttention( embed_dim = d_model+seq_len, num_heads = 4, kdim = seq_len, vdim = seq_len)
        self.forecasting_head = nn.Linear(seq_len+seq_len+d_model, pred_len)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    def forward(self, end_token, exo_var):
        '''
        end_token: [B, 1, d_model]
        exo_var: [B, M-1, L] 
        '''
        end_token = end_token.permute(1,0,2) # 1 B d
        exo_var = exo_var.permute(1,0,2) # M-1 B d

        end_series,_ = self.MHA(end_token,exo_var,exo_var) # 1 B d
        end_series = self.dropout(end_series)
        pred = self.forecasting_head(torch.cat([end_token[:,:,:-self.d_model],end_series],dim=-1))

        return pred.permute(1,0,2) # B 1 pred_len
class Textual_Tokenizer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.MHA = nn.MultiheadAttention( embed_dim = d_model, num_heads = 4)

    def forward(self, embeddings, prompts):
        '''
        embeddings [ B, x, d] 
        prompts [B, x, d]
        '''
        embeddings = embeddings.permute(1,0,2)
        prompts = prompts.permute(1,0,2)
        tokens, _ = self.MHA(embeddings,prompts,prompts)
        return tokens.permute(1,0,2)
class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        self.data_name = configs.data
        prompts = torch.load(configs.root_path+self.data_name+'.pt') 
        self.exd_exo_prompts = prompts[:-1] # 7 768
        self.dataset_prompt = prompts[-1:] # 1 768

        # LLM with finetuning for Forecasting
        self.LLM = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.LLM.h = self.LLM.h[:configs.e_layers]
        print("gpt2 = {}".format(self.LLM))
   
        
     
        self.Variable_Embedder = Variable_Embedder(configs.patch_len,configs.seq_len,configs.d_model) # 
    
        self.end_tokenizer = Textual_Tokenizer(configs.d_model)
        self.exo_tokenizer = Textual_Tokenizer(configs.d_model)
      
        self.Endogenous_Detokenizer = Sequential_Detokenizer(configs.d_model, configs.seq_len, configs.pred_len,configs.dropout) #nn.Linear(configs.d_model, configs.pred_len)
        #self.forecasting_head = nn.Linear(configs.seq_len+configs.pred_len,configs.pred_len)
        for i, (name, param) in enumerate(self.LLM.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_batch_prompts(self, x):
        bs, seq_len, n_vars = x.shape[0], x.shape[1], x.shape[2] # nvars

        expanded_tensor = self.exd_exo_prompts.unsqueeze(0) # [n,768]->[1,n,768]
        batch_variable_knowledge = expanded_tensor.repeat(bs, 1, 1).to(x.device)           #[bs, n_vars, 768]
        
        expanded_tensor = self.dataset_prompt.unsqueeze(0) # [n,768]->[1,n,768] # bs 1 768
        batch_prompt = expanded_tensor.repeat(bs, 1, 1).to(x.device)  # bs 1 768

        batch_exo_prompt = torch.cat([batch_prompt,batch_variable_knowledge[:,:-1,:]],dim=1)
        batch_end_prompt = torch.cat([batch_prompt,batch_variable_knowledge[:,-1:,:]],dim=1)
        return batch_exo_prompt, batch_end_prompt, batch_prompt
    def forward(self, x, x_mark):           # x: [Batch, Input length, Channel]
        '''
        x: [Batch, Input length, Channel]
        x_mark: [Batch, Input length, num of time_stamp types
        '''
        B, L, M = x.shape
        batch_exo_prompt, batch_end_prompt,prompt = self.get_batch_prompts(x)
   
        # 1. REVIN
        means = x.mean(1, keepdim=True).detach() # B 1 M
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()  # B 1 M
        x /= stdev # x: [Batch, Input length, Channel]
        ## print(x.shape,stdev.shape) torch.Size([256, 336, 7]) torch.Size([256, 1, 7])
        # 2. LAST
        # last = x[:,-1:,-1] # B 1
        x = rearrange(x, 'b l m -> b m l') # B M L
    
        # Variable_Embedder
        variable_embeddings = self.Variable_Embedder(x) # B M d_model(768)
        #print(variable_tokens.shape) #torch.Size([256, 7, 768])


      
        exo_tokens = self.exo_tokenizer(variable_embeddings[:,:-1,:],batch_exo_prompt)# B M-1 768 -> B M-1 768
        exd_tokens = self.end_tokenizer(variable_embeddings[:,-1:,:],batch_end_prompt)# B M-1 768 -> B M-1 768 
        #print(variable_llm.shape)# torch.Size([256, 7, 768])
     
        llm_input = torch.cat([prompt,exo_tokens,exd_tokens],dim=1)            # B M+1 768
        #print(llm_input.shape) #torch.Size([256, 14, 768])

     
        outputs = self.LLM(inputs_embeds=llm_input).last_hidden_state #
        #print(outputs.shape)
        outputs = self.Endogenous_Detokenizer(end_token =  torch.cat([x[:,-1:,:],outputs[:,-1:,:]],dim=-1), exo_var = x[:,:-1,:])[:,-1,:] # B T   outputs.reshape(B,-1)
        #outputs = torch.cat([x[:,-1,:],outputs],dim=-1)
        #outputs = self.forecasting_head(outputs)
        #print(outputs.shape)

        # 1 REVIN
        outputs = outputs * stdev[:,:,-1] # B T
        outputs = outputs + means[:,:,-1] # B T
        # 2 LAST
        # outputs = outputs + last
        return outputs.unsqueeze(-1)