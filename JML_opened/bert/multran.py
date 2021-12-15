import torch
from torch import nn
import torch.nn.functional as F

from bert.transformer import TransformerEncoder
from pdb import set_trace as stop
class MULTModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_c,self.orig_d_l, self.orig_d_v, self.orig_d_a=input_dim,input_dim,input_dim,input_dim
        self.d_c,self.d_l, self.d_v, self.d_a = input_dim, input_dim, input_dim,input_dim
        # output_dim = 512 
        self.vonly = True
        self.aonly = True
        self.lonly = False
        self.conly = False
        self.num_heads = 4
        self.layers = 6
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.0
        self.attn_dropout_v =0.0
        self.attn_dropout_c =0.0
        self.relu_dropout = 0.1 
        self.res_dropout =0.1
        self.out_dropout =0.0
        self.embed_dropout = 0.25
        self.attn_mask = False

        # combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly+self.conly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2*self.d_l*self.partial_mode
            # combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
              # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        # self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        #use linear instead
        # self.align_l=nn.Linear(self.orig_d_l,self.d_l)
        self.align_a=nn.Linear(self.orig_d_a,self.d_a)
        self.align_v=nn.Linear(self.orig_d_v,self.d_v)
        # self.align_c=nn.Linear(self.orig_d_c,self.d_c)

        # 2. Crossmodal Attentions
        if self.conly:
            self.trans_c_with_l = self.get_network(self_type='cl')
            self.trans_c_with_a = self.get_network(self_type='ca')
            self.trans_c_with_v = self.get_network(self_type='cv')
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            self.trans_l_with_c = self.get_network(self_type='lc')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            self.trans_a_with_c = self.get_network(self_type='ac')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            self.trans_v_with_c = self.get_network(self_type='vc')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        # self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        # self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        # self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        # self.trans_c_mem = self.get_network(self_type='c_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        #model
        # self.trans_l1=nn.Linear(3*self.d_l,3*self.d_l)
        # self.trans_l2=nn.Linear(3*self.d_l,3*self.d_l)
        self.trans_a1=nn.Linear(self.d_a,self.d_a)
        self.trans_a2=nn.Linear(self.d_a,self.d_a)
        self.trans_v1=nn.Linear(self.d_v,self.d_v)
        self.trans_v2=nn.Linear(self.d_v,self.d_v)
        # self.trans_c1=nn.Linear(3*self.d_c,3*self.d_c)
        # self.trans_c2=nn.Linear(3*self.d_c,3*self.d_c)
        self.last_out1=nn.Linear(self.d_a,self.d_a)
        self.last_out2=nn.Linear(2*self.d_a,output_dim)

        # self.out_layer_l = nn.Linear(3*self.d_l, output_dim)
        self.out_layer_a = nn.Linear(self.d_a, output_dim)
        self.out_layer_v = nn.Linear(self.d_v, output_dim)
        # self.out_layer_c = nn.Linear(3*self.d_c, output_dim)
        self.out_cat = nn.Linear(self.partial_mode*output_dim,output_dim)



    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl','cl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va','ca']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av','cv']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['c', 'lc', 'ac','vc']:
            embed_dim, attn_dropout = self.d_c, self.attn_dropout_c
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 3*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 3*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 3*self.d_v, self.attn_dropout
        elif self_type == 'c_mem':
            embed_dim, attn_dropout = 3*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_v, x_a):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #use linear instead
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.align_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.align_v(x_v)
        
        
        # proj_x_a = proj_x_a.transpose(1, 2)
        # proj_x_v = proj_x_v.transpose(1, 2)
        


        proj_x_a = proj_x_a.permute(1, 0, 2)
        proj_x_v = proj_x_v.permute(1, 0, 2)#[seq_len,batch_size,convId_dim]
        
        if self.aonly:
            # (L,V) --> A
            # h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            # stop()
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            # h_a_with_cs = self.trans_a_with_c(proj_x_a, proj_x_c, proj_x_c)
            # h_as = torch.cat([h_a_with_ls, h_a_with_vs,h_a_with_cs], dim=2)
            h_as = h_a_with_vs
            # h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            # h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            # h_v_with_cs = self.trans_v_with_c(proj_x_v, proj_x_c, proj_x_c)
            # h_vs = torch.cat([h_v_with_ls, h_v_with_as,h_v_with_cs], dim=2)
            h_vs = h_v_with_as
            # h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        if self.partial_mode == 2:
            last_hs = torch.cat([last_h_v,last_h_a], dim=1)
        # A residual block #b*512
        # last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        # resdual #
        # last_ha_proj = self.trans_a2(F.dropout(F.relu(self.trans_a1(h_as)), p=self.out_dropout))
        # last_hv_proj = self.trans_v2(F.dropout(F.relu(self.trans_v1(h_vs)), p=self.out_dropout))
        # last_hs_proj += last_hs
        # last_hl_proj += last_h_l
        # last_ha_proj += last_h_a
        # last_hv_proj += last_h_v
        
        #out
        # output = self.out_layer(last_hs_proj)
        # output_l = self.out_layer_l(last_hl_proj).transpose(0,1).contiguous()
        # output_a = self.out_layer_a(last_ha_proj).transpose(0,1).contiguous()
        # output_v = self.out_layer_v(last_hv_proj).transpose(0,1).contiguous()
        # output_c = self.out_layer_c(last_hc_proj).transpose(0,1).contiguous()
        #b*3*D
        # last
        last_ha_proj = self.last_out1(F.dropout(F.relu(self.last_out2(last_hs)), p=self.out_dropout))
        # output_cat=torch.cat([output_a,output_v],1)
        # output = self.out_cat(output_cat)
        return last_ha_proj, last_hs