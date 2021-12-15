import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.util import *
# from resnet_model import resnet34
import math
import resnet.resnet as resnet
from resnet.resnet_utils import myResnet
from bert.modeling import BertModel, BERTLayerNorm
from utils.multihead import MultiHeadAttention
import os
from pdb import set_trace as stop
from tqdm import tqdm
from torch.autograd import Variable
from bert.multran import MULTModel

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))




class BertRel(torch.nn.Module):
    def __init__(self, params):
        super(BertRel, self).__init__()
        self.params = params
        # self.mybert = ResNetVLBERT_Trans(params)
        self.mybert = ResNetVLBERT(params)
        self.pair_preject = torch.nn.Linear(in_features=768, out_features=4)

    def forward(self, sentence_ids, img_obj, mask,mode=None):
        # Get the text features
        # u = self.text_encoder(sentence, sentence_lens, chars)
        # batch_size = sentence.shape[0]
        # if mode == "test":
        #     batch_size = 1
        hidden_states = self.mybert(sentence_ids, img_obj, mask)
        
        # print(all_encoder_layers.shape)
        pair_out = self.pair_preject(hidden_states)
        pair_out = pair_out.reshape(-1,2,2)
        # pair_out = F.relu(pair_out)
        # print(pair_out.shape)
        if mode == 'fix':
            pair_out = Variable(pair_out.data)
        return pair_out

class ResNetVLBERT(nn.Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__()
        self.config = config
       
        # image infomation
        net = getattr(resnet, 'resnet152')()
        # net = resnet.resnet152(pretrained=False)
        net.load_state_dict(torch.load(os.path.join('./resnet', 'resnet152.pth')))
        self.pre_resnet = myResnet(net,True) #True
        # self.pre_resnet = resnet152()
        # self.pre_resnet.load_state_dict(torch.load('/home/data/datasets/resnet152-b121ed2d.pth'))
        print('load resnet152 pretrained rpbert')

        self.aling_img_1 = nn.Linear(2048,config.hidden_size)
        # self.lstm_fusion = nn.LSTMl(config.hidden_size,config.hidden_size/2,bidirectional=True)
        self.img2txt = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        self.txt2img = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        self.txt2txt = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        self.img2img = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        self.aaali = nn.Linear(config.hidden_size * 4,config.hidden_size)
        # init weights
        

        def init_weight(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            # if isinstance(module, nn.Linear):
            #     module.bias.data.zero_()
        # self.apply(init_weight)
        self.bert = BertModel(config)
        # self.pre_resnet = myresnet


    def fix_params(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        for name,param in self.pre_resnet.named_parameters():
            param.requires_grad = False
    def forward(self,sentence_ids, img_obj, attention_mask):
        # self.fix_params()
        # stop()
        batch_size = sentence_ids.shape[0]
        images = img_obj

        _,_,img_feature_raw = self.pre_resnet(images)
        img_feature = img_feature_raw.view(batch_size, 2048, 7 * 7).transpose(2, 1)
        img_info = self.aling_img_1(img_feature) #B *V * D
        all_encoder_layers, _ = self.bert(sentence_ids, None, attention_mask)
        sequence_output_raw = all_encoder_layers[-1] #[B,L,D]

        # fusion mean()
        img2txt = self.img2txt(img_info,sequence_output_raw,sequence_output_raw)[0].mean(1)
        txt2img = self.txt2img(sequence_output_raw,img_info,img_info)[0].mean(1)
        txt2txt = self.txt2txt(sequence_output_raw,sequence_output_raw,sequence_output_raw)[0].mean(1)
        img2img = self.img2img(img_info,img_info,img_info)[0].mean(1)
        hidden_states = self.aaali(torch.tanh(torch.cat([img2txt,txt2img,txt2txt,img2img],dim=-1)))
        return hidden_states

class ResNetVLBERT_Trans(nn.Module):
    def __init__(self, config):

        super(ResNetVLBERT_Trans, self).__init__()
        self.config = config
       
        # image infomation
        net = getattr(resnet, 'resnet152')()
        # net = resnet.resnet152(pretrained=False)
        net.load_state_dict(torch.load(os.path.join('./resnet', 'resnet152.pth')))
        self.pre_resnet = myResnet(net,True) #True
        # self.pre_resnet = resnet152()
        # self.pre_resnet.load_state_dict(torch.load('/home/data/datasets/resnet152-b121ed2d.pth'))
        print('load resnet152 pretrained rpbert')
        self.mult = MULTModel(config.hidden_size,config.hidden_size)
        self.aling_img_1 = nn.Linear(2048,config.hidden_size)
        # self.lstm_fusion = nn.LSTMl(config.hidden_size,config.hidden_size/2,bidirectional=True)
        # self.img2txt = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        # self.txt2img = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        # self.txt2txt = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        # self.img2img = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        self.aaali = nn.Linear(config.hidden_size * 4,config.hidden_size)
        # init weights
        

        def init_weight(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            # if isinstance(module, nn.Linear):
            #     module.bias.data.zero_()
        # self.apply(init_weight)
        self.bert = BertModel(config)
        # self.pre_resnet = myresnet


    def fix_params(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        for name,param in self.pre_resnet.named_parameters():
            param.requires_grad = False
    def forward(self,sentence_ids, img_obj, attention_mask):
        # self.fix_params()
        # stop()
        batch_size = sentence_ids.shape[0]
        images = img_obj

        _,_,img_feature_raw = self.pre_resnet(images)
        img_feature = img_feature_raw.view(batch_size, 2048, 7 * 7).transpose(2, 1)
        img_info = self.aling_img_1(img_feature) #B *V * D
        all_encoder_layers, _ = self.bert(sentence_ids, None, attention_mask)
        sequence_output_raw = all_encoder_layers[-1] #[B,L,D]
        # mult
        seq_img,_ = self.mult(sequence_output_raw,img_info)
        # fusion mean()
        # img2txt = self.img2txt(img_info,sequence_output_raw,sequence_output_raw)[0].mean(1)
        # txt2img = self.txt2img(sequence_output_raw,img_info,img_info)[0].mean(1)
        # txt2txt = self.txt2txt(sequence_output_raw,sequence_output_raw,sequence_output_raw)[0].mean(1)
        # img2img = self.img2img(img_info,img_info,img_info)[0].mean(1)
        # hidden_states = self.aaali(F.tanh(torch.cat([img2txt,txt2img,txt2txt,img2img],dim=-1)))
        return seq_img
