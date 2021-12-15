from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import torch
from bert.optimization import BERTAdam

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

def prepare_optimizer(args, model, num_train_steps):
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    return optimizer, param_optimizer
def prepare_optimizer_2(args, model, num_train_steps):
    # pretrain_model_params = list(map(id, model.pretrain_model.parameters()))
    resnet152_params = list(map(id, model.resnet152.parameters()))
    bert_params = list(map(id, model.bert.parameters()))
    # id(p) not in resnet152_params and id(p) not in pretrain_model_params and
    base_params = filter(lambda p:  
                    id(p) not in bert_params and id(p) not in resnet152_params,
                    # and p.requires_grad,
                    model.parameters())
    # no_decay = ['bias', 'LayerNorm']
    paras = [
                # {'params':model.pretrain_model.parameters(),'lr':args.learning_rate},
                {'params':model.resnet152.parameters(),'lr':args.learning_rate_pretrained},
                {'params':model.bert.parameters(),'lr':args.learning_rate_pretrained},
                # {'params':model.imgLa2text.parameters(),'lr':args.learning_rate_pretrained},
                {'params':base_params,'lr':args.learning_rate},
            ]
    # paras=[
    #         {'params':}
    #         ]
    # torch.optim.Adam(paras, weight_decay=args.)
    # torch.optim.Adam(paras, weight_decay=args.wdecay)
    # scheduler_rel = torch.optim.lr_scheduler.LambdaLR(optimizer)
    optimizer = BERTAdam(paras,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    return optimizer,paras
def prepare_optimizer_3(args, model, num_train_steps):
    
    fix_params = [['resnet152.'+n,p]for n,p in model.resnet152.named_parameters()]+[['bert.'+n,p] for n,p in model.bert.named_parameters()]
    # base_params_names= [n for n,p in base_params]
    fix_params_names= [n for n,p in fix_params]
    base_params = [item for item in list(model.named_parameters()) if item[0] not in fix_params_names]
    # resnet152.resnet.layer4.0.downsample.0.weight
    # for n,p in base_params:
    #     # if 'resnet' in n:
    #     #     print(n)
    #     for f,c in fix_params:
    #         if n == f:
    #             print(n)
    

    no_decay = ['bias', 'LayerNorm']
    paras = [
                {'params':[p for n, p in fix_params if not any(nd in n for nd in no_decay)],'lr':args.learning_rate_pretrained, 'weight_decay': 0.01},
                {'params':[p for n, p in fix_params if  any(nd in n for nd in no_decay)],'lr':args.learning_rate_pretrained, 'weight_decay': 0.0},
                {'params':[p for n, p in base_params if not any(nd in n for nd in no_decay)],'lr':args.learning_rate, 'weight_decay': 0.01},
                {'params':[p for n, p in base_params if  any(nd in n for nd in no_decay)],'lr':args.learning_rate, 'weight_decay': 0.0}
                
            ]
    # paras = [
    #             {'params':[p for n, p in fix_params ],'lr':args.learning_rate_pretrained},
    #             {'params':[p for n, p in base_params ],'lr':args.learning_rate}
                
    #         ]

    optimizer = BERTAdam(paras,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    return optimizer,paras
def post_process_loss(args, n_gpu, loss):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if args.fp16 and args.loss_scale != 1.0:
        # rescale loss for fp16 training
        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
        loss = loss * args.loss_scale
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    return loss

def bert_load_state_dict(model, state_dict):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict 
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    return model