# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SemEval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.sentiment_modeling import JointMMwithRel

from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, RawFinalResult, RawSpanResult, span_annotate_candidates
from absa.run_base import copy_optimizer_params_to_model, set_optimizer_params_grad, prepare_optimizer,prepare_optimizer_2,prepare_optimizer_3, post_process_loss, bert_load_state_dict
from absa.eval_metric import eval_absa

from utils.data_loader_bb import DataLoader as DLbb
from torch import nn as nn
from tqdm import tqdm
from pdb import set_trace as stop
from torch.nn import functional as F
from sklearn.metrics import f1_score

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def read_train_data(args, tokenizer, logger):
    train_path = os.path.join(args.data_dir, args.train_file)
    train_set = read_absa_data(train_path)
    train_examples = convert_absa_data(dataset=train_set, args=args,verbose_logging=args.verbose_logging)# transform  the data into the example class
    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)
       
    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    logger.info("Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_positions for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_positions for f in train_features], dtype=torch.long)
    all_bio_labels = torch.tensor([f.bio_labels for f in train_features],dtype=torch.long)
    all_polarity_positions = torch.tensor([f.polarity_positions for f in train_features],dtype=torch.long)
    all_image_labels = torch.tensor([f.image_labels for f in train_features],dtype=torch.long)
    all_image_raw_data = torch.stack([f.raw_image_data for f in train_features])#,dtype=torch.float
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)


    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index,all_start_positions, all_end_positions, all_bio_labels,all_polarity_positions, all_image_labels,all_image_raw_data)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_examples, train_features, train_dataloader, num_train_steps

def read_eval_data(args, tokenizer, logger):
    eval_path = os.path.join(args.data_dir, args.predict_file)
    eval_set = read_absa_data(eval_path)
    eval_examples = convert_absa_data(dataset=eval_set, args=args,verbose_logging=args.verbose_logging)

    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                 args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_start_positions = torch.tensor([f.start_positions for f in eval_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_positions for f in eval_features], dtype=torch.long)
    all_bio_labels = torch.tensor([f.bio_labels for f in eval_features],dtype=torch.long)
    all_polarity_positions = torch.tensor([f.polarity_positions for f in eval_features],dtype=torch.long)
    all_image_labels = torch.tensor([f.image_labels for f in eval_features], dtype=torch.long)
    all_image_raw_data = torch.stack([f.raw_image_data for f in eval_features])#,dtype=torch.float

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index,all_start_positions, all_end_positions, all_bio_labels,all_polarity_positions, all_image_labels,all_image_raw_data)
  
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader

def eval_relation_f1(relation_pred,relation_true):
    relation_pred= relation_pred.squeeze(1).numpy().tolist()
    relation_true= relation_true.squeeze(1).numpy().tolist()
    # print(relation_pred)
    recall_num = sum(relation_pred)
    recision_num = sum(relation_true)
    common = 0
    for p ,t in zip(relation_pred,relation_true):
        if p == t and p ==1:
            common +=1
    recall = common / recall_num if recall_num >0 else 0
    recision = common / recision_num if recision_num > 0 else 0
    f1 = 2*recall*recision/(recall+recision) if (recall+recision) >0 else 0
    return f1

def eval_relations(pair_out_pred,ifpairs_true):
    pair_out_pred = torch.cat(pair_out_pred).cpu()
    pred_RR,pred_VV = pair_out_pred.split(1,dim=1)
    ifpairs_true = torch.cat(ifpairs_true).cpu()
    true_RR,true_VV = ifpairs_true.split(1,dim=1)
    
    f1_RR = eval_relation_f1(pred_RR,true_RR)
    f1_VV = eval_relation_f1(pred_VV,true_VV)

    assert len(pred_RR) == len(true_RR)
    all_len = len(pair_out_pred)
    RR, VV =0,0
    pair_out_pred = pair_out_pred.numpy().tolist()
    ifpairs_true =  ifpairs_true.numpy().tolist()
    for p,t in zip(pair_out_pred,ifpairs_true):
        if p[0] == t[0]:
            RR+=1
        if p[1] == t[1]:
            VV+=1
       

    return (round(RR/all_len,4),round(VV/all_len,4),round(f1_RR,4),round(f1_VV,4))

def eval_relations_sklearn(pair_out_pred_sklearn,ifpairs_true_sklearn):
    
    pair_out_pred_sklearn = torch.cat(pair_out_pred_sklearn).cpu().squeeze()
    ifpairs_true = torch.cat(ifpairs_true_sklearn).cpu().squeeze()
    ifpairs_sklean =  ifpairs_true.numpy().tolist()
    pair_out_pred_sklearn = pair_out_pred_sklearn.numpy().tolist()
    f1 = f1_score(ifpairs_sklean,pair_out_pred_sklearn, average='micro')
    print('sklearn f1 score :{}'.format(f1))


def detach_submodel(submodel):
    for n, p in submodel.named_parameters():
        p.requires_grad = False
    
def train_relation_model(args, optimizer_rel,scheduler_rel, 
        tokenizer,global_step, model, device):
    dlbb = DLbb(args,tokenizer)
    if os.path.exists(args.pre_train_path) and not os.path.isdir(args.pre_train_path):
        checkpoint = torch.load(args.pre_train_path,device)
        model.pretrain_model.load_state_dict(checkpoint['model_rel'])
        # model.pre_load_state_dict(checkpoint['model_rel'])
        print("read the model")
    elif 'test' in args.pre_train_path:
        print('-'*10,'test train pretrain','-'*10)
        fp_pre = open(os.path.join(args.output_dir,'pretrain.txt'),"a")
        train2_phase1_loader = dlbb.train2_phase1_loader 
        test2_phase1_loader = dlbb.test2_phase1_loader 
        f1_RR_best, f1_VV_best =0,0
        for i in range(args.relation_epoches):
            loss_function_relation = nn.CrossEntropyLoss()
            ifpairs_true = []
            pair_out_pred = []
            model.pretrain_model.train()
            for (x, x_obj, y, mask, lens, ifpairs) in train2_phase1_loader:
                # model.pretrain_model.train()
                optimizer_rel.zero_grad()
                if device:
                    x = x.to(device)
                    x_obj = x_obj.to(device)
                    mask = mask.to(device)
                    ifpairs = ifpairs.float().to(device)
                pair_out = model.pretrain_model(x, x_obj, mask)  # seq_len * bs * labels

                B,N,S = pair_out.shape
                pair_out_flatten = pair_out.reshape(-1,S)
                ifpairs_flatten = ifpairs.reshape(-1).long()
                # pair_out_pred = 
                # loss1 = loss_function_relation(pair_out.squeeze(1), ifpairs)
                loss1 = loss_function_relation(pair_out_flatten, ifpairs_flatten)
                # pair_out_pred.append(torch.gt(F.sigmoid(pair_out),0.5))
                pair_out_pred.append(torch.max(F.softmax(pair_out_flatten,dim=1),dim=1)[1].reshape(B,N))
                ifpairs_true.append(ifpairs)


                loss1.backward()
                optimizer_rel.step()
            # evalatue
            # stop()
            score = eval_relations(pair_out_pred,ifpairs_true)
            logger.info("train relation f1 {} , loss:{}".format(score,loss1))
            scheduler_rel.step()
            # test
            ifpairs_test_true = []
            pair_out_test_pred = []
            ifpairs_test_true_sklearn = []
            pair_out_test_pred_sklearn  = []
            model.pretrain_model.eval()
            for (x, x_obj, y, mask, lens, ifpairs) in test2_phase1_loader:
                if device:
                    x = x.to(device)
                    x_obj = x_obj.to(device)
                    mask = mask.to(device)
                    ifpairs = ifpairs.float().to(device)

                with torch.no_grad():
                    pair_out = model.pretrain_model(x, x_obj, mask)
                    B,N,S = pair_out.shape
                    pair_out_flatten = pair_out.reshape(-1,S)
                    ifpairs_flatten = ifpairs.reshape(-1).long()
                    pair_out_test_pred.append(torch.max(F.softmax(pair_out_flatten,dim=1),dim=1)[1].reshape(B,N))
                    ifpairs_test_true.append(ifpairs)
            score = eval_relations(pair_out_test_pred,ifpairs_test_true)
            # score2 = eval_relations_sklearn(pair_out_test_pred_sklearn,ifpairs_test_true_sklearn)
            logger.info("train relation f1 {} ".format(score))
            print("test relation Acc_RR: {},Acc_VV: {},f1_RR: {} ,f1_VV: {}".format(score[0],score[1],score[2],score[3]),file=fp_pre)
            print("  ",file = fp_pre)
            f1_RR, f1_VV = score[2],score[3]
            if f1_RR_best+f1_VV<f1_RR+f1_VV:
                f1_RR_best =f1_RR_best
                f1_VV_best =f1_VV
                torch.save({
                                # 'model_rel': model.pretrain_model.state_dict(),
                                'model_rel': model.pretrain_model.state_dict(),
                                'optimizer_rel': optimizer_rel.state_dict(),
                            }, os.path.join(args.pre_train_path,'rel_mode_{}_{}.pth'.format(f1_RR,f1_VV)))

    else:
        assert 0 
        train_phase1_loader = dlbb.train_phase1_loader 
        for i in range(args.relation_epoches):
            # loss_function_relation = nn.BCEWithLogitsLoss()
            loss_function_relation = nn.CrossEntropyLoss()
            ifpairs_true = []
            pair_out_pred = []
            for (x, x_obj, y, mask, lens, ifpairs) in train_phase1_loader:
                model.pretrain_model.train()
                optimizer_rel.zero_grad()
                if device:
                    x = x.to(device)
                    x_obj = x_obj.to(device)
                    mask = mask.to(device)
                    ifpairs = ifpairs.float().to(device)
                pair_out = model.pretrain_model(x, x_obj, mask)  # seq_len * bs * labels
                B,N,S = pair_out.shape
                pair_out_flatten = pair_out.reshape(-1,S)
                ifpairs_flatten = ifpairs.reshape(-1).long()
                # loss1 = loss_function_relation(pair_out.squeeze(1), ifpairs)
                loss1 = loss_function_relation(pair_out_flatten, ifpairs_flatten)
                # pair_out_pred.append(torch.gt(F.sigmoid(pair_out),0.5))
                pair_out_pred.append(torch.max(F.softmax(pair_out_flatten),dim=1)[1].reshape(B,N))
                ifpairs_true.append(ifpairs)
                loss1.backward()
                optimizer_rel.step()
            # evalatue
            score = eval_relations(pair_out_pred,ifpairs_true)
            logger.info("relation f1 {} , loss:{}".format(score,loss1))
            scheduler_rel.step()
            # torch.cuda.empty_cache()
        torch.save({
                        # 'model_rel': model.pretrain_model.state_dict(),
                        'model_rel': model.pretrain_model.state_dict(),
                        'optimizer_rel': optimizer_rel.state_dict(),
                    }, args.pre_train_path)

    # torch.cuda.empty_cache()
    
    return model
def run_train_epoch(args,optimizer_rel,scheduler_rel,train_phase1_loader, global_step, model, param_optimizer,
                    train_examples, train_features, train_dataloader,
                    eval_examples, eval_features, eval_dataloader,
                    test_examples, test_features, test_dataloader,
                    optimizer, n_gpu, device, logger, log_path, save_path,
                    save_checkpoints_steps, start_save_steps, best_f1):

    running_loss, count = 0.0, 0
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self

        #all_input_ids, all_input_mask, all_segment_ids, all_example_index,all_start_positions, all_end_positions, all_bio_labels,all_polarity_positions, all_image_labels,all_image_raw_data)
        input_ids, input_mask, segment_ids, example_indices,start_positions, end_positions,\
            bio_labels, polarity_positions,image_labels,image_raw_data = batch
        batch_start_logits, batch_end_logits, _,_ = model('extraction', input_mask, input_ids=input_ids,\
             token_type_ids=segment_ids,image_labels=image_labels,image_raw_data=image_raw_data)
        batch_features, batch_results = [], []
        # torch.cuda.empty_cache()
        for j, example_index in  enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            train_feature = train_features[example_index.item()]
            unique_id = int(train_feature.unique_id)
            batch_features.append(train_feature)
            batch_results.append(RawSpanResult(unique_id = unique_id, start_logits = start_logits, end_logits = end_logits))
        span_starts, span_ends, labels, label_masks = span_annotate_candidates(train_examples, batch_features,
                                                                               batch_results,
                                                                               args.filter_type, True,
                                                                               args.use_heuristics,
                                                                               args.use_nms,
                                                                               args.logit_threshold,
                                                                               args.n_best_size,
                                                                               args.max_answer_length,
                                                                               args.do_lower_case,
                                                                               args.verbose_logging, logger)

        span_starts = torch.tensor(span_starts, dtype=torch.long)
        span_ends = torch.tensor(span_ends, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        label_masks = torch.tensor(label_masks, dtype=torch.long)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        labels = labels.to(device)
        label_masks = label_masks.to(device)

        loss = model('train', input_mask, input_ids=input_ids, token_type_ids=segment_ids,
                     start_positions=start_positions, end_positions=end_positions,
                     span_starts=span_starts, span_ends=span_ends,
                     polarity_labels=labels, label_masks=label_masks,
                     image_labels=image_labels,image_raw_data=image_raw_data)
        
        # torch.cuda.empty_cache()
        loss = post_process_loss(args, n_gpu, loss)
        loss.backward()
        running_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 or args.optimize_on_cpu:
                if args.fp16 and args.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        param.grad.data = param.grad.data / args.loss_scale
                is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    args.loss_scale = args.loss_scale / 2
                    model.zero_grad()
                    continue
                optimizer.step()
                copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
            else:
                optimizer.step()
            model.zero_grad()
            global_step += 1
            count += 1

            if global_step % save_checkpoints_steps == 0 and count != 0:
                logger.info("step: {}, loss: {:.4f}".format(global_step, running_loss / count))
            if global_step % save_checkpoints_steps == 0 and global_step > start_save_steps and count != 0:  # eval & save model
                logger.info("***** Running evaluation *****")
                model.eval()
                metrics =      evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger)
                metrics_test = evaluate(args, model, device, test_examples, test_features, test_dataloader, logger)
                
                    
                # torch.cuda.empty_cache()
                f = open(log_path, "a")
                print("dev step: {}, loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {}),(relation_R: {},relation_V: {})"
                      .format(global_step, running_loss / count, metrics['p'], metrics['r'],
                              metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant'],metrics['relations_R'],metrics['relations_V']), file=f)
                f = open(log_path, "a")
                print("test step: {}, loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {}),(relation_R: {},relation_V: {})"
                      .format(global_step, running_loss / count, metrics_test['p'], metrics_test['r'],
                              metrics_test['f1'], metrics_test['common'], metrics_test['retrieved'], metrics_test['relevant'],metrics_test['relations_R'],metrics_test['relations_V']), file=f)
                print(" ", file=f)
                f.close()
                running_loss, count = 0.0, 0
                model.train()
                # detach_submodel(model.pretrain_model)
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': global_step
                    }, save_path)
                if args.debug:
                    break
    return global_step, model, best_f1


def evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):
    all_results = []
    relations_twitters=[]
    input_ids_all=[]
    image_raw_data_all=[]
    attention_mask_all =[]
    relations_raw_all=[]
    img_info_all,soraw_all =[],[]
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, example_indices,start_positions, end_positions,bio_labels,polarity_positions, image_labels,image_raw_data = batch
        # input_ids, input_mask, segment_ids, example_indices = batch
        input_ids_all.append(input_ids)
        image_raw_data_all.append(image_raw_data)
        attention_mask_all.append(input_mask)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, sequence_output,relations_twitter = model('extraction', input_mask,
                                                                          input_ids=input_ids,
                                                                          token_type_ids=segment_ids,
                                                                          image_labels=image_labels,
                                                                          image_raw_data=image_raw_data)
        batch_features, batch_results = [], []
        for j, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            batch_features.append(eval_feature)
            batch_results.append(RawSpanResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

        span_starts, span_ends, _, label_masks = span_annotate_candidates(eval_examples, batch_features, batch_results,
                                                                          args.filter_type, False,
                                                                          args.use_heuristics, args.use_nms,
                                                                          args.logit_threshold, args.n_best_size,
                                                                          args.max_answer_length, args.do_lower_case,
                                                                          args.verbose_logging, logger)

        span_starts = torch.tensor(span_starts, dtype=torch.long)
        span_ends = torch.tensor(span_ends, dtype=torch.long)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        sequence_output = sequence_output.to(device)
        with torch.no_grad():
            batch_ac_logits,relations_twitter= model('classification', input_mask, input_ids=input_ids,
                                    span_starts=span_starts,
                                    span_ends=span_ends, sequence_input=sequence_output,
                                    image_labels=image_labels,image_raw_data=image_raw_data)    # [N, M, 4]
        relations_twitters.append(relations_twitter)
        for j, example_index in enumerate(example_indices):
            cls_pred = batch_ac_logits[j].detach().cpu().numpy().argmax(axis=1).tolist()
            start_indexes = span_starts[j].detach().cpu().tolist()
            end_indexes = span_ends[j].detach().cpu().tolist()
            span_masks = label_masks[j]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

    metrics, all_nbest_json = eval_absa(eval_examples, eval_features, all_results,
                                        args.do_lower_case, args.verbose_logging, logger)

    relations_twitters = torch.cat(relations_twitters).cpu()
    metrics['relations_R'] = round(torch.sum(relations_twitters[:,0]).item() / relations_twitters.shape[0],4)
    metrics['relations_V'] = round(torch.sum(relations_twitters[:,1]).item() / relations_twitters.shape[0],4)
    if write_pred:
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))
    return metrics


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default='data/semeval_14', type=str, help="SemEval data dir")
    parser.add_argument("--train_file", default=None, type=str, help="SemEval xml for training")
    parser.add_argument("--predict_file", default=None, type=str, help="SemEval csv for prediction")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=96, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_pretrained", default=1e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_proportion", default=0.5, type=float,
                        help="Proportion of steps to save models for. E.g., 0.5 = 50% of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=12, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--logit_threshold", default=8., type=float,
                        help="Logit threshold for annotating labels.")
    parser.add_argument("--filter_type", default="f1", type=str, help="Which filter type to use")
    parser.add_argument("--use_heuristics", default=True, action='store_true',
                        help="If true, use heuristic regularization on span length")
    parser.add_argument("--use_nms", default=True, action='store_true',
                        help="If true, use nms to prune redundant spans")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--gpu_idx",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--image_path", default="data/twitter2015_images/")
    parser.add_argument("--pre_split_file", default="data/")
    parser.add_argument("--pre_image_obj_features_dir", default="data/Twitter_images")
    parser.add_argument("--pre_train_path", default="data/relations_pretrained_models/rel_pre_model.pth")
    parser.add_argument("--cache_dir", default="data/image_cache_dir/")
    parser.add_argument("--TRC_batch_size", default=8,type=int)
    parser.add_argument("--relation_epoches", default=10,type=int)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.0000001)
    
    
    args = parser.parse_args()

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train and not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict and not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")
    if args.local_rank == -1 or args.no_cuda:
        if args.gpu_idx ==-1:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu",args.gpu_idx)
        # n_gpu = torch.cuda.device_count()
        n_gpu = 1
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))
    save_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    log_path = os.path.join(args.output_dir, 'performance.txt')
    network_path = os.path.join(args.output_dir, 'network.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')
    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
        print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()

    logger.info("***** Preparing model *****")
    model = JointMMwithRel(bert_config)
    if args.init_checkpoint is not None and not os.path.isfile(save_path):
        # relation

        # bert 
        model.bert = bert_load_state_dict(model.bert, torch.load(args.init_checkpoint, map_location='cpu'))
        model.pretrain_model = bert_load_state_dict(model.pretrain_model, torch.load(args.init_checkpoint, map_location='cpu'))
        
        logger.info("Loading model from pretrained checkpoint: {}".format(args.init_checkpoint))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        step = checkpoint['step']
        logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                    .format(save_path, step))

    f = open(network_path, "w")
    for n, param in model.named_parameters():
        print("name: {}, size: {}, dtype: {}, requires_grad: {}"
              .format(n, param.size(), param.dtype, param.requires_grad), file=f)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # the numel means the number of params in a tensor  
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: {}".format(total_trainable_params), file=f)
    print("Total parameters: {}".format(total_params), file=f)
    f.close()

    logger.info("***** Preparing data *****")
    train_examples, train_features, train_dataloader, num_train_steps = None, None, None, None
    eval_examples, eval_features, eval_dataloader = None, None, None
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    if args.do_train:
        logger.info("***** Preparing training *****")
        train_examples, train_features, train_dataloader, num_train_steps = read_train_data(args, tokenizer, logger)
        logger.info("***** Preparing evaluation *****")
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)
        logger.info("***** Preparing testting *****")
        args.predict_file = 'test.txt'
        test_examples, test_features, test_dataloader = read_eval_data(args, tokenizer, logger)

    logger.info("***** Preparing optimizer *****")

    paras = dict(model.pretrain_model.named_parameters())
    paras_new = []
    for k, v in paras.items():
        # paras_new += [{'params': [v], 'lr': 2e-5}]
        if 'bert' in k or 'pre_resnet' in k:
            paras_new += [{'params': [v], 'lr': 1e-6}]
        else:
            paras_new += [{'params': [v], 'lr': 1e-4}]
    # model.fix_params()
    optimizer, param_optimizer = prepare_optimizer(args, model, num_train_steps)
    # optimizer, param_optimizer = prepare_optimizer_2(args, model, num_train_steps)
    # optimizer, param_optimizer = prepare_optimizer_3(args, model, num_train_steps)
    optimizer_rel = torch.optim.Adam(paras_new, weight_decay=args.wdecay)
    scheduler_rel = torch.optim.lr_scheduler.LambdaLR(optimizer_rel, burnin_schedule)
    # for relation model



    global_step = 0
    if os.path.isfile(save_path) :
        checkpoint = torch.load(save_path,map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        logger.info("Loading optimizer from finetuned checkpoint: '{}' (step {})".format(save_path, step))
        global_step = step
    if args.do_train:
        logger.info("***** Running training *****")
        best_f1 = 0
        save_checkpoints_steps = int(num_train_steps / (5 * args.num_train_epochs))
        start_save_steps = int(num_train_steps * args.save_proportion)

        dlbb = DLbb(args,tokenizer)
        train_phase1_loader = dlbb.train_phase1_loader
        model = train_relation_model(args, optimizer_rel,scheduler_rel, tokenizer,global_step, model,device)   
        # detach_submodel(model.pretrain_model)
        #  [ p for n, p in model.pretrain_model.named_parameters()][0].requires_grad
        if args.debug:
            args.num_train_epochs = 1
            save_checkpoints_steps = 20
            start_save_steps = 0
        model.train()
        last_model_pp =None
        for epoch in range(int(args.num_train_epochs)):
            logger.info("***** Epoch: {} *****".format(epoch+1))
            global_step, model, best_f1 = run_train_epoch(args, optimizer_rel,scheduler_rel, train_phase1_loader,global_step, model, param_optimizer,
                                                          train_examples, train_features, train_dataloader,
                                                          eval_examples, eval_features, eval_dataloader,
                                                          test_examples, test_features, test_dataloader,
                                                          optimizer, n_gpu, device, logger, log_path, save_path,
                                                          save_checkpoints_steps, start_save_steps, best_f1)

    if args.do_predict:
        logger.info("***** Running prediction *****")
        if eval_dataloader is None:
            eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)
        #  best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path,map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
            logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()
        model.fix_params()
        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=True)
        f = open(log_path, "a")
        print(metrics)
        print("threshold: {}, step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {}),(relation_R: {},relation_V: {})"
              .format(args.logit_threshold, global_step, metrics['p'], metrics['r'],
                      metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant'],metrics['relations_R'],metrics['relations_V']), file=f)
        print(" ", file=f)
        f.close()
def burnin_schedule(i):
    if i < 10:
        factor = 1
    elif i < 20:
        factor = 0.1
    else:
        factor = 0.01
    return factor
if __name__=='__main__':
    main()