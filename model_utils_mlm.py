import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers import AdamW 
from tqdm import tqdm,trange
import numpy as np
from collections import deque
from copy import deepcopy
import scipy
import ipdb as pdb
import gc

model_classes = {
    'mlm': AutoModelForMaskedLM,
}

class cm_model_mlm(torch.nn.Module):
    def __init__(self, model_name='roberta-base', cache_dir="./cache", model_type='mlm', load_model=False):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        model_class = model_classes[model_type]
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        if load_model:
            self.model = model_class.from_pretrained(model_name, from_tf=False,
                                                   config=self.config,cache_dir=cache_dir)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
             
    def encode(self, tokens, enable_grad=False, additional_truncation=0):
        max_len = self.model.config.max_position_embeddings-additional_truncation - 2
        len_tokens = [len(ch) for ch in tokens['context']]

        if max(len_tokens) > max_len:
            for ch_idx in range(len(tokens['context'])):
                if len(tokens['context'][ch_idx]) > max_len:
                    print ("{} truncating {}->{}".format(tokens['context'][ch_idx][:5], len(tokens['context'][ch_idx]), max_len))
                    tokens['context'][ch_idx] = [tokens['context'][ch_idx][0]] + tokens['context'][ch_idx][-max_len+1:]
        
        ch_outputs = []
        start_idx = tokens['choice_start_idx']
        spans = tokens['choice_spans']
        tokens['context'] = [torch.tensor(v).unsqueeze(0).to(self.device) for v in tokens['context']]
        for ch_idx, ch in enumerate(tokens['context']):
            out = {}
            out['input_ids'] = ch[:, start_idx:start_idx+spans[ch_idx]].detach().clone()
            ch[:, start_idx:start_idx+spans[ch_idx]] = self.tokenizer.mask_token_id
            model_outputs = self.model(input_ids=ch, return_dict=True)
            out['logits'] = model_outputs['logits'][:, start_idx:start_idx+spans[ch_idx], :]
            ch_outputs.append(out)
        return ch_outputs

    def compute_nll(self, ch_outputs, enable_grad=True):
        with torch.set_grad_enabled(enable_grad):
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            
            labels = ch_outputs['input_ids'].contiguous()
            logits = ch_outputs['logits']
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            prob = F.softmax(logits, 2)
            prob = prob.mean(dim=1, keepdim=True)
                        
        outputs = {
            'prob': prob.detach().cpu().numpy(),
            'labels': labels.view(-1).detach().cpu().numpy(),
            'loss': loss.view(-1).detach().cpu().numpy(),
            'score': torch.sum(loss).item(),
        }
        return outputs

    def process_context_and_choices(self, context, choices, additional_truncation=0, tokens=None):
        enable_grad=False
        scores = []
        ch_outputs = self.encode(tokens, additional_truncation=additional_truncation)

        for ch_idx in range(len(choices)):
            self.model.zero_grad()
            ch_output = ch_outputs[ch_idx]
            score = self.compute_nll(ch_output, enable_grad=enable_grad)
            scores.append(score)
        return scores

    def get_token_ids(self, context, dummy_context, choices):
        mask_token_id = self.tokenizer.mask_token_id

        c_tokens = self.tokenizer(context)
        cd_tokens = self.tokenizer(dummy_context)
    
        extra_len = len(c_tokens['input_ids'])-len(cd_tokens['input_ids'])
        mask_index = cd_tokens['input_ids'].index(mask_token_id)

        cd_tokens['input_ids'] = cd_tokens['input_ids'][:mask_index+1]  + [mask_token_id]*extra_len + cd_tokens['input_ids'][mask_index+1:]

        ch_tokens = [self.tokenizer(ch) for ch in choices]

        c_ch_tokens = [c_tokens['input_ids'][:-1] + item['input_ids'][1:] for item in ch_tokens]
        cd_ch_tokens = [cd_tokens['input_ids'][:-1] + item['input_ids'][1:] for item in ch_tokens]

        tokens = {
            'context': c_ch_tokens,
            'choice_start_idx': len(c_tokens['input_ids'][:-1]),
            'choice_spans': [len(item['input_ids'])-2 for item in ch_tokens],
        }

        tokens_dummy = {
            'context': cd_ch_tokens,
            'choice_start_idx': len(cd_tokens['input_ids'][:-1]),
            'choice_spans': [len(item['input_ids'])-2 for item in ch_tokens],
        }
        return tokens, tokens_dummy

    @torch.inference_mode()
    def score(self, example, reduction='sum', additional_truncation=0):
        scores = [{} for _ in range(len(example['choices']))]

        tokens, tokens_dummy = self.get_token_ids(example['context'], example['dummy_context'], example['choices'])

        temp_scores = self.process_context_and_choices(example['context'], example['choices'], additional_truncation=additional_truncation, tokens=tokens)
        for ch_idx in range(len(example['choices'])):
            scores[ch_idx]['res0'] = temp_scores[ch_idx]

        temp_scores = self.process_context_and_choices(example['dummy_context'], example['choices'], tokens=tokens_dummy)
        for ch_idx in range(len(example['choices'])):
            scores[ch_idx]['res1'] = temp_scores[ch_idx]                 
                    
        for ch_idx in range(len(example['choices'])):
            pc,pn = scores[ch_idx]['res0']['prob'][0,0,:], scores[ch_idx]['res1']['prob'][0,0,:]
            scores[ch_idx]['c_prob0'] = pc.reshape(-1)
            scores[ch_idx]['n_prob0'] = pn.reshape(-1)

            del scores[ch_idx]['res0']['prob']
            del scores[ch_idx]['res1']['prob']
        return scores
