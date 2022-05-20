import torch.multiprocessing as mp
import data_utils
import model_utils
import model_utils_mlm
from copy import deepcopy
from tqdm import tqdm, trange
from collections import defaultdict, Counter
import numpy as np
import json
import torch
import ipdb as pdb
import pickle
import os
import sys
import argparse

from transformers import AutoConfig
from analysis_utils import get_factor_stats

model_classes = {
    'clm': model_utils.cm_model,
    'mlm': model_utils_mlm.cm_model_mlm,
}

#set seed
def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()                                
    parser.add_argument("--type_of_model", default="clm", type=str,
                        help="clm/mlm")
    parser.add_argument("--model_name", default='gpt2-xl', type=str,
                        help="Model name.")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="Dir to download models in")  
    parser.add_argument("--data_name", default=None, type=str, required=True,        
                        help="Dataset to eval on")
    parser.add_argument("--data_config", default=None, type=str)
    parser.add_argument("--st_data_path", default="./external/self_talk/data", type=str,
                        help="Path to data")
    parser.add_argument("--data_split", default="dev", type=str,       
                        help="Data split")
    parser.add_argument("--num_processes", default=4, type=int,       
                        help="Number of processes to use") 
    parser.add_argument("--disable_cache", action='store_true')

    parser.add_argument("--fsl_sampling_seed", default=1, type=int)
    parser.add_argument("--fsl_sampling_split", default='train', type=str)
    parser.add_argument("--fsl_n_samples", default=0, type=int)
    parser.add_argument("--fsl_n_episodes", default=100, type=int)
    parser.add_argument("--fsl_episode_start_idx", default=0, type=int)

    parser.add_argument("--compute_ece", action="store_true")

    args = parser.parse_args()
    if args.fsl_n_samples == 0:
        args.fsl_n_episodes = 1
    else:
        args.disable_cache = True

    print (args)

    set_seed(0)

    split = args.data_split
    mp.set_start_method("spawn")
    
    cached_output_name = 'cached_output/{}_{}_{}.p'.format(
        '{}{}'.format(args.data_name, '' if args.data_config is None else args.data_config),
        args.model_name,
        args.data_split
    )
   
    if not args.disable_cache:
        print ('Cached output path {} Exists? {}'.format(cached_output_name,
                                                 os.path.exists(cached_output_name)))

        cache_exists = os.path.exists(cached_output_name)
    else:
        cache_exists = False
   
    model_class = model_classes[args.type_of_model]

    model = model_class(model_name=args.model_name, cache_dir=args.cache_dir, load_model=not cache_exists)
    mask_token = model.tokenizer.mask_token
    D = data_utils.load_data(args.data_name, args.st_data_path, mask_token=mask_token, tokenizer=model.tokenizer, data_config=args.data_config)
    print (D['processed'][split][:3])

    if args.fsl_n_samples > 0:
        len_fsl_pool = len(D['processed'][args.fsl_sampling_split])
        frng = np.random.RandomState(seed=args.fsl_sampling_seed)

    for ep in range(args.fsl_n_episodes):
        if ep < args.fsl_episode_start_idx:
            continue
        fsl_name = ''
        if args.fsl_n_samples > 0:
            sep_token = '\n'
            fsl_name = 'fsl{}_e{}'.format(args.fsl_n_samples, ep)
            fsl_train_indices = frng.choice(len_fsl_pool, size=args.fsl_n_samples, replace=False)
            fsl_examples = [data_utils.format_labeled_example(D['converted'][args.fsl_sampling_split][idx]) for idx in fsl_train_indices]
            fsl_prefix = sep_token.join(fsl_examples)
            print ('*'*80)
            print ("Ep", ep, "Train Indices", fsl_train_indices)
            print ("FSL prefix", fsl_prefix)

            D_fsl = deepcopy(D)
            for e_idx in range(len(D_fsl['processed'][split])):
                 D_fsl['processed'][split][e_idx]['context'] = sep_token.join([fsl_prefix, D_fsl['processed'][split][e_idx]['context']])
                 D_fsl['processed'][split][e_idx]['dummy_context'] = sep_token.join([fsl_prefix, D_fsl['processed'][split][e_idx]['dummy_context']])
            print (D_fsl['processed'][split][:3])
        else:
            D_fsl = D

        results_file = 'results/{}_{}_{}{}.json'.format(
            '{}{}'.format(args.data_name, '' if args.data_config is None else args.data_config),
            args.model_name,
            args.data_split,
            fsl_name
        )
        print ('results at', results_file)

        scores = get_scores(args, model, D_fsl, split, cached_output_name, cache_exists)
        process_model_outputs(args, model, D_fsl, split, scores, results_file)

def get_scores(args, model, data, split, cached_output_name=None, cache_exists=False):
    D = data
    if not cache_exists:
        num_processes = args.num_processes

        if num_processes > 1:
            model.share_memory()
            manager = mp.Manager()
            return_dict = manager.dict()
        else:
            return_dict = {}

        processes = []
        block_size = int((len(D['processed'][split])+num_processes)/num_processes)
        indices = list(range(len(D['processed'][split])))
        for rank in range(num_processes):
            start_idx = block_size*rank
            end_idx = block_size*(rank+1)
            p_indices = indices[start_idx:end_idx]
            if num_processes == 1:
                model_utils.score_fn(rank, model, D['processed'][split],
                                     p_indices, return_dict)
            else:
                p = mp.Process(target=model_utils.score_fn, args=(rank, model,
                                                  D['processed'][split],
                                                  p_indices, return_dict))
                p.start()
                processes.append(p)

        if num_processes > 1:
            for p in processes:
                p.join()

        scores = [return_dict[idx] for idx in range(len(D['processed'][split]))]
        
        if not args.disable_cache:
            with open(cached_output_name, 'wb') as f:
                pickle.dump(scores, f)    
    else:
        with open(cached_output_name, 'rb') as f:
            scores = pickle.load(f)
    
    return scores

def process_model_outputs(args, model, data, split, scores, results_file):
    D = data
    results = defaultdict(dict)
    saved_preds = {}
    for abl in ['answer_only', 'answer_only_worst']:
        preds = model_utils.predict(scores, mode=abl)[0]
        saved_preds[abl] = preds

    examples = D['converted'][split]
    factor_kwargs = {}
    factor_kwargs['tokenizer'] = model.tokenizer
    factor_kwargs['min_th'] = 25
    factor_kwargs['neutral_preds'] = saved_preds['answer_only']
    factor_kwargs['neutral_worst_preds'] = saved_preds['answer_only_worst']

    for abl in ['answer_only_norm', 'answer_only_worst_norm']:
        preds = model_utils.predict(scores, mode=abl)[0]
        saved_preds[abl] = preds
    factor_kwargs['neutral_preds_norm'] = saved_preds['answer_only_norm']
    factor_kwargs['neutral_worst_preds_norm'] = saved_preds['answer_only_worst_norm']

    for abl in (['answer_only', 'answer_only_worst', 'answer_only_norm', 'answer_only_worst_norm', 'uncalibrated', 'length_normalized'] + 
                (['token_calibration'] if args.type_of_model == 'clm' else []) +
                ['alc_unscaled', 'alc_tvd', 'alc_bc']):

        preds, preds_info = model_utils.predict(scores, mode=abl)
        saved_preds[abl] = preds
        acc = D['acc_fn'](split, preds)
        results[f'overall_{abl}'] = acc
        print_str = '{} {} {:.2f}'.format(abl, split, acc)

        #ece
        if args.compute_ece:
            from misc_utils import compute_ece
            acc_list = D['acc_fn'](split, preds, reduction='none')
            ece = compute_ece(acc_list, preds_info['conf'])
            print_str = '{} ece {:.2f}'.format(print_str, ece)
            results[f'ece_{abl}'] = ece

        print (print_str)

        for k in preds_info:
            if k in ['conf']:
                continue
            results[f'overall_{k}_{abl}'] = float(preds_info[k])

        for bias in ['longest', 'shortest']:
            factor_kwargs['bias'] = bias
            res = get_factor_stats(preds, examples, 'length', **factor_kwargs)
            results = update_results(results, abl, res, f'length_{bias}')  

        for bias in ['first']:
            factor_kwargs['bias'] = bias
            res = get_factor_stats(preds, examples, 'pos', **factor_kwargs)
            results = update_results(results, abl, res, f'pos_{bias}')  

        for bias in ['neutral', 'neutral_worst']:
            factor_kwargs['bias'] = bias
            res = get_factor_stats(preds, examples, 'lm', **factor_kwargs)
            results = update_results(results, abl, res, f'lm_{bias}')  

        for bias in ['neutral_norm', 'neutral_worst_norm']:
            factor_kwargs['bias'] = bias
            res = get_factor_stats(preds, examples, 'lm', **factor_kwargs)
            results = update_results(results, abl, res, f'lm_{bias}')  

    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4, sort_keys=True))

def update_results(results, abl, res, tag):
    #add two versions, one flattened for ease of comparing quickly
    results[f'qsel_{abl}_{tag}'] = res
    for k,v in res.items():
        if isinstance(v, dict):
            for kk,vv in v.items():
                resname = 'sel_{}_{}_{}_{}'.format(tag, k, kk, abl)
                results[resname] = vv
        else:
            resname = 'sel_{}_{}_{}'.format(tag, k, abl)
            results[resname] = v
    return results

if __name__ == "__main__":
    main()
