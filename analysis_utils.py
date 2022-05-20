import numpy as np
from nltk import pos_tag, word_tokenize
from collections import defaultdict, Counter
from sklearn.metrics import matthews_corrcoef
from copy import deepcopy
import pdb

NULLSTR = "None"
NULLVAL = -100

def get_factor_stats_lm(examples, kwargs):
    if kwargs['bias'] in ['neutral']:
        sel_answers = kwargs['neutral_preds']
    elif kwargs['bias'] in ['neutral_worst']:
        sel_answers = kwargs['neutral_worst_preds']
    elif kwargs['bias'] in ['neutral_norm']:
        sel_answers = kwargs['neutral_preds_norm']
    elif kwargs['bias'] in ['neutral_worst_norm']:
        sel_answers = kwargs['neutral_worst_preds_norm']
    
    factor_labels = []
    factor_preds = None
    if kwargs['bias'] in ['neutral', 'neutral_worst', 'neutral_norm', 'neutral_worst_norm']:
        factor_label_list = [kwargs['bias'], kwargs['bias']+'_compl']
        for ex_idx, ex in enumerate(examples):
            sel_idx = sel_answers[ex_idx]
            factor_labels.append([factor_label_list[0] if idx == sel_idx
                            else factor_label_list[1] for idx in range(len(ex['choices']))])
            factor_preds = np.array(sel_answers)

    results = {
            'factor_labels': factor_labels,
            'factor_label_list': factor_label_list,
            'factor_preds': factor_preds
    }
    return results


def get_factor_stats_pos(examples, kwargs):
    if not get_factor_stats_pos.all_tags:
        all_tags = []
        for ex_idx, ex in enumerate(examples):
            tags = [pos_tag(word_tokenize(item), tagset='universal') for item in ex['choices']]
            all_tags.append(tags)
        get_factor_stats_pos.all_tags = all_tags
    else:
        all_tags = get_factor_stats_pos.all_tags

    factor_label_list = ['ADJ', 'ADP', 'ADV', 'CONV', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    factor_labels = []
    factor_preds = None
    if kwargs['bias'] in ['first']:
        factor_labels = [[item[0][1] for item in ex_list] for ex_list in all_tags]
    
    results = {
            'factor_labels': factor_labels,
            'factor_label_list': factor_label_list,
            'factor_preds': factor_preds
    }
    return results

get_factor_stats_pos.all_tags = []
   
def get_factor_stats_length(examples, kwargs):
    if not get_factor_stats_length.lengths:
        lengths = []
        for ex_idx, ex in enumerate(examples):
            lengths.append([len(kwargs['tokenizer'](item).input_ids) for item in ex['choices']])
        get_factor_stats_length.lengths = lengths
    else:
        lengths = get_factor_stats_length.lengths
    max_choices = np.max([len(ex['choices']) for ex in examples])
    max_length = np.max([np.max(item) for item in lengths])

    if kwargs['bias'] == 'longest':
        sel_answers = [np.argmax(item) for item in lengths]
    elif kwargs['bias'] == 'shortest':
        sel_answers = [np.argmin(item) for item in lengths]

    #keep this as a list, to allow variable number of choices
    factor_labels = []
    factor_preds = None
    if kwargs['bias'] in ['longest', 'shortest']:
        factor_label_list = [kwargs['bias'], kwargs['bias']+'_compl']
        for ex_idx, ex in enumerate(examples):
            sel_idx = sel_answers[ex_idx]
            factor_labels.append([factor_label_list[0] if idx == sel_idx
                            else factor_label_list[1] for idx in range(len(ex['choices']))])
        factor_preds = np.array(sel_answers)

    results = {
            'factor_labels': factor_labels,
            'factor_label_list': factor_label_list,
            'factor_preds': factor_preds
    }
    return results
get_factor_stats_length.lengths = []   

def get_factor_stats(preds, examples, factor, **kwargs):
    if not get_factor_stats.stats:
        stats = {}
        labels = [ex['label'] for ex in examples]
        stats['labels'] = np.array(labels)
        get_factor_stats.stats = stats
    else:
        stats = get_factor_stats.stats

    #factors will remain constant for a run
    cached_factor_name = factor if not kwargs.get('bias', '') else "{}_{}".format(factor, kwargs['bias'])
    if 1:# cached_factor_name not in stats:
        factor_stats = get_factor_stats_fn[factor](examples, kwargs)
        get_factor_stats.stats[cached_factor_name] = factor_stats
    else:
        factor_stats = get_factor_stats.stats[cached_factor_name]

    factor_labels = factor_stats['factor_labels']
    factor_label_list = factor_stats['factor_label_list']

    if len(factor_label_list) == 0:
        null_val = NULLSTR
    elif type(factor_label_list[0]) is str:
        null_val = NULLSTR
    else:
        null_val = NULLVAL

    #is_binary = True if len(factor_label_list) == 1 else False
    #if is_binary:
    #    factor_label_list.append(factor_label_list[0]+'_compl')
    
    #lfc:label factor count, pfc; pred factor count, pfcc: pred factor correct count
    preds = np.array(preds)
    lfc, pfc, pfcc = defaultdict(int), defaultdict(int), defaultdict(int)
    p,r,f1 = defaultdict(int), defaultdict(int), defaultdict(int)
    label_factors = np.array([factor_labels[ex_idx][idx] for ex_idx,idx in enumerate(stats['labels'])])
    pred_factors = np.array([factor_labels[ex_idx][idx] for ex_idx,idx in enumerate(preds)])
    common_factors = pred_factors.copy()
    try:
        common_factors[stats['labels']!=preds] = null_val
    except:
        pdb.set_trace()
    for factor_label in factor_label_list:
        lfc[factor_label] = int(np.sum(label_factors==factor_label))
        pfc[factor_label] = int(np.sum(pred_factors==factor_label))
        pfcc[factor_label] = int(np.sum(common_factors==factor_label))
        
        p[factor_label] = 100.0*pfcc[factor_label]/pfc[factor_label] if pfc[factor_label] > 0 else 0
        r[factor_label] = 100.0*pfcc[factor_label]/lfc[factor_label] if lfc[factor_label] > 0 else 0
        f1[factor_label] = 2 * p[factor_label] * r[factor_label] / (p[factor_label]+r[factor_label]) if p[factor_label]+r[factor_label] > 0 else 0

    min_th = kwargs["min_th"]
    for factor_label in deepcopy(factor_label_list):
        if lfc[factor_label] < min_th:
            factor_label_list.remove(factor_label)

    #in case some examples are not considered, their labels will be marked 'None'
    #and micro scores will be different from overall acc
    #micro scores
    #print (factor, kwargs['bias'], factor_label_list)
    n_correct = np.sum([pfcc[item] for item in factor_label_list])
    n_preds = np.sum([pfc[item] for item in factor_label_list])
    n_labels = np.sum([lfc[item] for item in factor_label_list])
    micro_p = 100.0 * n_correct/n_labels if n_labels > 0 else 0 #these calculations are redundant
    micro_r = 100.0 * n_correct/n_labels if n_labels > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p+micro_r) if micro_p+micro_r > 0 else 0

    #macro scores
    macro_p = 1.0/len(factor_label_list) * np.sum([p[item] for item in factor_label_list]) if len(factor_label_list) > 0 else 0
    macro_r = 1.0/len(factor_label_list) * np.sum([r[item] for item in factor_label_list]) if len(factor_label_list) > 0 else 0
    macro_f1 = 1.0/len(factor_label_list) * np.sum([f1[item] for item in factor_label_list]) if len(factor_label_list) > 0 else 0

    #mcc, agreement
    mcc, agreement = None, None
    pred_is_correct = stats['labels']==preds
    if factor_stats['factor_preds'] is not None:
        factor_is_correct = stats['labels']==factor_stats['factor_preds']
        mcc = matthews_corrcoef(factor_is_correct, pred_is_correct)
        agreement = 100.0 * np.sum(factor_is_correct==pred_is_correct)/len(pred_is_correct) 

    results = {
        'lfc': lfc, 'pfc': pfc, 'pfcc': pfcc,
        'p': p, 'r': r, 'f1': f1,
        'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1,
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'length': int(n_labels),
        'mcc': mcc, 'agreement': agreement
    }
    return results

get_factor_stats.stats = {}

get_factor_stats_fn = {
    'length': get_factor_stats_length,
    'pos': get_factor_stats_pos,
    'lm': get_factor_stats_lm,
}
