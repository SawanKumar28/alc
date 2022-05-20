import os
import json
import numpy as np
from collections import Counter
import re
import ipdb as pdb
from copy import deepcopy
import torch

G_MASK_TOKEN = "[[[MASK]]]"

def convert_commonsenseqa(example, **kwargs):
    question =  example['question']['stem'].strip()
    if question.endswith('.') or question.endswith(','):
        question = question[:-1].strip() + '?'
    if not question.endswith('?'):
        question = question + '?'

    choices = [ch['text'].strip() for ch in example['question']['choices']]
    label = ['A','B','C','D','E'].index(example['answerKey']) if 'answerKey' in example else None

    output = {
        'context': '',
        'question': 'Question: {} Answer:'.format(question),
        'question_masked': 'Question: {} Answer:'.format(G_MASK_TOKEN),
        'choices': choices,
        'label': label,
        'dummy_context': 'Answer:',
    }
    return output

def convert_copa(example, **kwargs):
    question = example['premise']
    question = question[0].upper() + question[1:]
    if question.endswith('.') or question.endswith(','):
        question = question[:-1].strip()

    choices = [example['choice1'], example['choice2']]
    choices = [ch[0].lower()+ch[1:] for ch in choices]
    label = example['label'] if 'label' in example else None

    question = '{}, because'.format(question) if example['question'] == 'cause' else '{}, so'.format(question)
    dummy_ctx_str = ', because' if example['question'] == 'cause' else ', so'

    output = {
        'context': '',
        'question': question,
        'question_masked': G_MASK_TOKEN,
        'choices': choices,
        'label': label,
        'dummy_context': dummy_ctx_str,
    }
    return output

def convert_piqa(example, **kwargs):
    context = example['goal'].strip()
    if context.endswith(','):
        context = context[:-1] + '.'
    elif not context.endswith('?') and not context.endswith('.'):
        context = context + '.'
    context = context[0].upper() + context[1:]
    question = 'Question: {} Answer:'.format(context)

    choices = [example['sol1'], example['sol2']]
    choices = [ch[0].upper()+ch[1:] for ch in choices]
    label = example['label'] if 'label' in example else None

    output = {
        'context': '',
        'question': question,
        'question_masked': 'Question: {} Answer:'.format(G_MASK_TOKEN),
        'choices': choices,
        'label': label,
        'dummy_context': 'Answer:',
    }
    return output

#Taken from self-talk https://github.com/vered1986/self_talk.git
QUESTION_TO_ANSWER_PREFIX_socialiqa = {
              "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
              "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
              "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
              "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
              "What will (.*) do next?": r"[SUBJ] then",
              "How would (.*) feel after?": r"[SUBJ] then",
              "How would you describe (.*)?": r"[SUBJ] is seen as",
              "What kind of person is (.*)?": r"[SUBJ] is seen as",
              "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
              "Why did (.*) do that?": r"Before, [SUBJ] wanted",
              "Why did (.*) do this?": r"Before, [SUBJ] wanted",
              "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
              "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What will happen to (.*)?": r"[SUBJ] then",
              "What will happen to (.*) next?": r"[SUBJ] then"
}
def convert_socialiqa(example, **kwargs):
    context = example['context']
    context = context if context.endswith('.') else context + '.'
    question = example['question']
    choice_names = ['answer'+item for item in ['A', 'B', 'C']]
    choices = [example[item] for item in choice_names]
    answer_prefix = ""
    for template, ans_prefix in QUESTION_TO_ANSWER_PREFIX_socialiqa.items():
        m = re.match(template, question)
        if m is not None:
            answer_prefix = ans_prefix.replace("[SUBJ]", m.group(1))
            break
    if answer_prefix == "":
        answer_prefix = question.replace("?", "is")
    answer_prefix = answer_prefix.replace("?", "")

    formatted_choices = []
    for choice in choices:
        ch_text = choice.strip().lower()
        for item in ['wanted to ', 'needed to ', 'to ']:
            if ch_text.startswith(item) and answer_prefix.endswith(item[:-1]):
                ch_text = ch_text[len(item):].strip()
                break
        formatted_choices.append(' '+ch_text)            
    
    label = ['A', 'B', 'C'].index(example['correct'])  

    output = {
        'context': context,
        'question': answer_prefix,
        'choices': formatted_choices,
        'label': label,
        'question_masked': '{} {}'.format(G_MASK_TOKEN, answer_prefix),
        'dummy_context': answer_prefix,
    }
    return output

def convert_mctaco(example, **kwargs):
    context = example['context'].strip()
    question = example['question']
    if question.endswith('.') or question.endswith(','):
        question = question[:-1].strip() + '?'
    if not question.endswith('?'):
        question = question + '?'   
    question = 'Question: {} Answer:'.format(question)
    
    choices = example['choices']
    label = example['label'] if 'label' in example else None
    
    output = {
        'context': context,
        'question': question,
        'question_masked': 'Question {} Answer:'.format(G_MASK_TOKEN),
        'choices': choices,
        'label': label,
        'dummy_context': 'Answer:',
    }
    return output

def convert_winogrande(example, **kwargs):
    text = example['sentence'].strip()
    left, right = text.split('_')
    context = left.strip()
    choices = [example['option1'], example['option2']]
    choices = [item+right for item in choices]

    label = int(example['answer'])-1  if 'answer' in example else None
   
    dummy_context = ' '
    question_masked = G_MASK_TOKEN
    left_split = left.split()
    if len(left_split) > 0:
        dummy_context = left_split[-1]
        question_masked = '{} {}'.format(G_MASK_TOKEN, left_split[-1])

    output = {
        'context': '',
        'question': context,
        'choices': choices,
        'label': label,
        'question_masked': question_masked,
        'dummy_context': dummy_context,
    }
    return output

def convert_hendrycks_test(example, **kwargs):
    question =  example['question'].strip()
    label = example['answer']
    choices = [ch.strip() for ch in example['choices']]

    output = {
        'context': '',
        'question': 'Question: {} Answer:'.format(question),
        'choices': choices,
        'label': label,
        'dummy_context': 'Answer:',
    }
    return output

def convert_swag(example, data_version='v1', **kwargs):
    question =  example['startphrase'].strip()
    sent2 = example['sent2'].strip()

    choices = [example[ch_name] for ch_name in ['ending0', 'ending1', 'ending2', 'ending3']]
    choices = [ch.strip() for ch in choices]

    label = example['label']
    output = {
        'context': '',
        'question': question,
        'choices': choices,
        'label': label,
        'dummy_context': sent2,
    }
    return output

def convert_dream(example, data_version='v1', **kwargs): #TODO, another newline before the question
    context = '\n'.join(example['dialogue'])
    question = example['question']
    choices = example['choice']
    label = example['choice'].index(example['answer'])
    output = {
        'context': context,
        'question': 'Question: {} Answer:'.format(question),
        'choices': choices,
        'label': label,
        'dummy_context': 'Question: {} Answer:'.format(question),
    }
    return output

def convert_arc(example, data_version='v1', **kwargs):
    question =  example['question']['stem'].strip()
    if question.endswith('.') or question.endswith(','):
        question = question[:-1].strip() + '?'
    if not question.endswith('?'):
        question = question + '?'

    choices = [ch['text'].strip() for ch in example['question']['choices']]
    label_list = [ch['label'] for ch in example['question']['choices']]
    label = label_list.index(example['answerKey']) if 'answerKey' in example else None

    output = {
        'context': '',
        'question': 'Question: {} Answer:'.format(question),
        'choices': choices,
        'label': label,
        'dummy_context': 'Answer:',
    }
    return output

data_converters = {
    'COPA': convert_copa,
    'commonsenseqa': convert_commonsenseqa,
    'piqa': convert_piqa,
    'arc_easy': convert_arc,
    'arc_challenge': convert_arc,
    'socialiqa': convert_socialiqa,
    'mctaco': convert_mctaco,
    'winogrande': convert_winogrande,
    'hendrycks_test': convert_hendrycks_test,
    'swag': convert_swag,
    'dream': convert_dream,
}

def format_example(example, mask_token=None, tokenizer=None):
    formatted_text = {}
    if type(example['context']) is list:
        formatted_text['context'] = example['context']
    elif example['context'] == "":
        formatted_text['context'] = '{}'.format(example['question'])
    elif example['question'] == "":
        formatted_text['context'] = '{}'.format(example['context'])
    else:
        formatted_text['context'] = '{} {}'.format(example['context'], example['question'])
        
    formatted_text['choices'] = [' '+ch for ch in example['choices']]
        
    if mask_token is None:
        formatted_text['dummy_context'] = example['dummy_context']
    else: #assume mlm
        if tokenizer is None:
            raise ValueError("If mask token is used, tokenizer must be set")
        formatted_text['dummy_context'] = example['question_masked'].replace(G_MASK_TOKEN, mask_token)
    
    return formatted_text

def format_labeled_example(example):
    formatted_text = {}
    if type(example['context']) is list:
        formatted_text['context'] = example['context']
    elif example['context'] == "":
        formatted_text['context'] = '{}'.format(example['question'])
    elif example['question'] == "":
        formatted_text['context'] = '{}'.format(example['context'])
    else:
        formatted_text['context'] = '{} {}'.format(example['context'], example['question'])
    
    answer_text = example['choices'][example['label']]
    formatted_text['context'] = '{} {}'.format(formatted_text['context'], answer_text)
    return formatted_text['context']

def load_test_labels(data_name, examples, data_path='./external/'):
    if data_name == 'COPA':
        dir_path = os.path.join(data_path, 'COPA', 'COPA-resources', 'results')
        fname = os.path.join(dir_path, 'gold.test')
        with open(fname) as f:
            lines = f.readlines()
        labels = [int(item.split()[2]) for item in lines]
        for ex_idx in range(len(examples)):
            examples[ex_idx]['label'] = labels[ex_idx]
    return examples

def get_lines(p):
    if p.endswith('.jsonl'):
        with open(p) as f:
            jsonl_content = f.read()
        result = [json.loads(jline) for jline in jsonl_content.splitlines()]
    else:
        df = pd.read_csv(p)
        result = df.to_dict('record')
    return result

hf_data = [
    'hendrycks_test',
    'swag',
    'dream',
]
def get_hf_data(data_name, data_config, cache_dir="./cache/"):
    from datasets import load_dataset, Dataset

    if data_name == 'hendrycks_test':
        from hendrycks_test_categories import category_maps
        if data_config in category_maps:
            categories = category_maps[data_config]
            
            raw_datasets = load_dataset(data_name, categories[0], cache_dir=cache_dir)
            raw_datasets = {k:raw_datasets[k].to_pandas() for k in ['dev', 'test']}

            for c in categories[1:]:
                raw_datasets_c = load_dataset(data_name, c, cache_dir=cache_dir)
                raw_datasets_c = {k:raw_datasets_c[k].to_pandas() for k in ['dev', 'test']}
                for split in ['dev', 'test']:
                    raw_datasets[split] = raw_datasets[split].append(raw_datasets_c[split])
            raw_datasets = {k:Dataset.from_pandas(v) for k,v in raw_datasets.items()}
        else:
            raw_datasets = load_dataset(data_name, data_config, cache_dir=cache_dir)
            raw_datasets = {k:v for k,v in raw_datasets.items()}
    else:
        raw_datasets = load_dataset(data_name, data_config, cache_dir=cache_dir)
        raw_datasets = {k:v for k,v in raw_datasets.items()}
        raw_datasets['dev'] = raw_datasets['validation']
        del raw_datasets['validation']
    return raw_datasets

def get_datasets(data_name, data_path='./external', data_config=None):
    if data_name in hf_data:
        return get_hf_data(data_name, data_config)

    devname, testname = 'dev.jsonl', 'test.jsonl'
    if data_name == 'arc_easy':
        data_dir = os.path.join('./external', "ARC/ARC-V1-Feb2018-2/ARC-Easy")
        devname, testname = 'ARC-Easy-Dev.jsonl', 'ARC-Easy-Test.jsonl'
    elif data_name == 'arc_challenge':
        data_dir = os.path.join('./external', "ARC/ARC-V1-Feb2018-2/ARC-Challenge")
        devname, testname = 'ARC-Challenge-Dev.jsonl', 'ARC-Challenge-Test.jsonl' 
    else:
        data_dir = os.path.join(data_path, data_name)
    dir_download_path = os.path.join('external', data_name)
    
    datasets = {}
    datasets['dev'] =  get_lines(os.path.join(data_dir, devname))
    datasets['test'] = get_lines(os.path.join(data_dir, testname))
            
    if data_name == 'commonsenseqa':
        #datasets['train'] = get_lines(os.path.join(dir_download_path, 'train_rand_split.jsonl'))
        #The original website seems down, try hf
        from datasets import load_dataset, Dataset
        raw_datasets = load_dataset('commonsense_qa', None, cache_dir='./cache')
        raw_datasets = {k:v for k,v in raw_datasets.items()}
        def map_to_orig(example):
            out = {
                'answerKey': example['answerKey'],
                'id': example['id'],
                'question': {
                    'question_concept': example['question_concept'],
                    'choices': [{'label':l, 'text': t} for l,t in zip(example['choices']['label'], example['choices']['text'])],
                    'stem': example['question'],
                }
            }
            return out
        datasets['train'] = raw_datasets['train'].map(map_to_orig, remove_columns=['question_concept', 'choices'])
    elif data_name == 'socialiqa':
        datasets['train'] = get_lines(os.path.join(dir_download_path, 'socialIQa_v1.4_trn.jsonl'))
    elif data_name == 'piqa':
        datasets['train'] = get_lines(os.path.join(dir_download_path, 'train.jsonl'))
        with open(os.path.join(dir_download_path, 'train-labels.lst'), 'r') as f:
            train_labels = f.readlines()
        train_labels = [int(item.strip()) for item in train_labels]
        for idx in range(len(datasets['train'])):
            datasets['train'][idx]['label'] = train_labels[idx]
    elif data_name == 'winogrande':
        datasets['train'] = get_lines(os.path.join(dir_download_path,'winogrande_1.1/train_debiased.jsonl'))
        with open(os.path.join(dir_download_path, 'winogrande_1.1/train_debiased-labels.lst')) as f:
            train_labels = f.readlines()
        train_labels = [int(item.strip()) for item in train_labels]
        for idx in range(len(datasets['train'])):
            datasets['train'][idx]['label'] = train_labels[idx]
    elif data_name == 'arc_easy':
        datasets['train'] = get_lines(os.path.join(data_dir, 'ARC-Easy-Train.jsonl'))
    elif data_name == 'arc_challenge':
        datasets['train'] = get_lines(os.path.join(data_dir, 'ARC-Challenge-Train.jsonl'))

    return datasets

#mode: baseline, dummy_context_vx, dummy_context_ensemble_vx
def load_data(data_name, data_path='./external', mask_token=None, tokenizer=None, data_config=None):
    datasets = get_datasets(data_name, data_path=data_path, data_config=data_config)
    if data_name in ['COPA']:
        datasets['test'] = load_test_labels('COPA', datasets['test'])
    
    kwargs = {}
    data_converter = data_converters[data_name]
    converted_datasets = {k:[data_converter(example, **kwargs) for example in v] for k,v in datasets.items()}
    processed_datasets = {k:[format_example(example, mask_token=mask_token, tokenizer=tokenizer) for example in v] for k,v in converted_datasets.items()}
    
    labels = {k:np.array([example['label'] for example in v]) for k,v in converted_datasets.items()}
    def compute_accuracy(split, preds, reduction="mean"):
        preds = np.array(preds)
        labels_for_split = labels[split]
        if reduction == "mean":
            acc = 100.0 * np.sum(preds==labels_for_split)/len(labels_for_split)
        elif reduction == "none":
            acc = (preds==labels_for_split)
        return acc
    context_lengths = {k: [len(tokenizer(item['context']).input_ids) for item in v] for k,v in processed_datasets.items()}
    return {
        'original': datasets,
        'converted': converted_datasets,
        'processed': processed_datasets,
        'acc_fn': compute_accuracy,
        'context_lengths': context_lengths
    }

    
