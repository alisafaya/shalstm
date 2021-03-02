
import random
import json
import argparse
import string
import re

import numpy as np
import torch.nn as nn
import torch

from ..tokenizer import SHALSTMTokenizer
from .model import SHALSTMforQuestionAnswering 

from datasets import load_metric
from sklearn.metrics import classification_report, accuracy_score, f1_score
from nltk import edit_distance

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_dataset_with_ids(datadir, tokenizer, ans_dir="", batch_size=32, mc=False):

    with open(datadir) as fi:
        lines = fi.read().splitlines()
        questions, answers = tuple(np.array(x) for x in zip(*[ l.split("\t") for l in lines ]))
        
        if mc:
            choice_list = np.array([ re.split('\s+\([A-I]\)\s+', q.split("\\n")[1]) for q in questions ], dtype=object)

    if ans_dir == "":
        gold_answers = np.array(answers)
    else:
        with open(ans_dir) as fi:
            lines = fi.read().splitlines()
            gold_answers = np.array([ json.loads(s) for s in lines ], dtype=object)

    q_lens = np.array([ len(tokenizer.encode(x)) for x in questions ])
    indices = np.argsort(q_lens)

    questions = list(chunks(list(questions[indices]), batch_size))
    answers = list(chunks(list(answers[indices]), batch_size))
    gold_answers = list(gold_answers[indices])
    
    if mc:
        choice_list = list(choice_list[indices])

    batches = []
    for q, a in zip(questions, answers):
        batches.append(tokenizer.encode_for_qa(q, a))

    if mc:
        return batches, gold_answers, choice_list
        
    return batches, gold_answers



def get_predictions(
        model,
        val_data,
        use_amp=True
        ):

    model.eval()
    predictions = []
    with torch.no_grad():
        for input, attn_mask, type_ids, input_length in val_data:
            # calculate loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model.conditional_generate(input[:input_length], attn_mask[:input_length], type_ids[:input_length])

            predictions += output.t().cpu().tolist()

    return predictions

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_set(s):
    return list(map(normalize_answer, s))

def get_nearest_choice(args):
    pred, possible_choices = args
    dists = [ edit_distance(pred, c) for c in possible_choices ]
    return possible_choices[dists.index(min(dists))]

def main(args):

    device = torch.device(args.device)
    use_amp = args.no_amp

    model = SHALSTMforQuestionAnswering.from_pretrained(args.model, device=torch.device(args.device))
    tokenizer = SHALSTMTokenizer.from_file(args.tokenizer)

    if args.metric == "mc":
        val_data, val_gold, choice_list = load_dataset_with_ids(args.val_dir, tokenizer, ans_dir=args.val_ans, batch_size=args.batch_size, mc=True)
        choice_list = list(map(normalize_set, choice_list))
    else:
        val_data, val_gold = load_dataset_with_ids(args.val_dir, tokenizer, ans_dir=args.val_ans, batch_size=args.batch_size)
    
    predictions = get_predictions(model, val_data, use_amp=use_amp)

    decoded = tokenizer.decode_batch(predictions)
    decoded = [ s.split("</s>")[0] for s in decoded ]

    if args.metric == "squad":
        ## squad v1.1
        
        metric = load_metric("squad")
        
        predictions = [{'prediction_text':x , "id": str(y) } for y, x in enumerate(decoded, start=1) ]
        references = [{'answers': {'answer_start': [1]*len(x), 'text': x  }, 'id': str(y) } for y, x in enumerate(val_gold, start=1) ]
        print(metric.compute(predictions=predictions, references=references))
    
    elif args.metric == "squad_v2":
        ## squad v2
        
        metric = load_metric("squad_v2")
        predictions = [{'prediction_text': x, "id": str(y), 'no_answer_probability': 0.9 if x.startswith("<No") else 0.1 } for y, x in enumerate(decoded, start=1) ]
        references = [{'answers': {'answer_start': [1]*len(x), 'text': "" if "<No Answer>" in x else x }, 'id': str(y)} for y, x in enumerate(val_gold, start=1) ]

        print(metric.compute(predictions=predictions, references=references, no_answer_threshold=0.5))
        
    elif args.metric == "rouge":
        
        ## NarrativeQA, 
        metric = load_metric("rouge")
        results = metric.compute(predictions=decoded, references=val_gold)
        print(results["rougeL"].mid)

    elif args.metric == "mc":

        decoded = list(map(get_nearest_choice, zip(decoded, choice_list)))
        predictions = normalize_set(decoded)
        references = normalize_set(val_gold)
        print(accuracy_score(references, predictions))

        with open("pred.json", "w") as fo:
            fo.write(json.dumps(list(zip(choice_list, references, predictions)), ensure_ascii=False, indent=2))
        
    elif args.metric == "accuracy":
        ## BoolQ

        predictions = decoded
        references = val_gold
        print(accuracy_score(references, predictions))


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--val_ans", type=str, default="")
    parser.add_argument("--metric", type=str, default="")

    args = parser.parse_args()

    print()
    print(args)
    print()

    main(args)
