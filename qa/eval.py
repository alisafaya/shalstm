
import random
import json
import argparse

import numpy as np
import torch.nn as nn
import torch

from tokenizer import SHALSTMTokenizer
from qa.model import SHALSTMforQuestionAnswering 

from datasets import load_metric


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_dataset_with_ids(datadir, iddir, tokenizer, batch_size=32):

    with open(datadir) as fi:
        lines = fi.read().splitlines()
        questions, answers = tuple(np.array(x) for x in zip(*[ l.split("\t") for l in lines ]))

    with open(iddir) as fi:
        lines = fi.read().splitlines()
        gold_answers = np.array([ json.loads(s) for s in lines ], dtype=object)

    q_lens = np.array([ len(tokenizer.encode(x)) for x in questions ])
    indices = np.argsort(q_lens)

    questions = list(chunks(list(questions[indices]), batch_size))
    answers = list(chunks(list(answers[indices]), batch_size))
    gold_answers = list(gold_answers[indices])

    batches = []
    for q, a in zip(questions, answers):
        batches.append(tokenizer.encode_for_qa(q, a))

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
                output, h, hidden, mems = model(input, attn_mask, type_ids)

            output = torch.argmax(output, dim=-1)
            predictions += output[input_length - 1:].t().cpu().tolist()

    return predictions


def main(args):

    device = torch.device(args.device)
    use_amp = args.no_amp

    model = SHALSTMforQuestionAnswering.from_pretrained(args.model, device=torch.device(args.device))
    tokenizer = SHALSTMTokenizer.from_file(args.tokenizer)

    val_data, val_gold = load_dataset_with_ids(args.val_dir, args.val_ans, tokenizer, batch_size=args.batch_size)
    predictions = get_predictions(model, val_data, use_amp=use_amp)

    tokenizer.decode_batch(predictions)

    decoded = tokenizer.decode_batch(predictions)
    decoded = [ s.split("</s>")[0] for s in decoded ]

    metric = load_metric("squad")

    predictions = [{'prediction_text':x , "id": str(y) } for y, x in enumerate(decoded, start=1) ]
    references = [{'answers': {'answer_start': [1]*len(x), 'text': x  }, 'id': str(y) } for y, x in enumerate(val_gold, start=1) ]
    print(metric.compute(predictions=predictions, references=references))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--val_ans", type=str, required=True)

    args = parser.parse_args()

    main(args)