from tokenizers.implementations import BaseTokenizer
from tokenizers import Tokenizer

import torch

class SHALSTMTokenizer(BaseTokenizer):

    @staticmethod
    def from_file(path):
        return SHALSTMTokenizer(Tokenizer.from_file(path))

    def encode_for_qa(self, data, targets=None, direction=None):

        assert targets is not None or direction is not None, "You should specify one of the arguments (direction can be 'right' or 'left')."

        if targets is not None:
            direction = 'left'

        padding = self.padding
        org_direction = padding["direction"]
        
        # set padding direction
        padding["direction"] = direction
        self.enable_padding(**padding)

        encodings = self.encode_batch(data, add_special_tokens=True)

        # reset padding direction
        padding["direction"] = org_direction
        self.enable_padding(**padding)

        input = torch.stack([torch.tensor(x.ids) for x in encodings], dim=1)
        attn_mask = torch.stack([torch.tensor(x.attention_mask) for x in encodings], dim=1)
        type_ids = torch.stack([torch.tensor(x.type_ids) for x in encodings], dim=1)
        
        input_length = input.size(0)
        if targets is not None:
            tinput, tattn_mask, ttype_ids, t_length = self.encode_for_qa(targets, direction='right')
            input = torch.cat([input, tinput[1:]], dim=0)
            attn_mask = torch.cat([attn_mask, tattn_mask[1:]], dim=0)
            type_ids = torch.cat([torch.zeros_like(type_ids), tattn_mask[1:]], dim=0)

        return input, attn_mask, type_ids, input_length


if __name__ == '__main__':
    tokenizer = SHALSTMTokenizer.from_file("tokenizer/tokenizer.json")

    questions = [
               "question one ?",
               "another question to check ?",
            ]

    answers = [
               "this answer for question one ?",
               "This is another answer for another question ?",
            ]

    input, attn_mask, type_ids, input_length = tokenizer.encode_for_qa(questions, answers)
