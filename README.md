# Single Headed Attention LSTM

For full details see the paper [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423).

Original implementation: [sha-rnn](https://github.com/Smerity/sha-rnn)


### SHALSTM for Question Answering

Uses the same format as [UnifiedQA](https://github.com/allenai/unifiedqa), please refer to their README for the details.


**`SHALSTMforQuestionAnswering`** API:


```python
import torch
from shalstm.qa import SHALSTMforQuestionAnswering
from shalstm.tokenizer import SHALSTMTokenizer

model = SHALSTMforQuestionAnswering.from_pretrained("unifiedqa_shalstm_base/model", device=torch.device("cuda"))
tokenizer = SHALSTMTokenizer.from_file("shalstm/tokenizer/tokenizer.json")

questions = [ "Why does the station want to get rid of their current singer? \\n  Frankie Ryan works as a page boy at a radio station located in Hollywood. His friend Jeff works in the same place, but as a porter. Their real dream is to perform as radio comedians on the air, with their own show. Unfortunately they haven't convinced anyone about their great sense of humor yet. When they try to help the station receptionist, Anne Mason, by setting up a false audition for the position as singer, they are almost fired for their antics. The station has financial problems related to their current moody singer Rita Wilson, and try to find a way to get rid of her. Their prayers are heard when Rita is shot and killed during a blackout when she is rehearsing for a broadcast. Police detectives Marty Phillips and Delaney arrive at the scene, and even though they haven't found the murder weapon, they start suspecting a wannabe cowboy singer, Tex Barton, who tried to slip out the back door after the shooting. He was in the audience when Rita was rehearsing before the blackout. Station producer Farrell is afraid of being suspected as well, since he had an argument with Rita not long before the shooting. He asks Frankie, who overheard the discussion, to not tell the police about it. As a sign of gratitude, Farrell promises to give Anne a real audition for the position as singer, which is empty since Rita is gone. Frankie soon finds the weapon used to shoot Rita, hidden in a ventilator duct. It turns out the gun belongs to Tex, and has been used in a prior shooting by a woman named Gladys Wharton. When Frankie and Jeff audition for a comedy spot on air (with Frankie in blackface as a disguise), the police come looking for Tex. Later, Tex is found murdered in the office of the station owner. Frankie and Jeff decide to do a little investigation of their own, and search Tex's room to see if they can find anything. The only thing of interest is a picture of Anne, suggesting that her real name is Gladys. Anne is therefore suspected of the murder and arrested by the police. However, a while later she makes bail and is released. Frankie discovers from a radio station in Cheyenne that the shooter Gladys Wharton was a blonde woman who fell for one of her superiors and left her husband - Tex. Since Anne is a true brunette, Frankie concludes that Rita could be Gladys instead of Anne. When all the station executives are gathered in one room by the police, one of them, Van Martin, pulls out a gun and confesses to both crimes. When Jeff enters the room unannounced, he accidentally knocks the gun out of Van's hand and the police arrest him." ]

# encode and predict
output = model.conditional_generate(*tokenizer.encode_for_qa(questions, direction="left")[:3], max_length=16)

# decode
decoded = tokenizer.decode_batch(output.t().tolist())
decoded = [ s.split("</s>")[0] for s in decoded ]

print(decoded[0])
print("Gold answer: She's moody, and they have financial problems.")
```