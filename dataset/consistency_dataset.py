__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}'

import json
from abc import ABC
from functools import partial
from typing import List, Dict, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class ConsistencyDataset(Dataset, ABC):
    def __init__(self,
                 claim_set: List[Dict[str, str]],
                 tokenizer: AutoTokenizer) -> None:
        super(ConsistencyDataset, self).__init__()
        self.claim_set = claim_set
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.claim_set)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Any]]:
        raw_claim = self.claim_set[idx]
        original_claim, mutated_claim = raw_claim['original'], raw_claim['mutated']

        if raw_claim['verdict'] == 'REFUTES':
            label = 0
        elif raw_claim['verdict'] == 'SUPPORTS':
            label = 1
        else:
            label = 2

        single_input = self.tokenizer(original_claim, mutated_claim, return_tensors='pt')

        return {
            'input': single_input,
            'label': torch.LongTensor([label])
        }

    @staticmethod
    def collate_fn(data: Dict[str, Any], pad_token_id: int) -> Dict[str, torch.Tensor]:
        input_ids = [input['input']['input_ids'] for input in data]
        attn_masks = [input['input']['attention_mask'] for input in data]
        # token_type_ids = [input['input']['token_type_ids'] for input in data]
        labels = [input['label'] for input in data]

        input_id_max_len = max([i.shape[1] for i in input_ids])

        input_ids = [torch.cat((input_id.squeeze(dim=0), torch.LongTensor([pad_token_id] * (input_id_max_len - input_id.shape[1])))).unsqueeze(dim=0)
                     for input_id in input_ids]
        attn_masks = [torch.cat((attn_mask.squeeze(dim=0), torch.LongTensor([0] * (input_id_max_len - attn_mask.shape[1])))).unsqueeze(dim=0)
                      for attn_mask in attn_masks]
        # token_type_ids = [torch.cat((token_type_id.squeeze(dim=0), torch.LongTensor([0] * (input_id_max_len - token_type_id.shape[1])))).unsqueeze(dim=0)
        #                   for token_type_id in token_type_ids]

        input_ids = torch.cat(input_ids, dim=0).long()
        attn_masks = torch.cat(attn_masks, dim=0).long()
        # token_type_ids = torch.cat(token_type_ids, dim=0).long()

        assert input_ids.shape == attn_masks.shape
        # assert input_ids.shape == token_type_ids.shape

        return {
            'input_ids': input_ids,
            'attention_mask': attn_masks,
            # 'token_type_ids': token_type_ids,
            'labels': torch.cat(labels, dim=0)
        }


if __name__ == '__main__':
    with open('../rsc/raw_claims/test.jsonl', 'r') as f:
        test_dataset = [json.loads(i) for i in tqdm(f)]
        f.close()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = ConsistencyDataset(test_dataset, tokenizer)
    pad_token_id = 5
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=partial(ConsistencyDataset.collate_fn,
                                                                       pad_token_id=pad_token_id))

    for i in dataloader:
        print(i)
        break
