__author__ = 'Eunhwan Jude Park'
__email__ = 'jude.park.96@navercorp.com'

import json

from nltk.tokenize import word_tokenize
from tqdm import tqdm
from glob import glob


def tokenize_claims_and_save(raw_claim_path: str, target_path: str) -> None:
    with open(raw_claim_path, 'r', encoding='utf-8') as f:
        claim_set = [json.loads(i) for i in tqdm(f)]

    with open(target_path, 'a+', encoding='utf-8') as f:
        for claim in tqdm(claim_set):
            claim['mutated'] = ' '.join(word_tokenize(claim['mutated'])).strip()
            claim['original'] = ' '.join(word_tokenize(claim['original'])).strip()

            f.write(json.dumps(claim) + '\n')


if __name__ == '__main__':
    raw_claim_path = glob('../../rsc/raw_claims/*.jsonl')
    print(raw_claim_path)
    target_path = '../../rsc/raw_claims/nltk_tokenized/'

    for file_path in raw_claim_path:
        tokenize_claims_and_save(file_path, target_path + file_path.split('/')[-1])
