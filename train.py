import argparse
import json
import logging
import math
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

from dataset.consistency_dataset import ConsistencyDataset

sys.path.append(os.getcwd() + "/../")  # noqa: E402
from utils.general_utils import print_args
from utils.gpu_utils import set_seed, init_gpu_params
from utils.checkpoint_utils import save_model_state_dict, write_log

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def sanity_checks(params):
    if os.path.isdir(params.save_checkpoints_dir):
        assert not os.listdir(params.save_checkpoints_dir), "checkpoint directory must be empty"
    else:
        os.makedirs(params.save_checkpoints_dir, exist_ok=True)


def _get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_ids", type=str, default="3",
                        help="comma separated list of devices ids in single node")

    parser.add_argument("--seed", type=int, default=203)
    parser.add_argument('--hf_model_name', type=str)

    parser.add_argument("--save_checkpoints_dir", type=str, default="checkpoints_for_baseline/")
    parser.add_argument("--save_checkpoints_steps", type=int, default=10000)
    parser.add_argument("--log_step_count_steps", type=int, default=10)

    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.06)

    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--classifier_dropout", type=float, default=0.0)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")

    return parser


def evaluate(dataloader, tokenizer, model) -> float:
    logger.info('eval start')
    model.eval()

    y_pred, y_true = [], []
    for batch in tqdm(dataloader, desc="Evaluate"):
        batch = {k: v.to('cuda') for k, v in batch.items()}

        with torch.no_grad():
            output = model(batch['input_ids'])
            logits = output['logits']

        prediction = logits.argmax(-1).cpu().detach().tolist()
        label = batch['labels'].cpu().detach().tolist()

        y_pred.extend(prediction)
        y_true.extend(label)

    report = classification_report(np.array(y_true), np.array(y_pred),
                                   target_names=['REFUTES', 'SUPPORTS', 'NOT_ENOUGH_INFO'], output_dict=True)

    return report


def main():
    args = _get_parser().parse_args()

    set_seed(args)
    sanity_checks(args)
    init_gpu_params(args)

    with open('./rsc/raw_claims/train.jsonl', 'r') as f:
        train_dataset = [json.loads(i) for i in tqdm(f)]

    with open('./rsc/raw_claims/test.jsonl', 'r') as f:
        test_dataset = [json.loads(i) for i in tqdm(f)]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)

    train_dataset, test_dataset = ConsistencyDataset(train_dataset, tokenizer), \
                                  ConsistencyDataset(test_dataset, tokenizer)

    train_sampler = RandomSampler(train_dataset) if not args.multi_gpu else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.per_gpu_train_batch_size,
                                  pin_memory=True,
                                  num_workers=8,
                                  collate_fn=partial(ConsistencyDataset.collate_fn, pad_token_id=tokenizer.pad_token_id))

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.per_gpu_train_batch_size,
                                 pin_memory=True,
                                 num_workers=8,
                                 collate_fn=partial(ConsistencyDataset.collate_fn, pad_token_id=tokenizer.pad_token_id))

    if args.hf_model_name == 'bert-base-cased':
        model = BertForSequenceClassification.from_pretrained(args.hf_model_name, num_labels=3)
    elif args.hf_model_name == 'roberta-base':
        model = RobertaForSequenceClassification.from_pretrained(args.hf_model_name, num_labels=3)
    else:
        raise ValueError('invalid hf_model_name')

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_epsilon)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = math.ceil(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    model.zero_grad()
    model.cuda()

    if args.multi_gpu:
        model = DistributedDataParallel(model,
                                        device_ids=[args.device_ids[args.local_rank]],
                                        output_device=args.device_ids[args.local_rank],
                                        find_unused_parameters=True)

    if args.is_master:
        print_args(args)
        configuration = vars(args)
        save_configuration_path = os.path.join(args.save_checkpoints_dir, f"configuration.json")
        with open(save_configuration_path, "w") as fp:
            json.dump(configuration, fp, indent=2, ensure_ascii=False)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.per_gpu_train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.multi_gpu else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_steps = 0
    best_f1_score = 0
    for epoch in range(args.num_train_epochs):
        if args.multi_gpu:
            train_sampler.set_epoch(epoch * 1000)

        model.train()

        iter_loss = 0

        iter_bar = tqdm(train_dataloader, desc="Iter", disable=not args.is_master)
        for step, batch in enumerate(iter_bar):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            output = model(batch['input_ids'], labels=batch['labels'])

            loss = output['loss']

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                global_steps += 1

                if global_steps % args.log_step_count_steps == 0 and args.is_master:
                    write_log(args.save_checkpoints_dir, "log_step.txt", iter_bar)

                if global_steps % args.save_checkpoints_steps == 0 and args.is_master:
                    eval_result = evaluate(test_dataloader, tokenizer, model)
                    f1 = eval_result['weighted avg']['f1-score']
                    with open(os.path.join(args.save_checkpoints_dir, "log_eval.jsonl"), "a+") as fp:
                        fp.write(
                            json.dumps(dict(epoch=epoch, global_steps=global_steps, evaluation=eval_result)) + "\n")

                    if best_f1_score < f1:
                        best_f1_score = f1
                        save_model_state_dict(args.save_checkpoints_dir, f"best_model.pth", model)

                    model.train()

            iter_loss += loss.item()
            iter_bar.set_postfix({
                "epoch": f"{epoch}",
                "global_steps": f"{global_steps}",
                "learning_rate": f"{scheduler.get_last_lr()[0]:.10f}",
                "rolling_loss": f"{iter_loss / (step + 1) * args.gradient_accumulation_steps:.5f}",
                "last_loss": f"{loss.item() * args.gradient_accumulation_steps:.5f}"
            })

        if args.is_master:
            write_log(args.save_checkpoints_dir, "log_iter.txt", iter_bar)

            eval_result = evaluate(test_dataloader, tokenizer, model)
            f1 = eval_result['weighted avg']['f1-score']
            with open(os.path.join(args.save_checkpoints_dir, "log_eval.jsonl"), "a+") as fp:
                fp.write(json.dumps(dict(epoch=epoch, global_steps=global_steps, evaluation=eval_result)) + "\n")

            if best_f1_score < f1:
                best_f1_score = f1
                save_model_state_dict(args.save_checkpoints_dir, f"best_model.pth", model)


if __name__ == "__main__":
    main()
