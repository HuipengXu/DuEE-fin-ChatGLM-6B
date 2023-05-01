import os
import argparse
from loguru import logger
from shutil import copy

import torch
import deepspeed
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from dataset import Seq2SeqDataSet, coll_fn

DATA_PATH = "DuEE_fin"


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", default=f"{DATA_PATH}/train.json", type=str, help=""
    )
    parser.add_argument(
        "--dev_path", default=f"{DATA_PATH}/dev.json", type=str, help=""
    )
    parser.add_argument("--model_dir", default="chatglm-6b", type=str, help="")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="")
    parser.add_argument("--train_batch_size", default=2, type=int, help="")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="")
    parser.add_argument("--output_dir", default="output", type=str, help="")
    parser.add_argument("--log_steps", type=int, default=10, help="")
    parser.add_argument("--max_len", type=int, default=2048, help="")
    parser.add_argument("--max_src_len", type=int, default=1024, help="")
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument("--lora_r", type=int, default=8, help="")
    parser.add_argument("--prompt_text", type=str, default="", help="prompt length 796")
    return parser.parse_args()


def main():
    args = set_args()

    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

    model = get_peft_model(model, config)
    model = model.half().cuda()

    conf = {
        "train_micro_batch_size_per_gpu": args.train_batch_size,
        "dev_micro_batch_size_per_gpu": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 5e-4,
            },
        },
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "steps_per_print": args.log_steps,
    }

    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    train_dataset = Seq2SeqDataSet(
        args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf["train_micro_batch_size_per_gpu"],
        sampler=RandomSampler(train_dataset),
        collate_fn=coll_fn,
        drop_last=True,
        num_workers=0,
    )
    dev_dataset = Seq2SeqDataSet(
        args.dev_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=conf["dev_micro_batch_size_per_gpu"],
        sampler=SequentialSampler(dev_dataset),
        collate_fn=coll_fn,
        drop_last=True,
        num_workers=0,
    )

    model_engine, _, _, _ = deepspeed.initialize(
        config=conf, model=model, model_parameters=model.parameters()
    )
    model_engine.train()
    global_step = 0
    for _ in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, labels=labels)
            loss = outputs[0]
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            if global_step % args.log_steps == 0:
                dev_iter = iter(dev_dataloader)
                model_engine.eval()
                dev_loss = 0.
                with torch.no_grad():
                    for step, batch in dev_iter:
                        input_ids = batch["input_ids"].cuda()
                        labels = batch["labels"].cuda()
                        outputs = model_engine.forward(input_ids=input_ids, labels=labels)
                        dev_loss += outputs[0].item()
                dev_loss /= len(dev_dataloader)
                print(f"train loss:{float(loss.item())}, global_step:{global_step}")
                print(f"dev loss:{dev_loss}, global_step:{global_step}")
                model_engine.train()
        save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        model_engine.save_pretrained(save_dir)
        copy(
            os.path.join(args.model_dir, "tokenizer_config.json"),
            os.path.join(save_dir, "tokenizer_config.json"),
        )
        copy(
            os.path.join(args.model_dir, "ice_text.model"),
            os.path.join(save_dir, "ice_text.model"),
        )


if __name__ == "__main__":
    main()
