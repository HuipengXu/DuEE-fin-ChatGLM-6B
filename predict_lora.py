import logging
import time
import os
import sys
import json
from tqdm import tqdm


import torch
from peft import PeftModel
from transformers import HfArgumentParser

from tokenization_chatglm import ChatGLMTokenizer
from arguments import ModelArguments, DataTrainingArguments
from modeling_chatglm import ChatGLMForConditionalGeneration

logger = logging.getLogger(__name__)

DATA_PATH = "DuEE_fin"


def main():
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (data_args, model_args) = parser.parse_args_into_dataclasses()

    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)
    model.eval()
    model = PeftModel.from_pretrained(
        model, model_args.peft_model_name_or_path, torch_dtype=torch.float32
    )
    model.half().to("cuda:0")
    # model.eval()
    save_data = []
    f1 = 0.0
    with open(data_args.rompt_text_path, "r", encoding="utf8") as f:
        prompt_text = f.read().strip()
    s_time = time.time()
    with open(data_args.test_file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Predicting"):
            with torch.no_grad():
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(prompt_text)

                if len(src_tokens) > data_args.max_source_length - len(prompt_tokens):
                    src_tokens = src_tokens[
                        : data_args.max_source_length - len(prompt_tokens)
                    ]

                tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_ids = torch.tensor([input_ids]).to("cuda:0")
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": data_args.max_target_length,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 1,
                }
                response = model.generate(input_ids, **generation_kwargs)
                res = []
                for i_r in range(generation_kwargs["num_return_sequences"]):
                    outputs = response.tolist()[i_r][input_ids.shape[1] :]
                    r = tokenizer.decode(outputs).replace("<eop>", "")
                    res.append(r)
                real_res = {
                    event.split("#")[0]: event.split("#")[1].split(",")
                    for event in sample["answer"].split("\n")
                }
                pre_res = {
                    rr.split("#")[0]: rr.split("#")[1].split(",")
                    for rr in res[0].split("\n")
                    if len(rr.split("#")) == 2 and rr.split("#")[0] in real_res
                }
                # TODO 先计算事件类型的prf，然后计算事件类型匹配正确情况下论元的prf
                    
                    
                same_res = set(pre_res) & set(real_res)
                p = len(same_res) / len(set(pre_res)) if set(pre_res) else 0.0
                r = len(same_res) / len(set(real_res))
                f = 2 * p * r / (p + r) if (p + r) != 0.0 else 0.0
                f1 += f
                save_data.append(
                    {
                        "text": sample["text"],
                        "ori_answer": sample["answer"],
                        "gen_answer": res[0],
                        "f1": f,
                    }
                )

    e_time = time.time()
    print(f"总耗时：{e_time - s_time}s")
    print(f1 / 50)
    save_path = os.path.join(os.path.dirname(data_args.train_file), "lora_answer.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
