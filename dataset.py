import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Seq2SeqDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):

        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(prompt_text)

                if len(src_tokens) > max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[: max_src_len - len(prompt_tokens)]

                tgt_tokens = tokenizer.tokenize(sample["answer"])
                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                tokens = (
                    prompt_tokens
                    + src_tokens
                    + ["[gMASK]", "<sop>"]
                    + tgt_tokens
                    + ["<eop>"]
                )
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1 :]

                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

                self.all_data.append(
                    {
                        "text": sample["text"],
                        "answer": sample["answer"],
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                )

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        return self.all_data[item]


def coll_fn(batch):
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    return {
        "input_ids": pad_sequence(
            input_ids_list, batch_first=True, padding_value=3
        ),
        "labels": pad_sequence(labels_list, batch_first=True, padding_value=3),
    }
