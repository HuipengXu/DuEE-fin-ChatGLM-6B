import json
from torch.utils.data import Dataset


class Seq2SeqDataSet(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_source_length,
        max_target_length,
        prompt_text_path,
    ):

        with open(prompt_text_path, "r", encoding="utf8") as f:
            prompt_text = f.read().strip()

        max_length = max_source_length + max_target_length + 3
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(prompt_text)

                if len(src_tokens) > max_source_length - len(prompt_tokens):
                    src_tokens = src_tokens[: max_source_length - len(prompt_tokens)]

                tgt_tokens = tokenizer.tokenize(sample["answer"])
                if len(tgt_tokens) > max_target_length:
                    tgt_tokens = tgt_tokens[:max_target_length]
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

                pad_len = max_length - len(input_ids)
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
