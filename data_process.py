import os
import json
import random
import numpy as np
from loguru import logger
from tqdm import tqdm


DATA_PATH = "DuEE_fin"


def format_data(path):
    examples = []
    num_bad = 0
    text_length = []
    answer_length = []
    with open(path, "r", encoding="utf8") as f:
        for line in tqdm(f):
            line = line.strip()
            example = json.loads(line)
            answer = []
            try:
                for event in example["event_list"]:
                    event_type = f'{event["trigger"]}_{event["event_type"]}'
                    roles = [
                        f'{role["argument"]}_{role["role"]}'
                        for role in event["arguments"]
                    ]
                    roles = ",".join(roles)
                    new_event = f"{event_type}#{roles}"
                    answer.append(new_event)
                answer = "\n".join(answer)
                new_example = {"text": example["text"], "answer": answer}
                text_length.append(len(example["text"]))
                answer_length.append(len(new_example["answer"]))
                examples.append(new_example)
            except KeyError:
                logger.info(f'Current line decode error: {example["id"]}')
                num_bad += 1
    logger.info(f"Total have {len(examples)} examples")
    logger.info(f"Total have {num_bad} bad examples")
    logger.info(f"{np.percentile(text_length, 95)}")
    logger.info(f"{np.percentile(answer_length, 95)}")
    return examples


def main():
    train_data_path = os.path.join(DATA_PATH, f"{DATA_PATH.lower()}_train.json")
    dev_data_path = os.path.join(DATA_PATH, f"{DATA_PATH.lower()}_dev.json")

    logger.info("Starting processing raw train data")
    train_examples = format_data(train_data_path)
    logger.info("Starting processing raw dev data")
    dev_examples = format_data(dev_data_path)
    formatted_data = train_examples + dev_examples
    logger.info(f"Total have {len(formatted_data)} available train data")

    save_train_path = f"{DATA_PATH}/train.json"
    save_dev_path = f"{DATA_PATH}/dev.json"
    random.shuffle(formatted_data)
    num_dev = int(0.15 * len(formatted_data))
    formatted_train_data = formatted_data[:-num_dev]
    formatted_dev_data = formatted_data[-num_dev:]
    save_data(formatted_train_data, save_train_path)
    save_data(formatted_dev_data, save_dev_path)

    # 生成 prompt
    logger.info("Starting generate prefix prompt")
    prompt = "请按照如下事件类型以及对应的角色抽取文本中的事件：\n"
    with open(
        f"{DATA_PATH}/{DATA_PATH.lower()}_event_schema.json", "r", encoding="utf8"
    ) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            event = json.loads(line)
            sub_prompt = f'事件类型{i}：{event["event_type"]}，事件角色：'
            roles = []
            for role in event["role_list"]:
                if "enum_items" in role:
                    role_desc = f'{role["role"]}（值不来自需要抽取的文本只能来自枚举值：{"、".join(role["enum_items"])}）'
                else:
                    role_desc = role["role"]
                roles.append(role_desc)
            sub_prompt += "，".join(roles) + "\n"
            prompt += sub_prompt
    with open(f"{DATA_PATH}/prompt.txt", "w", encoding="utf8") as f:
        prompt = (
            prompt
            + '抽取的结果使用"#"连接事件类型和事件角色，事件类型的触发词和类型用"_"连接，事件角色值和事件角色也使用"_"连接，多个事件之间使用\\n连接。\n待抽取的文本：'
        )
        logger.info(f"Prompt length: {len(prompt)}")
        f.write(prompt)


def save_data(formatted_data, save_path):
    with open(save_path, "w", encoding="utf8") as f:
        for example in tqdm(
            formatted_data,
            desc="Dump formatted data",
            total=len(formatted_data),
        ):
            example = json.dumps(example, ensure_ascii=False)
            f.write(f"{example}\n")


if __name__ == "__main__":
    main()
