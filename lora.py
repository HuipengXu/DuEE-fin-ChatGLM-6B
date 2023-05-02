import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

from peft import get_peft_model, LoraConfig

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer_seq2seq import Seq2SeqTrainer

from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
from arguments import ModelArguments, DataTrainingArguments, LoraArguments

logger = logging.getLogger(__name__)

from dataset import Seq2SeqDataSet

DATA_PATH = "DuEE_fin"


# def set_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--train_path", default=f"{DATA_PATH}/train.json", type=str, help=""
#     )
#     parser.add_argument(
#         "--dev_path", default=f"{DATA_PATH}/dev.json", type=str, help=""
#     )
#     parser.add_argument("--model_dir", default="chatglm-6b", type=str, help="")
#     parser.add_argument("--num_train_epochs", default=5, type=int, help="")
#     parser.add_argument("--train_batch_size", default=2, type=int, help="")
#     parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="")
#     parser.add_argument("--output_dir", default="output", type=str, help="")
#     parser.add_argument("--log_steps", type=int, default=10, help="")
#     parser.add_argument("--max_len", type=int, default=2048, help="")
#     parser.add_argument("--max_src_len", type=int, default=1024, help="")
#     parser.add_argument("--local_rank", type=int, default=0, help="")
#     parser.add_argument("--lora_r", type=int, default=8, help="")
#     parser.add_argument("--prompt_text", type=str, default="", help="prompt length 796")
#     return parser.parse_args()


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, LoraArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, lora_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            lora_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        (
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {training_args.local_rank != -1}, 16-bits training: {training_args.fp16}"
        )
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)

    config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=['query_key_value'],
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=lora_args.task_type,
        inference_mode=False,
    )

    model = get_peft_model(model, config)

    if model_args.quantization_bit is not None:
        logger.info(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    model.print_trainable_parameters()

    train_dataset = Seq2SeqDataSet(
        data_args.train_file,
        tokenizer,
        data_args.max_source_length,
        data_args.max_target_length,
        data_args.prompt_text_path,
    )

    eval_dataset = Seq2SeqDataSet(
        data_args.validation_file,
        tokenizer,
        data_args.max_source_length,
        data_args.max_target_length,
        data_args.prompt_text_path,
    )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        save_prefixencoder=model_args.pre_seq_len is not None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            do_sample=True,
            top_p=0.7,
            max_length=512,
            temperature=0.95,
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     logger.info("*** Predict ***")

    #     predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=512, do_sample=True, top_p=0.7, temperature=0.95)
    #     metrics = predict_results.metrics
    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)

    #     if trainer.is_world_process_zero():
    #         if training_args.predict_with_generate:
    #             predictions = tokenizer.batch_decode(
    #                 predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             predictions = [pred.strip() for pred in predictions]
    #             labels = tokenizer.batch_decode(
    #                 predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             labels = [label.strip() for label in labels]
    #             output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    #             with open(output_prediction_file, "w", encoding="utf-8") as writer:
    #                 for p, l in zip(predictions, labels):
    #                     res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
    #                     writer.write(f"{res}\n")
    return results


if __name__ == "__main__":
    main()
