"""Supervised fine-tuning (SFT) on Nepali instructions after CPT.

CPT teaches the model the new Nepali token embeddings. This SFT stage teaches
the model to follow instructions in Nepali, using the wiseyak-sft-nepali
dataset (113K Alpaca-format instruction pairs).

Loading the CPT result is non-trivial: PeftModel.from_pretrained does not
restore the custom TrainableTokenEmbedding wrapper from cpt_train.py. The
trained new-token embeddings must be manually copied from the CPT checkpoint's
adapter_model.safetensors into the resized embedding matrix. See
load_cpt_model() for the full procedure.

Usage:
    python sft_train.py
    python sft_train.py --cpt_adapter output/cpt-qwen3-4b-nepali/adapter
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen3-4B"
CPT_ADAPTER = "output/cpt-qwen3-4b-nepali/adapter"
SFT_DATASET = "sharad461/wiseyak-sft-nepali"
OUTPUT_DIR = "output/sft-qwen3-4b-nepali"

SEQ_LEN = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 5e-5
MAX_STEPS = 1500
SAVE_STEPS = 500
LOGGING_STEPS = 10


def load_cpt_model(cpt_adapter_path: str):
    """Load the CPT'd model with manually restored new-token embeddings.

    The CPT adapter checkpoint contains:
    - LoRA weights for transformer layers
    - new_weight tensors for the appended embedding rows

    PeftModel.from_pretrained loads the LoRA weights but does not know about
    the TrainableTokenEmbedding wrapper, so the new_weight tensors must be
    extracted from the safetensors file and copied into the resized embedding
    matrix manually.
    """
    tokenizer = AutoTokenizer.from_pretrained(cpt_adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))

    # Load CPT LoRA weights
    model = PeftModel.from_pretrained(
        model, cpt_adapter_path, torch_dtype=torch.bfloat16, is_trainable=False,
    )

    # Manually restore the trained new-token embeddings from CPT checkpoint
    sd = load_file(os.path.join(cpt_adapter_path, "adapter_model.safetensors"))
    bm = model.get_base_model()
    with torch.no_grad():
        embed_key = "base_model.model.model.embed_tokens.new_weight"
        lm_head_key = "base_model.model.lm_head.trainable_embedding.new_weight"
        bm.get_input_embeddings().weight[base_vocab_size:len(tokenizer)].copy_(
            sd[embed_key].to(device=bm.get_input_embeddings().weight.device, dtype=torch.bfloat16))
        bm.get_output_embeddings().weight[base_vocab_size:len(tokenizer)].copy_(
            sd[lm_head_key].to(device=bm.get_output_embeddings().weight.device, dtype=torch.bfloat16))

    # Merge CPT LoRA into base weights before applying a fresh SFT LoRA
    model = model.merge_and_unload()

    return model, tokenizer, base_vocab_size


def format_instruction(example):
    """Format an Alpaca-style row into a plain instruction prompt."""
    if example.get("input") and example["input"].strip():
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=SEQ_LEN,
        padding=False,
        add_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpt_adapter", type=str, default=CPT_ADAPTER)
    args = parser.parse_args()

    print("=" * 60)
    print("SFT: Nepali Instruction Tuning")
    print("=" * 60)

    # --- Load CPT model with restored embeddings ---

    print("\n1. Loading CPT'd model...")
    model, tokenizer, base_vocab_size = load_cpt_model(args.cpt_adapter)
    print(f"   Vocab: {len(tokenizer)}, base: {base_vocab_size}")

    # --- Fresh LoRA for SFT ---

    print("\n2. Setting up SFT LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---

    print(f"\n3. Loading SFT dataset: {SFT_DATASET}")
    dataset = load_dataset(SFT_DATASET, split="train")
    print(f"   Samples: {len(dataset)}")

    print("\n4. Formatting and tokenizing...")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    print(f"   Tokenized samples: {len(dataset)}")

    # --- Training ---

    print("\n5. Starting SFT...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
        lr_scheduler_type="cosine",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # --- Save ---

    print("\n6. Saving SFT adapter...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
    print(f"   Saved to {OUTPUT_DIR}/adapter")
    print("\nSFT complete!")


if __name__ == "__main__":
    main()
