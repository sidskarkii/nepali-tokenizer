"""Continued pretraining (CPT) for Qwen3-4B with an extended Nepali tokenizer.

Tokenizer extension adds Nepali token IDs to the vocabulary, but the model has
no trained embeddings for those IDs yet. This script teaches them via LoRA CPT
with a split learning rate: standard LoRA lr for transformer layers, and a
separate (higher) lr for the new embedding rows.

The key design decision is TrainableTokenEmbedding: instead of training the
full 166K embedding matrix (1.4B params via PEFT modules_to_save), we freeze
the original 151K base rows and only train the 15K appended rows (38M params).
This dropped step time from ~70s to ~20s on an A40.

Streams Nepali text from CulturaX and English from FineWeb (20% mix to reduce
catastrophic forgetting).

Usage:
    python cpt_train.py [--resume_from_checkpoint PATH]
"""

import argparse
import os
import random
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen3-4B"
EXTENDED_TOKENIZER = "tokenizers/extended/qwen3-nepali"
OUTPUT_DIR = "output/cpt-qwen3-4b-nepali"

SEQ_LEN = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
NEW_TOKEN_LR = 5e-4
WARMUP_RATIO = 0.03
MAX_STEPS = 3000
SAVE_STEPS = 500
LOGGING_STEPS = 10
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

TARGET_NEPALI_CHARS = 800_000_000
TARGET_ENGLISH_CHARS = 200_000_000


class TrainableTokenEmbedding(nn.Module):
    """Wraps an embedding layer so only the appended token rows are trainable.

    PEFT's modules_to_save approach trains the FULL embedding matrix, which for
    a 166K vocab means 1.4B params and ~70s/step. This wrapper freezes the
    original base rows and exposes only the new rows as trainable parameters,
    cutting to 38M params and ~20s/step.

    The .weight property concatenates base + new rows so PEFT's save logic
    captures the new_weight tensor in the adapter checkpoint.
    """

    def __init__(self, base_embed: nn.Module, base_vocab_size: int, new_weight_init: torch.Tensor):
        super().__init__()
        self.base_embed = base_embed
        self.base_vocab_size = base_vocab_size
        self.new_weight = nn.Parameter(new_weight_init.clone().detach())
        self.num_embeddings = base_vocab_size + self.new_weight.shape[0]
        self.embedding_dim = self.new_weight.shape[1]
        self.padding_idx = getattr(base_embed, "padding_idx", None)
        self.max_norm = getattr(base_embed, "max_norm", None)
        self.norm_type = getattr(base_embed, "norm_type", 2.0)
        self.scale_grad_by_freq = getattr(base_embed, "scale_grad_by_freq", False)
        self.sparse = getattr(base_embed, "sparse", False)

        for param in self.base_embed.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return torch.cat(
            [self.base_embed.weight[:self.base_vocab_size], self.new_weight],
            dim=0,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.base_embed(input_ids)
        mask = input_ids >= self.base_vocab_size
        if not mask.any():
            return outputs

        outputs = outputs.clone()
        new_token_ids = input_ids[mask] - self.base_vocab_size
        outputs[mask] = self.new_weight[new_token_ids]
        return outputs


class TrainableTokenLMHead(nn.Module):
    """Matching wrapper for the output projection (lm_head).

    When tie_word_embeddings is True, shares new_weight with the input
    embedding wrapper. Otherwise maintains its own trainable rows.
    """

    def __init__(
        self,
        base_lm_head: nn.Module,
        base_vocab_size: int,
        trainable_embedding: TrainableTokenEmbedding | None = None,
        new_weight_init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base_lm_head = base_lm_head
        self.base_vocab_size = base_vocab_size
        self.trainable_embedding = trainable_embedding

        if trainable_embedding is None:
            if new_weight_init is None:
                raise ValueError("new_weight_init is required when lm_head is not tied")
            self.new_weight = nn.Parameter(new_weight_init.clone().detach())
        else:
            self.new_weight = None

        self.in_features = base_lm_head.in_features
        self.out_features = base_vocab_size + self.get_new_weight().shape[0]

        for param in self.base_lm_head.parameters():
            param.requires_grad = False

    def get_new_weight(self) -> torch.Tensor:
        if self.trainable_embedding is not None:
            return self.trainable_embedding.new_weight
        return self.new_weight

    @property
    def weight(self):
        return torch.cat(
            [self.base_lm_head.weight[:self.base_vocab_size], self.get_new_weight()],
            dim=0,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        old_logits = F.linear(
            hidden_states,
            self.base_lm_head.weight[:self.base_vocab_size],
            None if self.base_lm_head.bias is None else self.base_lm_head.bias[:self.base_vocab_size],
        )
        new_logits = F.linear(hidden_states, self.get_new_weight(), bias=None)
        return torch.cat([old_logits, new_logits], dim=-1)


def is_good_nepali(text, min_ratio=0.5):
    if not text or len(text) < 50:
        return False
    devanagari = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
    return devanagari / len(text) > min_ratio


def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = ''.join(
        c for c in text
        if not unicodedata.category(c).startswith('C')
        or c in ('\n', '\t', '‍', '‌')
    )
    return text.strip()


def load_nepali_data():
    """Load Nepali text from HuggingFace sources."""
    print("   Loading CulturaX Nepali...")
    texts = []
    char_count = 0

    ds = load_dataset("uonlp/CulturaX", "ne", split="train", streaming=True)
    for example in ds:
        text = clean_text(example["text"])
        if is_good_nepali(text):
            texts.append(text)
            char_count += len(text)
            if char_count >= TARGET_NEPALI_CHARS:
                break
            if len(texts) % 50000 == 0:
                print(f"      {len(texts)} docs, {char_count/1e6:.0f}M chars")

    print(f"   Nepali: {len(texts)} docs, {char_count/1e6:.0f}M chars")
    return texts


def load_english_data():
    """Load English text for catastrophic forgetting prevention."""
    print("   Loading English (FineWeb sample)...")
    texts = []
    char_count = 0

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    for example in ds:
        text = example["text"].strip()
        if len(text) > 100:
            texts.append(text)
            char_count += len(text)
            if char_count >= TARGET_ENGLISH_CHARS:
                break

    print(f"   English: {len(texts)} docs, {char_count/1e6:.0f}M chars")
    return texts


def init_new_embeddings(model, tokenizer, base_vocab_size):
    """Initialize appended rows from the base-token decomposition.

    New Nepali tokens are not random-initialized. Each new token is first encoded
    by the original base tokenizer, then initialized from the mean embedding of
    those constituent base tokens. This gives CPT a sensible starting point.
    """
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    embed_in = model.get_input_embeddings()
    embed_out = model.get_output_embeddings()

    with torch.no_grad():
        for idx in range(base_vocab_size, len(tokenizer)):
            token = tokenizer.convert_ids_to_tokens(idx)
            sub_ids = base_tokenizer.encode(token, add_special_tokens=False)
            if sub_ids:
                mean_embed = embed_in.weight[sub_ids].mean(dim=0)
                embed_in.weight[idx] = mean_embed
                if embed_out is not None:
                    mean_out = embed_out.weight[sub_ids].mean(dim=0)
                    embed_out.weight[idx] = mean_out

    print(f"   Initialized {len(tokenizer) - base_vocab_size} new embeddings via mean-of-subword")


def tokenize_and_pack(examples, tokenizer):
    """Tokenize documents and pack them into fixed-length training blocks."""
    tokenized = tokenizer(
        examples["text"],
        truncation=False,
        add_special_tokens=False,
    )

    packed_ids = []
    packed_attention = []
    buffer = []

    for ids in tokenized["input_ids"]:
        buffer.extend(ids)
        while len(buffer) >= SEQ_LEN:
            packed_ids.append(buffer[:SEQ_LEN])
            packed_attention.append([1] * SEQ_LEN)
            buffer = buffer[SEQ_LEN:]

    return {"input_ids": packed_ids, "attention_mask": packed_attention}


def enable_new_token_training(model, base_vocab_size):
    """Replace full embedding/lm_head training with appended-row training only."""
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()

    new_input_init = input_embed.weight[base_vocab_size:].detach()
    trainable_input = TrainableTokenEmbedding(input_embed, base_vocab_size, new_input_init)
    model.set_input_embeddings(trainable_input)

    if output_embed is not None:
        if getattr(model.config, "tie_word_embeddings", False):
            trainable_output = TrainableTokenLMHead(
                output_embed,
                base_vocab_size,
                trainable_embedding=trainable_input,
            )
        else:
            new_output_init = output_embed.weight[base_vocab_size:].detach()
            trainable_output = TrainableTokenLMHead(
                output_embed,
                base_vocab_size,
                new_weight_init=new_output_init,
            )
        model.set_output_embeddings(trainable_output)

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_corpus", type=str, default=None,
                        help="Path to a local Nepali text file (one doc per line). "
                             "Skips CulturaX download.")
    args = parser.parse_args()

    print("=" * 60)
    print("CPT: Qwen3-4B + Nepali Tokenizer Extension")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # --- Tokenizer and model setup ---

    print("\n1. Loading extended tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(EXTENDED_TOKENIZER, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {len(tokenizer)}")

    print("\n2. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"   Base vocab: {base_vocab_size}, Extended vocab: {len(tokenizer)}")

    print("\n3. Resizing embeddings...")
    model.resize_token_embeddings(len(tokenizer))

    print("\n4. Initializing new embeddings (mean-of-subword)...")
    init_new_embeddings(model, tokenizer, base_vocab_size)

    # --- LoRA + selective embedding training ---

    print("\n5. Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # get_peft_model freezes everything except LoRA params. We need to
    # re-enable gradients on the new embedding rows after wrapping.
    print("\n6. Restricting embedding training to new token rows only...")
    model = enable_new_token_training(model, base_vocab_size)
    for n, p in model.named_parameters():
        if "new_weight" in n:
            p.requires_grad = True
    model.print_trainable_parameters()

    # --- Data loading ---

    print("\n7. Loading training data...")
    if args.local_corpus:
        print(f"   Using local corpus: {args.local_corpus}")
        nepali_texts = []
        with open(args.local_corpus, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    nepali_texts.append(line)
    else:
        nepali_texts = load_nepali_data()
    english_texts = load_english_data()
    all_texts = nepali_texts + english_texts
    print(f"   Total: {len(all_texts)} docs")

    random.seed(42)
    random.shuffle(all_texts)
    dataset = Dataset.from_dict({"text": all_texts})

    print("\n8. Tokenizing and packing...")
    dataset = dataset.map(
        lambda x: tokenize_and_pack(x, tokenizer),
        batched=True,
        batch_size=10000,
        remove_columns=["text"],
        num_proc=4,
    )
    print(f"   Packed sequences ({SEQ_LEN} tokens each): {len(dataset)}")
    total_tokens = len(dataset) * SEQ_LEN
    print(f"   Total tokens: {total_tokens:,} ({total_tokens / 1e6:.0f}M)")

    # --- Training with split learning rates ---

    print("\n9. Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Split optimizer: LoRA params get standard lr, new embedding rows get
    # a higher lr since they're starting from mean-of-subword initialization.
    embed_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "new_weight" in name:
            embed_params.append(param)
        else:
            other_params.append(param)

    print(f"   New-token params: {sum(p.numel() for p in embed_params):,}")
    print(f"   LoRA params: {sum(p.numel() for p in other_params):,}")

    trainer.optimizer = AdamW([
        {"params": other_params, "lr": LR},
        {"params": embed_params, "lr": NEW_TOKEN_LR},
    ])

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Save adapter (new_weight tensors are captured automatically) ---

    print("\n10. Saving adapter...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
    print(f"   Saved to {OUTPUT_DIR}/adapter")
    print("\nCPT complete!")


if __name__ == "__main__":
    main()
