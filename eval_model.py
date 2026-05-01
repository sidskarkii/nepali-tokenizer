"""Evaluate the CPT/SFT Nepali model against the base Qwen3-4B.

Loads both models, runs them on the same Nepali prompts, and compares:
  - Token efficiency (extended tokenizer vs base)
  - Perplexity on held-out Nepali text
  - Generation quality on comprehension, translation, and creative tasks

The SFT model requires manual restoration of the trained new-token embeddings
from the CPT checkpoint. See load_sft_model() for the procedure.

Usage:
    python eval_model.py
    python eval_model.py --cpt_checkpoint output/cpt-qwen3-4b-nepali/adapter \
                         --sft_adapter output/sft-qwen3-4b-nepali/adapter
"""

import argparse
import math
import os

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B"
CPT_CHECKPOINT = "output/cpt-qwen3-4b-nepali/adapter"
SFT_ADAPTER = "output/sft-qwen3-4b-nepali/adapter"

EVAL_TEXTS = [
    "नेपालको संविधानले नागरिकहरूलाई मौलिक अधिकारहरू प्रदान गर्दछ। यी अधिकारहरूमा बाँच्ने अधिकार, स्वतन्त्रताको अधिकार, समानताको अधिकार, र सम्पत्तिको अधिकार समावेश छन्।",
    "हिमालयन क्षेत्रमा जलवायु परिवर्तनको प्रभाव धेरै गहिरो छ। हिउँदमा हिमपातको मात्रा घटेको छ र तापक्रम बढेको छ। यसले कृषि, पर्यटन, र जनजीवनमा प्रभाव पारेको छ।",
    "काठमाडौं उपत्यकामा तीनवटा ऐतिहासिक शहरहरू छन्: काठमाडौं, ललितपुर, र भक्तपुर। यी तीनै शहरहरूमा प्राचीन मन्दिरहरू, दरबार क्षेत्रहरू, र सांस्कृतिक सम्पदाहरू छन्।",
]

GENERATION_PROMPTS = [
    {
        "label": "Comprehension",
        "prompt": (
            "### Instruction:\n"
            "नेपालको संविधान २०७२ सालमा जारी भयो। यसले नेपाललाई संघीय लोकतान्त्रिक "
            "गणतन्त्रको रूपमा परिभाषित गर्छ। संविधानमा सात प्रदेशको व्यवस्था गरिएको छ।\n\n"
            "नेपालको संविधान कहिले जारी भयो र यसले कतिवटा प्रदेशको व्यवस्था गरेको छ?\n\n"
            "### Response:\n"
        ),
    },
    {
        "label": "Translation",
        "prompt": (
            "### Instruction:\n"
            "Translate to English: नेपाल एक सुन्दर देश हो जहाँ विविध जातजाति र "
            "संस्कृतिका मानिसहरू बस्छन्।\n\n"
            "### Response:\n"
        ),
    },
    {
        "label": "Creative",
        "prompt": (
            "### Instruction:\n"
            "नेपालको प्रकृतिको बारेमा एउटा छोटो कविता लेख्नुहोस्।\n\n"
            "### Response:\n"
        ),
    },
]


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return model, tokenizer


def load_sft_model(cpt_checkpoint: str, sft_adapter: str):
    """Load the SFT model with manually restored new-token embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model, sft_adapter, torch_dtype=torch.bfloat16, is_trainable=False,
    )

    # Restore trained new-token embeddings from the CPT checkpoint
    sd = load_file(os.path.join(cpt_checkpoint, "adapter_model.safetensors"))
    bm = model.get_base_model()
    with torch.no_grad():
        bm.get_input_embeddings().weight[base_vocab_size:len(tokenizer)].copy_(
            sd["base_model.model.model.embed_tokens.new_weight"].to(
                device=bm.get_input_embeddings().weight.device, dtype=torch.bfloat16))
        bm.get_output_embeddings().weight[base_vocab_size:len(tokenizer)].copy_(
            sd["base_model.model.lm_head.trainable_embedding.new_weight"].to(
                device=bm.get_output_embeddings().weight.device, dtype=torch.bfloat16))

    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, texts, max_length=512):
    total_loss = 0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    return math.exp(total_loss / total_tokens)


def generate(model, tokenizer, prompt, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    if "### Response:" in full:
        return full.split("### Response:")[-1].strip()
    return full[len(prompt):].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpt_checkpoint", type=str, default=CPT_CHECKPOINT)
    parser.add_argument("--sft_adapter", type=str, default=SFT_ADAPTER)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    for label, loader in [
        ("Qwen3-4B BASE", load_base_model),
        ("Qwen3-4B + Nepali CPT + SFT", lambda: load_sft_model(args.cpt_checkpoint, args.sft_adapter)),
    ]:
        print("=" * 70)
        print(f"MODEL: {label}")
        print("=" * 70)

        model, tokenizer = loader()

        # Token efficiency
        test_text = " ".join(EVAL_TEXTS)
        n_tokens = len(tokenizer.encode(test_text, add_special_tokens=False))
        n_words = len(test_text.split())
        print(f"\n  Tokens/word: {n_tokens / n_words:.2f} ({n_tokens} tokens, {n_words} words)")

        # Perplexity
        ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS)
        print(f"  Perplexity:  {ppl:.2f}")

        # Generation
        print("\n  --- Generation ---")
        for item in GENERATION_PROMPTS:
            result = generate(model, tokenizer, item["prompt"])
            print(f"\n  [{item['label']}]")
            print(f"  {result[:300]}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
