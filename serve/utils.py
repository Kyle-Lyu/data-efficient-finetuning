import torch

from peft import PeftModel
from transformers import StoppingCriteria, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class KeyWordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        outputs = self.tokenizer.batch_decode(input_ids[:, self.start_len:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    

def load_model_tokenizer(base_model_path, lora_model_path, model_name, load_8bit=False, load_4bit=False, device=None):
    device_map = device if device else "auto"
    kwargs = {"device_map": device_map}

    if load_8bit:
        # transformers version 4.41.1
        # The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
        # kwargs["load_in_8bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif load_4bit:
        # transformers version 4.38.2
        # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time.
        # kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=True,
    )
    if "llama" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
    elif "deepseek" in model_name.lower():
        tokenizer.eos_token = "<|EOT|>"
        tokenizer.pad_token = "<｜end▁of▁sentence｜>"

    print(f"loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )
    if lora_model_path:
        print(f"loading adapter model from {lora_model_path}")
        lora_model = PeftModel.from_pretrained(
            model=base_model, 
            model_id=lora_model_path,
        )
        model = lora_model.merge_and_unload()
    else:
        model = base_model

    return model, tokenizer