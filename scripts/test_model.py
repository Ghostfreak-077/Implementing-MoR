import torch
from omegaconf import OmegaConf
from transformers import AutoConfig
from utils.config import Config
from models.mor_models.MoRLlamaForCausalLM import MoRLlamaForCausalLM
from transformers import AutoTokenizer

cfg = Config()
attn_implementation = "flash_attention_2"
torch_dtype = torch.bfloat16

config = AutoConfig.from_pretrained(
    "HuggingFaceTB/SmolLM-360M",
    # attn_implementation=attn_implementation,
    torch_dtype=torch_dtype,
    local_files_only=False, # Assume offline mode for this script,
    map_location=cfg.device
)

def load_tokenizer_from_config(cfg: None):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
            # if cfg.tokenizer in ["smollm", "smollm2"]:
            # '<|endoftext|>'
            tokenizer.pad_token_id = 0
        # else:
        #     raise ValueError(f"Tokenizer {cfg.tokenizer} does not have a pad token, please specify one in the config")
    return tokenizer

model = MoRLlamaForCausalLM(config=config).to(cfg.device)
tokenizer = load_tokenizer_from_config(cfg=None)

input_text = "Hello, this is a test for the MoR model."
inputs = tokenizer(input_text, return_tensors="pt")

print(f"Input text: '{input_text}'")
print(f"Input tensor shape: {inputs['input_ids'].shape}")


inputs = {k: v.to("cuda") for k, v in inputs.items()}

# model.transform_layer_to_mor_expert(config)

with torch.no_grad():
    outputs = model(**inputs)
    
print(f"Output logits shape: {outputs.logits.shape}")
print("\nScript finished. You have successfully built an MoR model.")