import torch
from omegaconf import OmegaConf
from transformers import AutoConfig
from utils.config import Config
from models.mor_models.MoRLlamaForCausalLM import MoRLlamaForCausalLM
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

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

print("Model configuration loaded.")

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
print("Model loaded.")

tokenizer = load_tokenizer_from_config(cfg=None)

print("Tokenizer loaded.")

input_text = "Hello, this is a test for the MoR model."
inputs = tokenizer(input_text, return_tensors="pt")

print("Input tokenized.")

print(f"Input text: '{input_text}'")
print(f"Input tensor shape: {inputs['input_ids'].shape}")


inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

config_path = "conf/250720_pretrain_smollm-360m_rec4_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001.yaml"
omegaConfig = OmegaConf.load(config_path)

model.transform_layer_to_mor_expert(omegaConfig)

with torch.no_grad():
    outputs = model(**inputs)
    
print(f"Output logits shape: {outputs.logits.shape}")
print("\nScript finished. You have successfully built an MoR model.")