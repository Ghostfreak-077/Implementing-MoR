from utils.config import Config
import torch
import torch.nn as nn
from models.mor_models.MoRLlamaModel import MoRLlamaModel
from typing import Optional, Tuple, Union, List
from transformers.cache_utils import Cache
from models.mor_models.MoRDecoderLayers import MoRLlamaDecoderLayer
from models.mor_models.MoROutputWithPast import MoRCausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

cfg = Config()

class MoRLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = MoRLlamaModel(config).to(cfg.device)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def transform_layer_to_mor_expert(self, cfg):
        # from model.mor_model.expert_choice_router import MoRLlamaDecoderLayer

        capacity = [float(cap) for cap in cfg.mor.capacity.split(',')]
        # warmup_step for capacity_factor
        if "cap_warmup_step" in cfg.mor.expert and cfg.mor.expert.cap_warmup_step is not None:
            cap_warmup_step = cfg.mor.expert.cap_warmup_step
        else:
            cap_warmup_step = cfg.num_warmup_steps * cfg.gradient_accumulation_steps

        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion
        num_hidden_layers = len(self.model.layers)

        # Cycle sharing is for early-exiting mechanism
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = nn.ModuleList(
                [
                    MoRLlamaDecoderLayer(self.config, nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]),
                                         cfg, capacity[recur_idx], cap_warmup_step,)
                    for recur_idx in range(num_recursion)
                ]
            )
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + \
                [
                    MoRLlamaDecoderLayer(self.config, nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]),
                                         cfg, capacity[recur_idx], cap_warmup_step,)
                    for recur_idx in range(num_recursion)
                ]
                + [self.model.layers[-1]]
            )

    def transform_layer_to_mor_token(self, cfg):

        # warmup_step for balancing
        bal_warmup_step = 0
        if "bal_warmup_step" in cfg.mor.token and cfg.mor.token.bal_warmup_step > 0:
            bal_warmup_step = cfg.mor.token.bal_warmup_step * cfg.gradient_accumulation_steps

        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion
        num_hidden_layers = len(self.model.layers)

        # Cycle sharing is for early-exiting mechanism
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = MoRLlamaDecoderLayer(
                self.config,
                nn.ModuleList([nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                cfg,
                bal_warmup_step,
            )
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + \
                [MoRLlamaDecoderLayer(
                    self.config,
                    nn.ModuleList([nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                    cfg,
                    bal_warmup_step,
                ),] + \
                [self.model.layers[-1]]
            )

    def set_kv_sharing_config(self, cfg):
        if cfg.kv_sharing.sharing in ["cycle", "sequence"]:
            base_depth = self.config.num_hidden_layers // cfg.kv_sharing.num_recursion
        elif cfg.kv_sharing.sharing in ["middle_cycle"]:
            base_depth = (self.config.num_hidden_layers - 2) // cfg.kv_sharing.num_recursion

        if "kv_sharing" in cfg:
            kwargs = {
                "enable": cfg.kv_sharing.enable,
                "base_depth": base_depth,
                "num_recursion": cfg.kv_sharing.num_recursion,
                "sharing": cfg.kv_sharing.sharing,
                "update_cache": cfg.kv_sharing.update_cache if "update_cache" in cfg.kv_sharing else False,
            }
            self.model.config.kv_sharing = kwargs
        else:
            self.model.config.kv_sharing = None

    # @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=MoRCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, MoRCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sampling_loss=outputs.sampling_loss,
            sampling_acc=outputs.sampling_acc,
            sampling_topk_acc=outputs.sampling_topk_acc,
            uniformity=outputs.uniformity,
            dead_token_seq=outputs.dead_token_seq,
            balancing_loss=outputs.balancing_loss,
            balancing_ratio=outputs.balancing_ratio,
            router_z_loss=outputs.router_z_loss,
        )
