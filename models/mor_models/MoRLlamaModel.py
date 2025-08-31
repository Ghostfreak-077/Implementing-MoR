import torch
from typing import Optional, Tuple, Union, Unpack
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, RecursiveDynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from models.mor_models.MoROutputWithPast import MoRBaseModelOutputWithPast
from transformers.utils import logging
from models.llama_models import LlamaModel
from models.mor_models.utils import logger
from transformers.utils.doc import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class MoRLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """
    def __init__(self, config):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, MoRBaseModelOutputWithPast]:

        # Performing checks to inputs

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            if "kv_sharing" in self.config and self.config.kv_sharing is not None:
                kwargs = self.config.kv_sharing
                past_key_values = RecursiveDynamicCache(kwargs["base_depth"], kwargs["num_recursion"], kwargs["sharing"], kwargs["update_cache"])
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        prev_selected_tokens = None
        sampling_loss = torch.tensor(0.0, device=hidden_states.device)
        sampling_acc_list = []
        sampling_topk_acc_list = []
        uniformity = None # torch.tensor(0.0, device=hidden_states.device)
        dead_token_seq = None # torch.tensor([0.0] * self.config.max_position_embeddings, device=hidden_states.device)
        balancing_loss = torch.tensor(0.0, device=hidden_states.device)
        balancing_ratio = torch.tensor(0.0, device=hidden_states.device)
        router_z_loss = torch.tensor(0.0, device=hidden_states.device)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # TODO: support MoRLlamaDecoderLayer
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if hasattr(decoder_layer, "mor") and decoder_layer.mor:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        prev_selected_tokens=prev_selected_tokens,
                        **flash_attn_kwargs,
                    )
                    if decoder_layer.mor_type == "expert":
                        prev_selected_tokens = layer_outputs.selected_tokens
                        if layer_outputs.sampling_loss is not None:
                            sampling_loss += layer_outputs.sampling_loss
                        if layer_outputs.sampling_acc is not None:
                            sampling_acc_list.append(layer_outputs.sampling_acc)
                        if layer_outputs.sampling_topk_acc is not None:
                            sampling_topk_acc_list.append(layer_outputs.sampling_topk_acc)
                        # if layer_outputs.uniformity is not None:
                        #     uniformity += layer_outputs.uniformity
                        # if layer_outputs.dead_token_seq is not None:
                        #     dead_token_seq = layer_outputs.dead_token_seq
                        if layer_outputs.router_z_loss is not None:
                            router_z_loss += layer_outputs.router_z_loss

                    elif decoder_layer.mor_type == "token":
                        if layer_outputs.balancing_loss is not None:
                            balancing_loss = layer_outputs.balancing_loss
                        if layer_outputs.balancing_ratio is not None:
                            balancing_ratio = layer_outputs.balancing_ratio
                        if layer_outputs.router_z_loss is not None:
                            router_z_loss = layer_outputs.router_z_loss
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = MoRBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sampling_loss=sampling_loss,
            sampling_acc=sum(sampling_acc_list)/len(sampling_acc_list) if len(sampling_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            sampling_topk_acc=sum(sampling_topk_acc_list)/len(sampling_topk_acc_list) if len(sampling_topk_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            uniformity=uniformity,
            dead_token_seq=dead_token_seq,
            balancing_loss=balancing_loss,
            balancing_ratio=balancing_ratio,
            router_z_loss=router_z_loss,
        )
        return output if return_dict else output.to_tuple()