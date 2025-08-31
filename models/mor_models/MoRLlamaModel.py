import torch
from typing import Optional, Tuple, Union, Unpack
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, List, Dict, Any
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

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

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

class RecursiveDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, base_depth: int, num_recursion: int, sharing: str, update_cache: bool = False) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        self.base_depth = base_depth
        self.num_recursion = num_recursion
        self.sharing = sharing
        self.update_cache = update_cache
        assert sharing in ["cycle", "middle_cycle", "sequence"], f"Invalid sharing type: {sharing}. Must be one of ['cycle', 'middle_cycle', 'sequence']"

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        
        if self.sharing == "cycle":
            return self.key_cache[layer_idx % self.base_depth], self.value_cache[layer_idx % self.base_depth]
        elif self.sharing == "middle_cycle":
            if layer_idx == self.base_depth * self.num_recursion + 1:
                return self.key_cache[self.base_depth + 1], self.value_cache[self.base_depth + 1]
            else:
                return self.key_cache[1 + (layer_idx - 1) % self.base_depth], self.value_cache[1 + (layer_idx - 1) % self.base_depth]
        elif self.sharing == "sequence":
            return self.key_cache[layer_idx // self.base_depth], self.value_cache[layer_idx // self.base_depth]

    def _update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (
            len(self.key_cache[layer_idx]) == 0
        ):  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if self.sharing == "cycle":
                if layer_idx < self.base_depth:
                    self._update(key_states, value_states, layer_idx)
                else:
                    return self.key_cache[layer_idx % self.base_depth], self.value_cache[layer_idx % self.base_depth]
                
            elif self.sharing == "middle_cycle":
                if layer_idx < self.base_depth + 1 or layer_idx == self.base_depth * self.num_recursion + 1:
                    if layer_idx == self.base_depth * self.num_recursion + 1:
                        layer_idx = self.base_depth + 1
                    self._update(key_states, value_states, layer_idx)
                else:
                    if not self.update_cache:
                        return self.key_cache[1 + (layer_idx - 1) % self.base_depth], self.value_cache[1 + (layer_idx - 1) % self.base_depth]
                    else:
                        assert "selected_tokens" in cache_kwargs, "selected_tokens must be provided when update_cache is True"
                        _, num_heads, _, head_dim = key_states.shape
                        selected_tokens = cache_kwargs.get("selected_tokens")
                        selected_tokens = selected_tokens.unsqueeze(1).expand(-1, num_heads, -1, head_dim)
                        
                        key_cache = torch.scatter(
                            self.key_cache[1 + (layer_idx - 1) % self.base_depth],
                            dim=2,
                            index=selected_tokens,
                            src=torch.gather(key_states, dim=2, index=selected_tokens),
                        )
                        value_cache = torch.scatter(
                            self.value_cache[1 + (layer_idx - 1) % self.base_depth],
                            dim=2,
                            index=selected_tokens,
                            src=torch.gather(value_states, dim=2, index=selected_tokens),
                        )
                        return key_cache, value_cache
                        
            elif self.sharing == "sequence":
                if layer_idx % self.num_recursion == 0:
                    layer_idx = layer_idx // self.num_recursion
                    self._update(key_states, value_states, layer_idx)
                else:
                    return self.key_cache[layer_idx // self.num_recursion], self.value_cache[layer_idx // self.num_recursion]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

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