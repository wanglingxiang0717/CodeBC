# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
# from transformers.deepspeed import HfDeepSpeedConfig
# from transformers.integrations import HfDeepSpeedConfig
try:
    from transformers.integrations import HfDeepSpeedConfig
    print("Using integrations.HfDeepSpeedConfig (new API)")
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig
    print("Using deepspeed.HfDeepSpeedConfig (old API)")
from transformers import LlamaForCausalLM, LlamaConfig


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    cuda_device_index=None,
                    torch_dtype=torch.bfloat16,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    # if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
    if ds_config is not None:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if cuda_device_index is None:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            # use_auth_token=True,
            # attn_implementation="flash_attention_2",
            )
    
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True,
            device_map={"": cuda_device_index},
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2",
            # use_auth_token=True
            )
        # model = model.to(f"cuda:{cuda_device_index}")

    # llama use eos_token_id but not end_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    # # compatible with OPT and llama2
    model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

# def create_hf_model(
#     model_class,
#     model_name_or_path,
#     tokenizer,
#     ds_config=None,
#     disable_dropout=False,
#     torch_dtype=torch.bfloat16,
# ):
#     model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
#     if disable_dropout:
#         model_config.dropout = 0.0

#     model = model_class.from_pretrained(
#         model_name_or_path,
#         config=model_config,
#         trust_remote_code=True,
#         torch_dtype=torch_dtype,
#     )

#     model.config.end_token_id = tokenizer.eos_token_id
#     model.config.pad_token_id = model.config.eos_token_id

#     if hasattr(model, "gradient_checkpointing_enable"):
#         model.gradient_checkpointing_enable()

#     if ds_config is not None:
#         dschf = HfDeepSpeedConfig(ds_config)

#     if torch.cuda.is_available():
#         model = model.to(torch.cuda.current_device())

#     return model