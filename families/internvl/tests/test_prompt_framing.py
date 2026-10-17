# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one-view native framing with actual upstream media insertion."""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")
pytest.importorskip("transformers.models.internvl.processing_internvl")

from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GotOcr2ImageProcessor, InternVLProcessor, PreTrainedTokenizerFast
from transformers.models.internvl.video_processing_internvl import InternVLVideoProcessor

from families.internvl.model import _InternVLModel


def test_native_prompt_matches_complete_upstream_media_processing() -> None:
    special = {
        "start_image_token": "<img>", "end_image_token": "</img>",
        "context_image_token": "<IMG_CONTEXT>", "video_token": "<video>",
    }
    tokens = ["[UNK]", "<|im_start|>", "<|im_end|>", *special.values()]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]",
        additional_special_tokens=tokens[1:], extra_special_tokens=special,
    )
    processor = InternVLProcessor(
        GotOcr2ImageProcessor(crop_to_patches=False), tokenizer, image_seq_length=256,
        video_processor=InternVLVideoProcessor(),
    )
    # A single user message, before the processor expands the image placeholder.
    chat = "<|im_start|>user\n<IMG_CONTEXT>\nQuestion?<|im_end|>\n<|im_start|>assistant\n"
    official = processor(
        text=chat, images=Image.new("RGB", (32, 48)), crop_to_patches=False,
    )["input_ids"][0]
    config = _InternVLModel().get_vl_config(
        SimpleNamespace(raw={"vision_config": {"patch_size": 14}}, hidden_size=16)
    )
    prompt = config["vl_prompt_template"].replace("{prompt}", "Question?").replace(
        "{image_pads}", config["image_token_str"] * config["num_image_pad_tokens"]
    )
    assert tokenizer.encode(prompt, add_special_tokens=False) == official
