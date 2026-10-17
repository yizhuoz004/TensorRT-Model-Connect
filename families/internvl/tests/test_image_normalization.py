# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
from families.internvl.model import _InternVLModel


def test_bundle_normalization_matches_internvl_imagenet_processor():
    config = SimpleNamespace(raw={"vision_config": {"patch_size": 14}}, hidden_size=16)
    runtime = _InternVLModel().get_vl_config(config)
    black_white = np.asarray([[0.0] * 3, [1.0] * 3])
    actual = (black_white - runtime["image_mean"]) / runtime["image_std"]
    expected = [[-2.11790393, -2.03571429, -1.80444444],
                [2.24890830, 2.42857143, 2.64000000]]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("value", [0, 13, True, 1.5])
def test_tile_budget_rejects_unsupported_values(tmp_path, value):
    from families.internvl.build_request import BuildRequest

    with pytest.raises(ValueError, match="max_image_tiles"):
        BuildRequest(model_dir=tmp_path, output_path=tmp_path / "model.bundle",
                     family="internvl", task="vision_language_generation",
                     precision="fp16", max_image_tiles=value)


def test_tile_budget_survives_legacy_request_conversion(tmp_path):
    from tensorrt_model_connect import BuildRequest as LegacyRequest
    from families.internvl.build_request import coerce_request

    request = LegacyRequest(model_dir=tmp_path, output_path=tmp_path / "model.bundle",
                            family="internvl", task="vision_language_generation", precision="fp16")
    assert coerce_request(request).max_image_tiles == 1


@pytest.mark.parametrize(("context", "tp", "error", "message"), [
    (384, 1, ValueError, "leave text space"),
    (4096, 2, NotImplementedError, "require TP1"),
])
def test_tile_budget_is_checked_before_loading_weights(tmp_path, monkeypatch, context, tp, error, message):
    from families.internvl import model
    from families.internvl.build_request import BuildRequest
    from families.internvl.edge_llm import dispatch

    class MetadataOnlyModel:
        @staticmethod
        def get_vl_config(_config):
            return {"num_image_pad_tokens": 256}

        @staticmethod
        def load_weights(*_args):
            pytest.fail("invalid tile capacity must fail before loading weights")

    config = SimpleNamespace(model_type="internvl", max_position_embeddings=32768)
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(model, "_InternVLModel", MetadataOnlyModel)
    monkeypatch.setattr(dispatch, "build", lambda request, writer, native: native(request, writer))
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "model.bundle",
                           family="internvl", task="vision_language_generation", precision="fp16",
                           max_image_tiles=12, max_sequence_length=context, tensor_parallel_size=tp)
    with pytest.raises(error, match=message):
        model.build(request, None)
