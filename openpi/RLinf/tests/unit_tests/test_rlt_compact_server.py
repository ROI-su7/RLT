"""Compact encoding equivalence and action priority without a robot/model."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.native_jax_rlt.compact_server import (
    ActionPriorityServer,
    CompactRLTPolicy,
)


class MeanEncoder(torch.nn.Module):
    def forward(self, prefix, mask):
        return (prefix * mask[..., None]).mean(dim=1, keepdim=True)


def test_compact_token_is_identical_to_existing_transferred_prefix_encoding():
    prefix = np.random.default_rng(7).normal(size=(768, 2048)).astype(np.float16)
    mask = np.ones(768, dtype=bool)
    action = np.ones((50, 14), np.float32)
    original = SimpleNamespace(
        _rlt_encoder=MeanEncoder(),
        _rlt_encoder_device=torch.device("cpu"),
        metadata={"rlt_encoder_fingerprint": "test-token"},
        infer=lambda request: {
            "rlt_prefix": prefix.copy(),
            "rlt_prefix_mask": mask.copy(),
            "actions": action,
            "policy_timing": {"infer_ms": 60},
        },
    )
    result = CompactRLTPolicy(original).infer({})
    expected = (
        original._rlt_encoder(
            torch.as_tensor(prefix, dtype=torch.bfloat16)[None],
            torch.as_tensor(mask)[None],
        )
        .reshape(-1)
        .float()
        .numpy()
    )
    np.testing.assert_array_equal(result["rlt_z_rl"], expected)
    np.testing.assert_array_equal(result["actions"], action)
    assert "rlt_prefix" not in result and "rlt_prefix_mask" not in result
    assert result["rlt_z_rl"].nbytes == 8192
    assert result["rlt_encoder_fingerprint"] == "test-token"


def test_action_request_precedes_queued_stride_observations():
    seen = []
    tokens = [np.full((1, 2048), index, np.float32) for index in range(4)]
    validity = np.ones(1, dtype=np.bool_)
    actions = np.zeros((50, 14), np.float32)

    class Policy:
        def infer(self, request):
            if "rlt_prefix_batch" in request:
                index = request["rlt_prefix_batch"][0]["index"]
                seen.append(index)
                return {
                    "rlt_z_rl": tokens[index],
                    "rlt_prefix_valid": validity,
                    "policy_timing": {"infer_ms": 1},
                }
            seen.append("action")
            return {"actions": actions}

    async def run():
        server = ActionPriorityServer(Policy())
        batch = asyncio.create_task(
            server.infer({"rlt_prefix_batch": [{"index": i} for i in range(4)]})
        )
        action = asyncio.create_task(server.infer({"state": 1}))
        # Queue all work first, then let one worker service it in priority order.
        for _ in range(3):
            await asyncio.sleep(0)
        worker = asyncio.create_task(server.worker())
        try:
            result, _ = await asyncio.wait_for(
                asyncio.gather(batch, action), timeout=10
            )
            assert seen == ["action", 0, 1, 2, 3]
            np.testing.assert_array_equal(result["rlt_z_rl"][:, 0], np.arange(4))
        except TimeoutError as exc:
            raise AssertionError(
                f"Inference worker stalled: seen={seen}, worker_done={worker.done()}, error={worker.exception() if worker.done() else None}"
            ) from exc
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            server._executor.shutdown(wait=True)

    asyncio.run(run())


def test_empty_stride_batch_is_rejected():
    async def run():
        with pytest.raises(ValueError):
            await ActionPriorityServer(None).infer({"rlt_prefix_batch": []})

    asyncio.run(run())


def test_compact_server_matches_real_openpi_websocket_client():
    from openpi_client import msgpack_numpy
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    class Policy:
        metadata = {"rlt_action_z_rl": True, "rlt_encoder_fingerprint": "test-token"}

        def infer(self, request):
            assert request["state"].shape == (14,)
            return {
                "actions": np.zeros((50, 14), np.float32),
                "rlt_z_rl": np.ones(2048, np.float32),
                "rlt_prefix_valid": np.ones(1, dtype=np.bool_),
                "rlt_encoder_fingerprint": "test-token",
            }

    async def run():
        service = ActionPriorityServer(Policy())
        worker = asyncio.create_task(service.worker())
        try:
            async with serve(
                service.handler, "127.0.0.1", 0, compression=None, max_size=None
            ) as listener:
                port = listener.sockets[0].getsockname()[1]
                async with connect(
                    f"ws://127.0.0.1:{port}", compression=None, max_size=None
                ) as client:
                    metadata = msgpack_numpy.unpackb(await client.recv())
                    assert metadata["rlt_action_z_rl"]
                    packer = msgpack_numpy.Packer()
                    await client.send(packer.pack({"state": np.zeros(14, np.float32)}))
                    response = msgpack_numpy.unpackb(await client.recv())
                    assert response["actions"].shape == (50, 14)
                    assert response["rlt_z_rl"].shape == (2048,)
                    assert response["rlt_encoder_fingerprint"] == "test-token"
                    assert response["server_timing"]["infer_ms"] >= 0
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            service._executor.shutdown(wait=True)

    asyncio.run(run())
