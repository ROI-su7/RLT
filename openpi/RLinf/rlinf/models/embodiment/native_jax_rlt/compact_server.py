"""Robot-local compact RLT responses and action-priority websocket serving.

This adapter reuses the original OpenPI policy without editing its model,
inference code, ROS mux, or CAN drivers. Raw prefixes stay on the robot.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch


class CompactRLTPolicy:
    """Encode the exact action-request prefix locally using the frozen token."""

    def __init__(self, policy: Any):
        if policy._rlt_encoder is None:
            raise ValueError("Compact serving requires a Stage-1 RLT token checkpoint.")
        self.policy = policy
        self.metadata = dict(policy.metadata)
        self.metadata["rlt_action_z_rl"] = True
        self.metadata["rlt_action_priority"] = True

    @torch.no_grad()
    def infer(self, request: dict) -> dict:
        response = self.policy.infer(request)
        if "rlt_prefix_batch" in request:
            return response
        start = time.monotonic()
        # Match the existing network path's fp16 -> bf16 conversion exactly.
        prefix = torch.as_tensor(
            response.pop("rlt_prefix"),
            device=self.policy._rlt_encoder_device,
            dtype=torch.bfloat16,
        ).unsqueeze(0)
        mask = torch.tensor(
            response.pop("rlt_prefix_mask"), device=prefix.device, dtype=torch.bool
        ).unsqueeze(0)
        response["rlt_z_rl"] = (
            self.policy._rlt_encoder(prefix, mask).reshape(-1).float().cpu().numpy()
        )
        response["rlt_prefix_valid"] = mask.all(dim=1).cpu().numpy()
        response["rlt_encoder_fingerprint"] = self.metadata["rlt_encoder_fingerprint"]
        response.setdefault("policy_timing", {})["rlt_encoder_ms"] = (
            time.monotonic() - start
        ) * 1000
        return response


class ActionPriorityServer:
    """Serialize model calls, serving action requests ahead of queued stride work.

    Each stride batch is split into single-observation jobs. An action waits
    for at most the currently running job, not a whole four-offset batch.
    The event loop stays responsive while the model executes in a worker.
    """

    def __init__(self, policy: CompactRLTPolicy):
        self.policy = policy
        self._queue = asyncio.PriorityQueue()
        self._sequence = itertools.count()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rlt-jax-infer"
        )

    async def _submit(self, request: dict, priority: int) -> dict:
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((priority, next(self._sequence), request, future))
        return await future

    async def worker(self) -> None:
        """Run only one JAX/Torch call at a time, off the websocket event loop."""
        while True:
            _, _, request, future = await self._queue.get()
            try:
                if future.cancelled():
                    continue
                response = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self.policy.infer, request
                )
                if not future.cancelled():
                    future.set_result(response)
            except Exception as exc:
                if not future.cancelled():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def infer(self, request: dict) -> dict:
        """Return actions immediately or combine ordered low-priority stride jobs."""
        if "rlt_prefix_batch" not in request:
            return await self._submit(request, 0)
        requests = request["rlt_prefix_batch"]
        if not requests or len(requests) > 16:
            raise ValueError("Expected 1..16 stride observations.")
        responses = await asyncio.gather(
            *[self._submit({"rlt_prefix_batch": [item]}, 1) for item in requests]
        )
        return {
            "rlt_z_rl": np.concatenate([item["rlt_z_rl"] for item in responses]),
            "rlt_prefix_valid": np.concatenate(
                [item["rlt_prefix_valid"] for item in responses]
            ),
            "policy_timing": {
                "infer_ms": sum(
                    item.get("policy_timing", {}).get("infer_ms", 0.0)
                    for item in responses
                )
            },
        }

    async def handler(self, websocket) -> None:
        """Serve the original OpenPI msgpack wire format and metadata handshake."""
        import websockets
        from openpi_client import msgpack_numpy

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        try:
            async for message in websocket:
                start = time.monotonic()
                try:
                    response = await self.infer(msgpack_numpy.unpackb(message))
                    response["server_timing"] = {
                        "infer_ms": (time.monotonic() - start) * 1000
                    }
                    await websocket.send(packer.pack(response))
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(code=1011, reason="RLT inference failed")
                    return
        except websockets.ConnectionClosed:
            return

    async def run(self, host: str, port: int) -> None:
        """Start the action-priority server without opening any robot interface."""
        from websockets.asyncio.server import serve

        task = asyncio.create_task(self.worker())
        try:
            async with serve(
                self.handler, host, port, compression=None, max_size=None
            ) as server:
                await server.serve_forever()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._executor.shutdown(wait=True)
