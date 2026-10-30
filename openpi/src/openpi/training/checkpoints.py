from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import pathlib
import shutil
from typing import Protocol

from etils import epath
from etils.epath import gpath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
from orbax.checkpoint._src.path import async_utils as orbax_path_async_utils

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def _resolve_checkpoint_dir(checkpoint_dir: epath.Path | pathlib.Path | str) -> pathlib.Path:
    """Resolve checkpoint directories as local pathlib paths.

    Orbax's save path currently passes a `mode=` argument into `mkdir`. That works for
    `pathlib.Path`, but fails for `etils.epath.GPath` on local filesystems with:
    `NotImplementedError: mkdir with custom mode= not supported`.
    """
    checkpoint_dir = str(checkpoint_dir)
    if "://" in checkpoint_dir and not checkpoint_dir.startswith("file://"):
        raise ValueError(
            "Remote checkpoint directories are not supported for training in this project. "
            f"Use a local path instead of: {checkpoint_dir}"
        )
    if checkpoint_dir.startswith("file://"):
        checkpoint_dir = checkpoint_dir[len("file://") :]
    return pathlib.Path(checkpoint_dir).expanduser().resolve()


_ORIGINAL_ORBAX_ASYNC_MAKEDIRS = orbax_path_async_utils.async_makedirs


def _patched_async_makedirs(
    path: epath.Path,
    *args,
    parents: bool = False,
    exist_ok: bool = True,
    **kwargs,
):
    # `etils.epath.PosixGPath.mkdir` raises on custom mode values, while Orbax
    # unconditionally forwards `mode=` when creating tmp checkpoint dirs.
    if isinstance(path, gpath.PosixGPath):
        kwargs.pop("mode", None)
    return _ORIGINAL_ORBAX_ASYNC_MAKEDIRS(path, *args, parents=parents, exist_ok=exist_ok, **kwargs)


orbax_path_async_utils.async_makedirs = _patched_async_makedirs


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | pathlib.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        # options=ocp.CheckpointManagerOptions(
        #     max_to_keep=1,
        #     keep_period=keep_period,
        #     create=False,
        #     async_options=ocp.AsyncOptions(timeout_secs=7200),
        # ),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            enable_async_checkpointing=False,
            async_options=ocp.AsyncOptions(
                timeout_secs=7200,
                create_directories_asynchronously=False,
            ),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
