# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The keyboard is found again after a USB reset gives it a new event node."""

import errno
from types import SimpleNamespace

import pytest

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener

EV_KEY = 1
ECODES = SimpleNamespace(
    EV_KEY=EV_KEY,
    **{name: code for code, name in enumerate(KeyboardListener.REQUIRED_KEY_NAMES)},
)
KEYBOARD_KEYS = list(range(len(KeyboardListener.REQUIRED_KEY_NAMES)))


class _FakeDevice:
    def __init__(self, path: str, keys: list[int]):
        self.path = path
        self.name = path
        self._keys = keys
        self.closed = False

    def capabilities(self, verbose=False):
        del verbose
        return {EV_KEY: self._keys}

    def close(self):
        self.closed = True


def _listener(devices: dict[str, list[int]], monkeypatch) -> KeyboardListener:
    """A listener without its reader thread, backed by fake event nodes."""
    monkeypatch.delenv("RLINF_KEYBOARD_DEVICE", raising=False)
    listener = object.__new__(KeyboardListener)
    listener._ecodes = ECODES
    listener._list_devices = lambda: list(devices)
    opened = []

    def open_device(path):
        if path not in devices:
            raise FileNotFoundError(errno.ENOENT, "gone", path)
        device = _FakeDevice(path, devices[path])
        opened.append(device)
        return device

    listener._input_device_cls = open_device
    listener.opened = opened
    return listener


def test_same_node_is_kept_when_it_is_still_the_keyboard(monkeypatch) -> None:
    listener = _listener({"/dev/input/event3": KEYBOARD_KEYS}, monkeypatch)

    device = listener._reopen_keyboard_device("/dev/input/event3")

    assert device.path == "/dev/input/event3"


def test_new_node_is_found_after_reenumeration(monkeypatch) -> None:
    listener = _listener(
        {"/dev/input/event2": [0], "/dev/input/event27": KEYBOARD_KEYS},
        monkeypatch,
    )

    device = listener._reopen_keyboard_device("/dev/input/event3")

    assert device.path == "/dev/input/event27"


def test_old_node_reused_by_another_device_is_rejected(monkeypatch) -> None:
    listener = _listener(
        {"/dev/input/event3": [0], "/dev/input/event27": KEYBOARD_KEYS},
        monkeypatch,
    )

    device = listener._reopen_keyboard_device("/dev/input/event3")

    assert device.path == "/dev/input/event27"
    stale = [d for d in listener.opened if d.path == "/dev/input/event3"]
    assert stale and all(d.closed for d in stale)


@pytest.mark.parametrize(
    "devices",
    [
        {},  # still unplugged
        {"/dev/input/event4": KEYBOARD_KEYS, "/dev/input/event5": KEYBOARD_KEYS},
    ],
)
def test_no_unique_keyboard_means_retry_later(devices, monkeypatch) -> None:
    listener = _listener(devices, monkeypatch)

    assert listener._reopen_keyboard_device("/dev/input/event3") is None
