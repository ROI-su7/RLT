#!/usr/bin/env python3
"""Piper leader linkage-target CAN-ID offset configuration.

Run only while the follower-arm aviation plug on the selected CAN bus is
physically disconnected. This prevents the broadcast 0x470 configuration frame
from changing the follower.
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", choices=("can_left", "can_right"), required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--restore-default",
        action="store_true",
        help="Restore the leader control frame base from 0x16x to 0x15x.",
    )
    parser.add_argument(
        "--i-confirm-follower-disconnected",
        action="store_true",
        dest="confirmed",
    )
    args = parser.parse_args()
    if not args.apply:
        print(
            "Dry run only. Disconnect the corresponding FOLLOWER aviation plug, "
            "then repeat with --apply --i-confirm-follower-disconnected."
        )
        return
    if not args.confirmed:
        raise SystemExit(
            "Refusing to broadcast 0x470 without explicit follower-disconnected "
            "confirmation."
        )

    from piper_sdk import C_PiperInterface_V2

    linkage_offset = 0x00 if args.restore_default else 0x10
    piper = C_PiperInterface_V2(can_name=args.can)
    piper.ConnectPort()
    try:
        time.sleep(0.5)
        piper.ClearRespSetInstruction()
        # The leader emits linkage-target frames. linkage_offset changes their
        # destination from 0x15x to 0x16x; ctrl_offset instead changes which
        # incoming control IDs this arm accepts and must remain at its default.
        piper.MasterSlaveConfig(0xFA, 0x00, 0x00, linkage_offset)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            response = piper.GetRespInstruction()
            if response.instruction_response.instruction_index == 0x70:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(
                "Piper did not acknowledge the 0x470 configuration within 1 second."
            )
    finally:
        piper.DisconnectPort()
    frame_base = "0x15x (default)" if args.restore_default else "0x16x (mode 2)"
    print(
        "{} leader acknowledged configuration for {}. Keep the follower "
        "disconnected and verify emitted frames before reconnecting; do not "
        "power-cycle before this first validation.".format(args.can, frame_base)
    )


if __name__ == "__main__":
    main()
