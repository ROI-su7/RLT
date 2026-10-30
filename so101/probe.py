#!/usr/bin/env python3
"""Read-only SO101 hardware probe (never writes to a servo).

Finds every Feetech STS3215 bus on the host, reads each servo, guesses leader/follower from the supply
voltage (leader ~5 V, follower ~12 V) and writes one JSON per arm pair. Serial ports are identified by
USB serial number (/dev/serial/by-id), so the result survives re-plugging and moving to another host.

    so101/.venv/bin/python so101/probe.py --out so101/probes/set1.json
    so101/.venv/bin/python so101/probe.py --merge so101/probes/set1.json so101/probes/set2.json \
        --left set1 --right set2 --out so101/dual_arm.yaml
"""
import argparse
import glob
import json
import os
import sys

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STS3215 = 777
BAUD = 1_000_000


def _scan(port):
    from scservo_sdk import PacketHandler, PortHandler

    ph = PortHandler(port)
    if not ph.openPort():
        return None
    ph.setBaudRate(BAUD)
    pk = PacketHandler(0)

    def r1(i, a):
        v, res, _ = pk.read1ByteTxRx(ph, i, a)
        return v if res == 0 else None

    def r2(i, a):
        v, res, _ = pk.read2ByteTxRx(ph, i, a)
        return v if res == 0 else None

    joints = {}
    for i, name in enumerate(JOINTS, start=1):
        model, res, _ = pk.ping(ph, i)
        if res != 0:
            ph.closePort()
            return None
        joints[name] = dict(id=i, model=model, pos=r2(i, 56), min=r2(i, 9), max=r2(i, 11), homing=r2(i, 31))
    volt = [r1(i, 62) for i in range(1, 7)]
    out = dict(
        joints=joints,
        voltage_v=round(sum(v for v in volt if v) / max(1, sum(1 for v in volt if v)) / 10, 1),
        torque_enabled=any(r1(i, 40) for i in range(1, 7)),
        pid_p=r1(1, 21),
    )
    ph.closePort()
    return out


def probe():
    arms = {}
    for link in sorted(glob.glob("/dev/serial/by-id/*")):
        res = _scan(os.path.realpath(link))
        if res is None:
            continue
        res["by_id"] = os.path.basename(link)
        res["role"] = "follower" if res["voltage_v"] > 8 else "leader"
        arms[res["role"] + ("" if res["role"] not in arms else "_2")] = res
    return arms


def merge(args):
    import yaml

    def load(path):
        return json.load(open(path))["arms"]

    sides = {"left": load(args.merge[0] if args.left == "set1" else args.merge[1]),
             "right": load(args.merge[1] if args.right == "set2" else args.merge[0])}
    cfg = {"robot": "so101_dual", "joints": JOINTS, "arms": {}}
    for side, arms in sides.items():
        cfg["arms"][side] = {
            role: {
                "port_by_id": arms[role]["by_id"],  # /dev/serial/by-id/<this>; adapt after moving hosts
                "calibration_file": f"so101/calibration/{side}_{role}.json",  # created on the target host
            }
            for role in ("leader", "follower")
        }
    cfg["cameras"] = {"note": "fill in on the target host (camera serial or /dev/v4l/by-id path)"}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--merge", nargs=2)
    ap.add_argument("--left", default="set1")
    ap.add_argument("--right", default="set2")
    args = ap.parse_args()
    if args.merge:
        import yaml

        text = yaml.safe_dump(merge(args), sort_keys=False, allow_unicode=True)
    else:
        arms = probe()
        if not arms:
            sys.exit("no SO101 bus found: check USB and the 5V/12V servo supply")
        text = json.dumps({"arms": arms}, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
