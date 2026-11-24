"""
McCulloch-Pitts Neurons: A Binary Adder from Scratch in NumPy scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""McCulloch-Pitts Neurons: A Binary Adder from Scratch in NumPy (McCulloch and Pitts, 1943).

Story: the paper's four elementary nets, each one neuron and one time step; a half adder
whose carry would come out a step too early without a relay; a 4-bit adder of 125 threshold
neurons that takes a new pair of numbers on every time step and gives each sum 17 steps
later; and the same wiring with every relay neuron removed, which still adds when the
inputs are held still and fails when they change. Nothing is learned anywhere.
Runs in about a second.
"""
import numpy as np


def spikes(raster, net, name):
    """One neuron's row of the raster as a string, '#' for a spike and '.' for silence."""
    return "".join("#" if v else "." for v in raster[:, net["index"][name]])


def read_sums(raster, net, outputs, times):
    """Decode the output neurons at the given time steps into integers."""
    return sum(raster[times, net["index"][name]] << j for j, name in enumerate(outputs))


def number_trains(x, y, n_bits, hold=1):
    """Spike trains for the pairs (x[k], y[k]), each pair presented for `hold` time steps."""
    trains = {}
    for i in range(n_bits):
        trains[f"a{i}"] = np.repeat((x >> i) & 1, hold)
        trains[f"b{i}"] = np.repeat((y >> i) & 1, hold)
    return trains


def main():
    # ---- 1. The four elementary nets ----
    a = [0, 0, 1, 1, 0, 1]
    b = [0, 1, 0, 1, 0, 0]
    net = compile_net(["a", "b"], {"relay(a)": gate("relay", "a"), "a or b": gate("or", "a", "b"),
                                   "a and b": gate("and", "a", "b"), "a and not b": gate("and_not", "a", "b")})
    raster = run_net(net, {"a": a, "b": b}, steps=len(a) + 1)
    print("[1] the four elementary nets: one neuron each, and each answers one time step late")
    for name in net["names"]:
        print(f"    {name:<12} {spikes(raster, net, name)}")

    # ---- 2. A half adder, and why its carry needs a relay ----
    neurons = {}
    s, c = half_adder(neurons, "a", "b", "h")
    net = compile_net(["a", "b"], neurons)
    raster = run_net(net, {"a": a, "b": b}, steps=len(a) + 2)
    print(f"\n[2] half adder, {len(neurons)} neurons, a new pair of bits on every step")
    for label, name in (("a", "a"), ("b", "b"), ("and neuron", "h.both"), ("carry", c), ("sum", s)):
        print(f"    {label:<12} {spikes(raster, net, name)}")
    print("    the and neuron answers after 1 step, the sum after 2; the relay holds the carry for the missing step")

    # ---- 3. The 4-bit adder on a stream ----
    n_bits = 4
    net, outputs, latency = ripple_adder(n_bits)
    n_neurons = len(net["names"]) - net["n_inputs"]
    n_relays = int(((net["theta"] == 1) & (net["E"].sum(axis=1) == 1) & (net["I"].sum(axis=1) == 0))[net["n_inputs"]:].sum())
    x = np.repeat(np.arange(16), 16)
    y = np.tile(np.arange(16), 16)
    got = add_stream(x, y, n_bits)
    print(f"\n[3] {n_bits}-bit adder: {n_neurons} neurons, {n_relays} of them relays, latency {latency} steps")
    print(f"    all {len(x)} pairs as one stream, a new pair on every step: {int((got == x + y).sum())} of {len(x)} sums correct "
          f"in {len(x) + latency} time steps")
    for k in (59, 187, 255):
        print(f"    pair {k:>3} entered at t = {k:>3}: {x[k]:>2} + {y[k]:>2} = {int(got[k]):>2}, read at t = {k + latency}")

    # ---- 4. The same wiring without relay neurons ----
    real_delay_line = delay_line
    globals()["delay_line"] = lambda neurons, src, k, name: src
    try:
        bare, bare_outputs, _ = ripple_adder(n_bits)
    finally:
        globals()["delay_line"] = real_delay_line
    bare_neurons = len(bare["names"]) - bare["n_inputs"]
    depth = 20
    raster = run_net(bare, number_trains(x, y, n_bits), steps=len(x) + depth)
    best = max(int((read_sums(raster, bare, bare_outputs, np.arange(len(x)) + d) == x + y).sum()) for d in range(depth + 1))
    hold = 12
    raster = run_net(bare, number_trains(x, y, n_bits, hold), steps=len(x) * hold)
    held = int((read_sums(raster, bare, bare_outputs, np.arange(len(x)) * hold + hold - 1) == x + y).sum())
    print(f"\n[4] the same adder with every relay removed ({bare_neurons} neurons)")
    print(f"    a new pair on every step, read at the best single delay: {best} of {len(x)} sums correct")
    print(f"    every pair held still for {hold} steps, read on the last one: {held} of {len(x)} sums correct")
    print("    the logic is intact; what the relays add is that signals from the same addition arrive together")


if __name__ == "__main__":
    main()

