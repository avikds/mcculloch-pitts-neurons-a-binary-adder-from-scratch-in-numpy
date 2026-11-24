# McCulloch-Pitts Neurons: A Binary Adder from Scratch in NumPy

A Logical Calculus of the Ideas Immanent in Nervous Activity (McCulloch and Pitts, 1943), rebuilt in NumPy. Write the paper's neuron: it is either firing or silent, it fires when enough of its excitatory synapses were active one time step earlier, and a single active inhibitory synapse stops it whatever else arrives. Simulate a net of these neurons step by step, with one synaptic delay for every neuron a signal passes through. Then build with the paper's four elementary nets (a relay that delays a signal, or, and, and and-not) and nothing else: a half adder, a full adder and a 4-bit adder. The paper contains no adder; the project applies its elements to one. Nothing is learned anywhere; every connection and threshold is set by hand. The work is in the timing. Signals that travel through different numbers of neurons arrive at different times, so you keep the early ones alive with relay neurons until the late ones catch up. The finished net of 125 neurons adds a new pair of 4-bit numbers on every time step and gives each sum 17 steps later, and the final script shows that the same wiring without its relay neurons gets most of those sums wrong.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** fire
- [x] **2.** compile_net
- [x] **3.** run_net
- [x] **4.** gate
- [x] **5.** delay_line
- [x] **6.** half_adder
- [x] **7.** full_adder
- [x] **8.** ripple_adder
- [x] **9.** add_stream

## Results

```
[1] the four elementary nets: one neuron each, and each answers one time step late
    a            ..##.#.
    b            .#.#...
    relay(a)     ...##.#
    a or b       ..###.#
    a and b      ....#..
    a and not b  ...#..#

[2] half adder, 5 neurons, a new pair of bits on every step
    a            ..##.#..
    b            .#.#....
    and neuron   ....#...
    carry        .....#..
    sum          ...##..#
    the and neuron answers after 1 step, the sum after 2; the relay holds the carry for the missing step

[3] 4-bit adder: 125 neurons, 94 of them relays, latency 17 steps
    all 256 pairs as one stream, a new pair on every step: 256 of 256 sums correct in 273 time steps
    pair  59 entered at t =  59:  3 + 11 = 14, read at t = 76
    pair 187 entered at t = 187: 11 + 11 = 22, read at t = 204
    pair 255 entered at t = 255: 15 + 15 = 30, read at t = 272

[4] the same adder with every relay removed (31 neurons)
    a new pair on every step, read at the best single delay: 88 of 256 sums correct
    every pair held still for 12 steps, read on the last one: 256 of 256 sums correct
    the logic is intact; what the relays add is that signals from the same addition arrive together
```
