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
