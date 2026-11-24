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

---

Built on Deep-ML.
