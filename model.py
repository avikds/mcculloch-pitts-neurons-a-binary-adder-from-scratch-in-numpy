"""
McCulloch-Pitts Neurons: A Binary Adder from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fire
import numpy as np

def fire(active, E, I, theta):
    """Which neurons fire next: enough active excitatory synapses (>= theta) and no active inhibitory synapse."""

    # Convert inputs to NumPy arrays without modifying the originals.
    active = np.asarray(active)
    E = np.asarray(E)
    I = np.asarray(I)
    theta = np.asarray(theta)

    # Calculate the total excitatory input received by each neuron.
    # E[i, j] represents the number of excitatory synapses from neuron j to neuron i.
    excitatory_input = E @ active

    # Check whether any active neuron sends an inhibitory signal to each neuron.
    inhibitory_input = np.any(
        (I != 0) & (active[np.newaxis, :] != 0),
        axis=1
    )

    # A neuron fires if its excitatory input reaches its threshold
    # and it receives no active inhibitory signal.
    firing = (excitatory_input >= theta) & (~inhibitory_input)

    # Return an integer NumPy array containing 1 (fires) or 0 (silent).
    return firing.astype(int)

# Step 2 - compile_net
def compile_net(inputs, neurons):
    """Turn input names and {name: {"theta", "exc", "inh"}} into a dict with names, index, E, I, theta, n_inputs."""

    # Store input neurons first, followed by the remaining neurons
    # in the order in which they appear in the dictionary.
    input_names = list(inputs)
    neuron_names = list(neurons.keys())
    names = input_names + neuron_names

    # Ensure that every neuron has a unique name.
    if len(names) != len(set(names)):
        raise ValueError("Duplicate neuron names are not allowed.")

    # Create a mapping from neuron names to their integer positions.
    index = {name: i for i, name in enumerate(names)}

    # Determine the total number of neurons and input neurons.
    n = len(names)
    n_inputs = len(input_names)

    # Initialize the excitatory and inhibitory connectivity matrices.
    # E[i, j] stores the number of excitatory synapses from neuron j to neuron i.
    # I[i, j] indicates whether neuron j inhibits neuron i.
    E = np.zeros((n, n), dtype=int)
    I = np.zeros((n, n), dtype=bool)

    # Initialize the threshold vector.
    # Input neurons have a default threshold of 1.
    theta = np.ones(n, dtype=int)

    # Process each neuron and configure its connections.
    for target_name, config in neurons.items():

        target_idx = index[target_name]

        # Set and validate the neuron's firing threshold.
        threshold = config["theta"]

        if threshold < 1 or int(threshold) != threshold:
            raise ValueError("Neuron thresholds must be positive integers.")

        theta[target_idx] = threshold

        # Configure excitatory synaptic connections.
        for source_name, count in config["exc"].items():

            # Ensure that the source neuron exists.
            if source_name not in index:
                raise ValueError(f"Unknown excitatory source neuron: {source_name}")

            # Each excitatory connection must have at least one synapse.
            if count < 1 or int(count) != count:
                raise ValueError("Excitatory synapse counts must be positive integers.")

            source_idx = index[source_name]

            # Store the number of excitatory synapses.
            E[target_idx, source_idx] = count

        # Configure inhibitory synaptic connections.
        for source_name in config["inh"]:

            # Ensure that the source neuron exists.
            if source_name not in index:
                raise ValueError(f"Unknown inhibitory source neuron: {source_name}")

            source_idx = index[source_name]

            # Mark the inhibitory connection.
            I[target_idx, source_idx] = True

    # Return the compiled neural network representation.
    return {
        "names": names,
        "index": index,
        "E": E,
        "I": I,
        "theta": theta,
        "n_inputs": n_inputs
    }

# Step 3 - run_net
def run_net(net, trains, steps=None):
    """Simulate the net; return raster[t, i] = 1 if neuron i fires at time t (inputs from trains, others via fire)."""

    # Extract the network structure.
    names = net["names"]
    index = net["index"]
    E = net["E"]
    I = net["I"]
    theta = net["theta"]
    n_inputs = net["n_inputs"]

    # Identify the input neurons.
    input_names = names[:n_inputs]

    # Validate that all names in trains refer to input neurons.
    for name in trains:
        if name not in input_names:
            raise ValueError(f"'{name}' is not an input neuron.")

    # If steps is not specified, use the longest input sequence.
    if steps is None:
        steps = max(
            (len(sequence) for sequence in trains.values()),
            default=0
        )

    # Determine the total number of neurons.
    n = len(names)

    # Initialize the firing record.
    # raster[t, i] = 1 if neuron i fires at time t, otherwise 0.
    raster = np.zeros((steps, n), dtype=int)

    # Simulate the network one time step at a time.
    for t in range(steps):

        # At time t, compute the firing states of all non-input neurons
        # using the complete firing state from the previous time step.
        if t > 0:
            previous_state = raster[t - 1]

            next_state = fire(previous_state, E, I, theta)

            # Update only non-input neurons.
            raster[t, n_inputs:] = next_state[n_inputs:]

        # Set the input neurons according to their respective trains.
        # Missing input trains and exhausted sequences produce zeros.
        for name in input_names:
            neuron_idx = index[name]
            sequence = trains.get(name, [])

            if t < len(sequence):
                raster[t, neuron_idx] = sequence[t]

    # Return the complete firing history.
    return raster

# Step 4 - gate
def gate(kind, *sources):
    """{"theta", "exc", "inh"} for one neuron: "relay" (a), "or" (2+), "and" (2+), "and_not" (a, b)."""

    # Validate the gate type.
    valid_kinds = {"relay", "or", "and", "and_not"}

    if kind not in valid_kinds:
        raise ValueError(f"Invalid gate type: {kind}")

    # Relay gate: Fires one time step after its single source fires.
    if kind == "relay":

        if len(sources) != 1:
            raise ValueError("A relay gate requires exactly one source.")

        return {
            "theta": 1,
            "exc": {sources[0]: 1},
            "inh": []
        }

    # OR gate: Fires when at least one source is active.
    if kind == "or":

        if len(sources) < 2:
            raise ValueError("An OR gate requires at least two sources.")

        return {
            "theta": 1,
            "exc": {source: 1 for source in sources},
            "inh": []
        }

    # AND gate: Fires only when all sources are active.
    if kind == "and":

        if len(sources) < 2:
            raise ValueError("An AND gate requires at least two sources.")

        return {
            "theta": len(sources),
            "exc": {source: 1 for source in sources},
            "inh": []
        }

    # AND-NOT gate: Fires when the first source is active
    # and the second source is inactive.
    if kind == "and_not":

        if len(sources) != 2:
            raise ValueError("An AND-NOT gate requires exactly two sources.")

        return {
            "theta": 1,
            "exc": {sources[0]: 1},
            "inh": [sources[1]]
        }

# Step 5 - delay_line
def delay_line(neurons, src, k, name):
    """Add k relays name.1 .. name.k in a chain behind src; return the last name (src itself when k == 0)."""

    # A negative delay is not valid.
    if k < 0:
        raise ValueError("The number of delay steps cannot be negative.")

    # No relay neurons are needed when k is zero.
    if k == 0:
        return src

    # Keep track of the neuron from which the next relay receives its signal.
    previous_source = src

    # Create a chain of k relay neurons.
    for i in range(1, k + 1):

        # Generate the name of the current relay neuron.
        relay_name = f"{name}.{i}"

        # Create a relay connected to the previous neuron.
        neurons[relay_name] = gate("relay", previous_source)

        # The current relay becomes the source for the next relay.
        previous_source = relay_name

    # Return the name of the final relay neuron.
    return previous_source

# Step 6 - half_adder
def half_adder(neurons, a, b, name):
    """Add a half adder for neurons a, b; return (sum_name, carry_name), both exactly 2 steps behind the inputs."""

    # Generate names for the intermediate neurons.
    a_only = f"{name}.a_only"
    b_only = f"{name}.b_only"
    sum_name = f"{name}.sum"
    both = f"{name}.both"

    # Detect when only input 'a' is active (a AND NOT b).
    neurons[a_only] = gate("and_not", a, b)

    # Detect when only input 'b' is active (b AND NOT a).
    neurons[b_only] = gate("and_not", b, a)

    # The SUM output fires when exactly one input is active.
    # This OR gate combines the two mutually exclusive conditions.
    neurons[sum_name] = gate("or", a_only, b_only)

    # Detect when both inputs are active (a AND b).
    neurons[both] = gate("and", a, b)

    # Delay the CARRY signal by one additional time step.
    # The AND gate produces its output after 1 step,
    # and the relay adds another step, synchronizing it with SUM.
    carry_name = delay_line(
        neurons,
        both,
        1,
        f"{name}.carry"
    )

    # Return the names of the synchronized SUM and CARRY outputs.
    return sum_name, carry_name

# Step 7 - full_adder
def full_adder(neurons, a, b, cin, name):
    """Add a full adder for neurons a, b, cin; return (sum_name, cout_name), both exactly 5 steps behind the inputs."""

    # First half adder: Add inputs a and b.
    # Both outputs arrive 2 time steps after the inputs.
    first_sum, first_carry = half_adder(
        neurons, a, b, f"{name}.h1"
    )

    # Delay the carry-in by 2 steps so that it arrives
    # at the second half adder simultaneously with first_sum.
    delayed_cin = delay_line(
        neurons, cin, 2, f"{name}.cin"
    )

    # Second half adder: Add the first sum and delayed carry-in.
    # Its outputs arrive 4 time steps after the original inputs.
    second_sum, second_carry = half_adder(
        neurons, first_sum, delayed_cin, f"{name}.h2"
    )

    # Delay the first carry by 2 additional steps
    # so that it arrives simultaneously with the second carry.
    delayed_first_carry = delay_line(
        neurons, first_carry, 2, f"{name}.c1"
    )

    # The final carry-out fires when either half adder
    # produces a carry.
    # Both carry signals arrive at time step 4,
    # so the OR gate produces the output at time step 5.
    cout_name = f"{name}.cout"
    neurons[cout_name] = gate(
        "or", delayed_first_carry, second_carry
    )

    # Delay the final sum by one additional time step
    # to synchronize it with the carry-out.
    sum_name = delay_line(
        neurons, second_sum, 1, f"{name}.sum"
    )

    # Return the synchronized sum and carry-out neuron names.
    return sum_name, cout_name

# Step 8 - ripple_adder
def ripple_adder(n_bits=4):
    """Build an n_bits adder on inputs a0.., b0..; return (net, outputs, latency) with latency = 2 + 5 * (n_bits - 1)."""

    # Validate the number of bits.
    if isinstance(n_bits, (bool, np.bool_)) or not isinstance(n_bits, (int, np.integer)) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer.")

    # Create input neuron names.
    # The bits are ordered from least significant to most significant.
    inputs = (
        [f"a{i}" for i in range(n_bits)] +
        [f"b{i}" for i in range(n_bits)]
    )

    # Store the descriptions of all non-input neurons.
    neurons = {}

    # Calculate the total output latency.
    # The first half adder requires 2 steps.
    # Each additional full adder requires 5 more steps.
    latency = 2 + 5 * (n_bits - 1)

    # Store the aligned sum output names.
    sum_outputs = []

    # ---------------------------------------------------------
    # Column 0: Use a half adder for the least significant bits.
    # ---------------------------------------------------------
    first_sum, carry = half_adder(
        neurons, "a0", "b0", "bit0"
    )

    # Delay the first sum until the final output time.
    first_sum_delay = latency - 2

    aligned_sum = delay_line(
        neurons,
        first_sum,
        first_sum_delay,
        "out0"
    )

    sum_outputs.append(aligned_sum)

    # ---------------------------------------------------------
    # Remaining columns: Use full adders with synchronized inputs.
    # ---------------------------------------------------------
    for i in range(1, n_bits):

        # The carry from the previous column arrives after:
        # 2 + 5 * (i - 1) time steps.
        input_delay = 2 + 5 * (i - 1)

        # Delay the current A and B bits so that they arrive
        # simultaneously with the incoming carry.
        delayed_a = delay_line(
            neurons,
            f"a{i}",
            input_delay,
            f"bit{i}.a"
        )

        delayed_b = delay_line(
            neurons,
            f"b{i}",
            input_delay,
            f"bit{i}.b"
        )

        # Construct the full adder for the current column.
        # Its outputs arrive 5 steps after its synchronized inputs.
        current_sum, carry = full_adder(
            neurons,
            delayed_a,
            delayed_b,
            carry,
            f"bit{i}"
        )

        # Calculate how many additional steps are required
        # to align this sum with the final column's output.
        output_delay = 5 * (n_bits - 1 - i)

        # Delay the sum output to the common output time.
        aligned_sum = delay_line(
            neurons,
            current_sum,
            output_delay,
            f"out{i}"
        )

        sum_outputs.append(aligned_sum)

    # The final carry is already synchronized with the final sum bit.
    outputs = sum_outputs + [carry]

    # Compile the complete neural network.
    net = compile_net(inputs, neurons)

    # Return the compiled network, output neuron names, and latency.
    return net, outputs, latency

# Step 9 - add_stream
def add_stream(xs, ys, n_bits=4):
    """Add xs[k] + ys[k] with the neural adder, one pair per time step; return the sums as an int array."""

    # Convert the input sequences into lists for consistent processing.
    xs = list(xs)
    ys = list(ys)

    # Both input sequences must contain the same number of elements.
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")

    # Build the ripple-carry neural adder.
    # The returned latency determines when each result becomes available.
    net, outputs, latency = ripple_adder(n_bits)

    # Calculate the maximum number representable using n_bits.
    max_value = (1 << int(n_bits)) - 1

    # Validate that all input values are integers within the allowed range.
    for value in xs + ys:

        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("All input values must be integers.")

        if value < 0 or value > max_value:
            raise ValueError(
                f"Input values must be between 0 and {max_value}."
            )

    # Determine the number of input pairs.
    L = len(xs)

    # Create spike trains for every input neuron.
    # Each bit position gets its own sequence of 0s and 1s.
    trains = {}

    for bit in range(n_bits):

        # Extract bit 'bit' from every number in xs.
        trains[f"a{bit}"] = [
            (int(x) >> bit) & 1 for x in xs
        ]

        # Extract bit 'bit' from every number in ys.
        trains[f"b{bit}"] = [
            (int(y) >> bit) & 1 for y in ys
        ]

    # Run the neural network for enough time to receive
    # the outputs of the final input pair.
    raster = run_net(
        net,
        trains,
        steps=L + latency
    )

    # Initialize the output array.
    # Each element will contain the sum of one input pair.
    sums = np.zeros(L, dtype=int)

    # Read the neural network outputs for every input pair.
    for k in range(L):

        # The result corresponding to input pair k
        # becomes available after the network latency.
        output_time = k + latency

        total = 0

        # Decode the output bits using the firing raster.
        # The outputs are ordered from least significant bit
        # to most significant bit (including the final carry).
        for bit, neuron_name in enumerate(outputs):

            neuron_idx = net["index"][neuron_name]

            # Convert the neuron's firing state into
            # its corresponding binary positional value.
            total += int(raster[output_time, neuron_idx]) * (2 ** bit)

        # Store the decoded result.
        sums[k] = total

    # Return the results as an integer NumPy array.
    return sums

