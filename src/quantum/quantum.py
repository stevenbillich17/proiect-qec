def explain_generator_taps(generator_octal, constraint_length):
    """
    Explains which shift register positions are tapped by a generator polynomial.

    Args:
        generator_octal (str): The generator polynomial in octal.
        constraint_length (int): The constraint length of the convolutional code.
    """
    print(f"\nGenerator '{generator_octal}':")
    # Convert octal to binary and pad
    binary_generator = bin(int(generator_octal, 8))[2:].zfill(constraint_length)
    print(f"Convert to binary (padded to constraint length {constraint_length}): {generator_octal} (octal) = {binary_generator} (binary)")
    print("Reading the binary from right to left, a '1' indicates a tap at that position in the shift register.")
    print(f"Binary: {' '.join(list(binary_generator))}")
    print(f"Pos'n:  {' '.join([str(i) for i in reversed(range(constraint_length))])}")

    tapped_positions = [i for i, bit in enumerate(reversed(binary_generator)) if bit == '1']
    print(f"Taps are at positions {', '.join(map(str, tapped_positions))} of the shift register.")


class ConvolutionalEncoder:
    """
    Convolutional encoder.
 """

def __init__(self, constraint_length, generator):
    """
    Initializes the ConvolutionalEncoder.

    Args:
    constraint_length (int): The constraint length of the convolutional code.
    generator (list): Generator polynomials in octal (list of strings).
    """
    self.constraint_length = constraint_length
    self.generator = generator
    self.memory = [0] * (constraint_length - 1)
    self.encoded_data = []
    # Convert octal generator polynomials to binary
    self.binary_generator = [
        [int(bit) for bit in bin(int(g, 8))[2:].zfill(constraint_length)]
        for g in generator
    ]

def encode_step(self, input_bit, listener=None):
    """
    Performs one step of the convolutional encoding process.

        Args:
            input_bit (int): The current input bit (0 or 1).
            listener (function, optional): A function to call with step details.

        Returns:
            list: The output bits for this step.
        """
    input_with_memory = [input_bit] + self.memory[:]
    output = []
    involved_bits_for_step = []

    for generator_index, generator_taps in enumerate(self.binary_generator):
        generator_output_bit, involved_bits_for_generator = self._calculate_generator_output(
            input_with_memory, generator_taps, listener
        )
        output.append(generator_output_bit)
        involved_bits_for_step.append(involved_bits_for_generator)

    self.encoded_data.extend(output)

    # Update memory\
    self.memory = [input_bit] + self.memory[:-1]
    return output

def _calculate_generator_output(self, input_with_memory, generator_taps, listener=None):
    """
    Calculates the output of a single generator for the current input and memory.

        Args:
            input_with_memory (list): The current input bit combined with memory.
            generator_taps (list): The binary representation of the generator polynomial.
            listener (function, optional): A function to call with step details (per generator).

        Returns:
            tuple: A tuple containing the output bit for this generator and a list of involved bits.
        """
    generator_output_bit = 0
    involved_bits_for_generator = []
    for position in range(self.constraint_length):
        # XOR the product of the input bit at the current position and the generator tap
        generator_output_bit ^= input_with_memory[position] * generator_taps[position]

        # If the generator tap is 1, this bit is involved in the calculation
        if generator_taps[position] == 1:
            involved_bits_for_generator.append((input_with_memory[position], position))

    if listener:
        # Report per generator for clarity
        # Note: This listener call is more suitable within the encode_step loop if needed per generator
        pass # Re-evaluate where generator listener is most useful. Currently moved listener call to encode_step

    return generator_output_bit, involved_bits_for_generator

def get_encoded_data(self):
    return self.encoded_data


# The original convolutional_encoder function is kept for backward compatibility if needed,
# but the main block will now use the class.
def convolutional_encoder(data, generator, constraint_length, listener=None):
    """
    Convolutional encoder (function based - kept for reference/backward compatibility).

    Args:
        data (list): Input binary data (list of 0s and 1s).
        generator (list): Generator polynomials in octal (list of strings).
        constraint_length (int): The constraint length of the convolutional code.

    Returns:
        list: Encoded data (list of 0s and 1s).
    """
    memory = [0] * (constraint_length - 1)
    encoded_data = []

    # Convert octal generator polynomials to binary
    binary_generator = [
        [int(bit) for bit in bin(int(g, 8))[2:].zfill(constraint_length)]
        for g in generator
    ]

    for i, bit in enumerate(data):
        input_with_memory = [bit] + memory[:]
        output = []
        for j, gen in enumerate(binary_generator):
            output_bit = 0
            involved_bits_with_positions = []
            for k in range(constraint_length):
                output_bit ^= input_with_memory[k] * gen[k]
                if gen[k] == 1:
                    involved_bits_with_positions.append((input_with_memory[k], k))
            output.append(output_bit)
            if listener:
                listener(
                    selected_bit=bit,
                    memory=memory[:],
                    involved_bits=involved_bits_with_positions,
                    generator_output_bit=output_bit,
                    accumulated_encoded_data=encoded_data[:]
                )
        encoded_data.extend(output)
        memory = [bit] + memory[:-1]


    return encoded_data


if __name__ == '__main__':
    def step_listener(selected_bit, memory, involved_bits, generator_output_bit, accumulated_encoded_data):
        print(f"  Step:")
        print(f"    Selected bit: {selected_bit}")
        input_with_memory = [selected_bit] + memory
        print(f"    Input with memory: {memory} + [{selected_bit}] = {input_with_memory} (Conceptual: this is what the generator taps are applied to)")
        print(f"    Involved bits (value, position): {involved_bits}")
        print(f"    Generator output bit: {generator_output_bit}")
        print(f"    Accumulated encoded data: {accumulated_encoded_data}")

    data = [1, 0, 1, 1, 0, 0, 1] # Example 7-bit data
    constraint_length = 4
    generator = ["3", "5"]
    encoded = ConvolutionalEncoder(constraint_length, generator).encode_step(data, generator, constraint_length, listener=step_listener)
    print(f"\nExample 2 (with listener):")
    print(f"Original data: {data}, Constraint length: {constraint_length}, Generators: {generator}")
    print(f"Encoded data: {encoded}")
