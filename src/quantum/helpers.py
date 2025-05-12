# Helper function to explain generator taps (unchanged from previous version)
def explain_generator_taps(generator_octal, constraint_length):
    """
    Explains which shift register positions are tapped by a generator polynomial.
    Args:
        generator_octal (str): The generator polynomial in octal.
        constraint_length (int): The constraint length K of the convolutional code.
                                 (K-1 is the number of memory elements)
    """
    print(f"\n--- Explaining Generator '{generator_octal}' (Constraint Length K={constraint_length}) ---")
    try:
        binary_generator_str = bin(int(generator_octal, 8))[2:].zfill(constraint_length)
    except ValueError:
        print(f"Error: Invalid octal string '{generator_octal}'.")
        return
    print(f"Octal '{generator_octal}' -> Binary (length K={constraint_length}): '{binary_generator_str}'")
    register_elements = ["u (current input)"]
    if constraint_length > 1:
        for i in range(constraint_length - 1):
            register_elements.append(f"M_{i} (memory cell {i})")
    elif constraint_length == 0:
        print("Error: Constraint length must be at least 1.")
        return
    print("\nThe binary string g = 'g_0 g_1 ... g_{K-1}' determines the connections:")
    for i in range(constraint_length):
        element_tapped_desc = register_elements[i] if i < len(register_elements) else "N/A"
        print(f"  g_{i} (bit {i} of binary string, from left) connects to: {element_tapped_desc}")
    print("\nVisualizing taps:")
    line_tap_indices   = "TAP INDEX:    "
    line_binary_values = "BINARY VALUE: "
    line_tapped_el     = "TAPPED EL.:   "
    for i in range(constraint_length):
        g_idx_str = f"g_{i}"
        bin_val_str = str(binary_generator_str[i])
        el_str = register_elements[i] if i < len(register_elements) else "Error"
        max_w = max(len(el_str), len(g_idx_str), 5) + 2
        line_tap_indices += g_idx_str.center(max_w)
        line_binary_values += bin_val_str.center(max_w)
        line_tapped_el += el_str.center(max_w)
    print(line_tap_indices)
    print(line_binary_values)
    print(line_tapped_el)
    tapped_connections_desc_list = []
    for i, bit_char in enumerate(binary_generator_str):
        if bit_char == '1':
            element_name = register_elements[i] if i < len(register_elements) else "Error"
            tapped_connections_desc_list.append(f"g_{i} (connects to {element_name})")
    if not tapped_connections_desc_list:
        print(f"\nSummary: Generator '{generator_octal}' (binary '{binary_generator_str}') has no active taps. It will always output 0.")
    else:
        print(f"\nSummary: Generator '{generator_octal}' (binary '{binary_generator_str}') taps: {', '.join(tapped_connections_desc_list)}.")
    print("--- End of Explanation ---")
