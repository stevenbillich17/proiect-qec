from helpers import explain_generator_taps
class ConvolutionalEncoder:
    """
    Convolutional encoder class.
    Configurable for constraint length and generator polynomials.
    Supports listeners for step-by-step encoding details and message lifecycle events.
    """

    def __init__(self, constraint_length, generators_octal):
        if not isinstance(constraint_length, int) or constraint_length < 1:
            raise ValueError("Constraint length (K) must be a positive integer.")
        self.constraint_length = constraint_length
        
        if not isinstance(generators_octal, list) or not all(isinstance(g, str) for g in generators_octal):
            raise ValueError("Generators must be a list of octal strings.")
        self.generators_octal = generators_octal

        try:
            self.binary_generators = [
                [int(bit) for bit in bin(int(g, 8))[2:].zfill(self.constraint_length)]
                for g in self.generators_octal
            ]
        except ValueError:
            raise ValueError("Invalid octal string found in generators.")

        self.memory_size = self.constraint_length - 1
        self.listeners = []
        self._reset_encoding_state() # Initialize memory, buffer, and message state

    def _reset_encoding_state(self):
        """
        Resets memory, encoded data buffer, and current message tracking state.
        Called during full reset or when a new message is loaded.
        """
        self.memory = [0] * self.memory_size
        self.encoded_data_buffer = [] 
        self.current_message = []
        self.current_message_bit_idx = 0 
        self.is_message_loaded = False
        self.is_encoding_complete_for_current_message = False

    def reset(self):
        """
        Performs a full reset of the encoder.
        Clears memory, encoded data buffer, any loaded message, and resets message progress.
        Encoder configuration (K, G) and listeners are preserved.
        """
        self._reset_encoding_state()
        self._notify_listeners({
            "type": "ENCODER_RESET",
            "status": "Full encoder state reset (memory, buffer, message progress)."
        })

    def add_listener(self, listener):
        if not callable(listener):
            raise ValueError("Listener must be a callable function or method.")
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener):
        try:
            self.listeners.remove(listener)
        except ValueError:
            pass 

    def _notify_listeners(self, event_data):
        for listener in self.listeners:
            listener(event_data)

    def load_message(self, message_bits):
        """
        Loads a new message for step-by-step or batch encoding.
        This operation will reset the encoder's memory, clear any previous
        encoded data buffer, and set up the new message for processing.
        Args:
            message_bits (list): A list of bits (0s or 1s) for the new message.
        """
        if not isinstance(message_bits, list) or not all(bit in [0,1] for bit in message_bits):
            raise ValueError("Message bits must be a list of 0s or 1s.")
        
        self._reset_encoding_state() # Ensures a clean start for the new message

        self.current_message = list(message_bits) # Store a copy
        self.is_message_loaded = True
        
        event_data = {
            "type": "MESSAGE_LOADED",
            "message_loaded": list(self.current_message),
            "message_length": len(self.current_message)
        }
        if not self.current_message: # Empty message
            self.is_encoding_complete_for_current_message = True # Considered complete
            event_data["status"] = "Empty message loaded; encoding considered complete."
        else:
            event_data["status"] = "Message loaded; ready for encoding."
        
        self._notify_listeners(event_data)

    def _calculate_generator_output(self, effective_register_contents, generator_taps_binary):
        output_bit = 0
        involved_source_bits_details = []
        for i in range(self.constraint_length):
            is_tap_active = (generator_taps_binary[i] == 1)
            bit_value_at_tap_position = effective_register_contents[i]
            if is_tap_active:
                output_bit ^= bit_value_at_tap_position
                tap_desc_str = "current input bit" if i == 0 else f"memory cell M_{i-1}"
                involved_source_bits_details.append({
                    "value_tapped": bit_value_at_tap_position,
                    "tap_index_in_effective_register": i, 
                    "tap_description": tap_desc_str 
                })
        return output_bit, involved_source_bits_details

    def encode_bit(self, input_bit):
        """
        Encodes a single input bit. Internal method used by go_to_next_step or direct stream processing.
        Updates memory, generates output bits for this step, appends to buffer, and notifies listeners.
        Args:
            input_bit (int): The current input bit (0 or 1).
        Returns:
            list: The output bits (one for each generator) generated for this input_bit.
        """
        if input_bit not in [0, 1]:
            raise ValueError("Input bit must be 0 or 1.")

        memory_before_update = list(self.memory) 
        effective_register_contents = [input_bit] + self.memory
        current_step_output_bits = []
        per_generator_calculation_details = []

        for g_idx, gen_taps_binary in enumerate(self.binary_generators):
            gen_octal_str = self.generators_octal[g_idx]
            output_for_this_gen, involved_bits = self._calculate_generator_output(
                effective_register_contents, gen_taps_binary
            )
            current_step_output_bits.append(output_for_this_gen)
            per_generator_calculation_details.append({
                "generator_octal": gen_octal_str,
                "generator_binary": ''.join(map(str, gen_taps_binary)),
                "output_bit": output_for_this_gen,
                "involved_source_bits": involved_bits 
            })

        if self.memory_size > 0:
            self.memory = [input_bit] + self.memory[:-1]
        memory_after_update = list(self.memory)

        event_data = {
            "type": "ENCODE_STEP",
            "input_bit": input_bit,
            "memory_before": memory_before_update,
            "effective_register_contents": effective_register_contents,
            "generators_details": per_generator_calculation_details,
            "output_bits_for_step": list(current_step_output_bits), 
            "memory_after": memory_after_update,
            "accumulated_encoded_data_total": list(self.encoded_data_buffer)
        }
        self._notify_listeners(event_data)
        self.encoded_data_buffer.extend(current_step_output_bits)
        return current_step_output_bits

    def go_to_next_step(self):
        """
        Processes the next bit of the loaded message.
        Returns:
            dict: {'status': str, 'output_bits': list or None, 'details': str or None}
        """
        if not self.is_message_loaded:
            return {"status": "no_message_loaded", "output_bits": None, "details": "No message loaded. Call load_message() first."}
        if self.is_encoding_complete_for_current_message:
            return {"status": "already_complete", "output_bits": None, "details": "Current message encoding is already complete."}

        if self.current_message_bit_idx < len(self.current_message):
            input_bit = self.current_message[self.current_message_bit_idx]
            output_bits_for_step = self.encode_bit(input_bit)
            self.current_message_bit_idx += 1

            if self.current_message_bit_idx == len(self.current_message):
                self.is_encoding_complete_for_current_message = True
                self._notify_listeners({
                    "type": "ENCODING_COMPLETE",
                    "message_processed": list(self.current_message),
                    "full_encoded_output": list(self.encoded_data_buffer)
                })
                return {"status": "encoding_complete", "output_bits": output_bits_for_step, "details": f"Processed bit {self.current_message_bit_idx}/{len(self.current_message)}. Encoding complete."}
            else:
                return {"status": "step_processed", "output_bits": output_bits_for_step, "details": f"Processed bit {self.current_message_bit_idx}/{len(self.current_message)}."}
        else: # Should be caught by is_encoding_complete flag, but as safeguard
            self.is_encoding_complete_for_current_message = True
            return {"status": "already_complete", "output_bits": None, "details": "Internal: Index out of bounds but not marked complete."}

    def encode(self, message_bits):
        """
        Encodes a sequence of message bits (batch processing).
        This loads the message (which resets internal state including memory)
        and processes it fully.
        Args:
            message_bits (list): A list of bits (0s or 1s) to encode.
        Returns:
            list: The complete encoded sequence for the given message bits.
        """
        self.load_message(message_bits) 

        if not self.current_message: # Empty message handled by load_message
            return list(self.get_current_encoded_data())

        while not self.is_current_message_fully_encoded():
            step_result = self.go_to_next_step()
            if step_result["status"] not in ["step_processed", "encoding_complete"]:
                raise RuntimeError(f"Unexpected state during batch encode: {step_result['details']}")
            
        return list(self.get_current_encoded_data())

    def get_current_encoded_data(self):
        return list(self.encoded_data_buffer)

    def get_current_memory(self):
        return list(self.memory)

    def get_current_message(self):
        return list(self.current_message) if self.is_message_loaded else None

    def get_current_message_pointer(self):
        return self.current_message_bit_idx

    def is_current_message_fully_encoded(self):
        return self.is_encoding_complete_for_current_message

# --- Main Execution ---
if __name__ == '__main__':
    print("===== Encoder Setup Explanation =====")
    target_constraint_length = 3  # K=3
    target_generators_octal = ["7", "5"] # G1=111, G2=101
    
    explain_generator_taps(target_generators_octal[0], target_constraint_length)
    explain_generator_taps(target_generators_octal[1], target_constraint_length)

    print("\n===== Step-by-Step Encoding with Listener (Class-based) =====")
    encoder = ConvolutionalEncoder(constraint_length=target_constraint_length, 
                                   generators_octal=target_generators_octal)

    def comprehensive_listener(event_data):
        event_type = event_data.get("type", "UNKNOWN_EVENT")
        print(f"\n🔔 LISTENER NOTIFICATION ({event_type}) 🔔")

        if event_type == "ENCODE_STEP":
            print(f"  Input Bit              : {event_data['input_bit']}")
            # Determine M_K-2 string for memory display
            mem_labels = []
            if target_constraint_length > 1: # encoder.constraint_length can be used too
                 mem_labels = [f"M_{i}" for i in range(target_constraint_length - 2, -1, -1)] # M_K-2 ... M_0
            
            mem_str = "N/A"
            if mem_labels: # If there's memory
                mem_str = str(event_data['memory_before']) + " (" + ' '.join(mem_labels) + ")"
            else: # No memory elements (K=1)
                mem_str = "[] (No memory elements for K=1)"

            print(f"  Memory (Before)        : {mem_str}")
            print(f"  Effective Register     : {event_data['effective_register_contents']} ([Input, M_0,...M_{target_constraint_length-2}])")
            
            print("  Generator Calculations:")
            for i, gen_detail in enumerate(event_data['generators_details']):
                print(f"    Generator {i+1} ('{gen_detail['generator_octal']}' / '{gen_detail['generator_binary']}'):")
                tapped_values_for_xor = [str(item['value_tapped']) for item in gen_detail['involved_source_bits']]
                xor_sum_str = " XOR ".join(tapped_values_for_xor) if tapped_values_for_xor else "0"
                if not gen_detail['involved_source_bits'] and gen_detail['generator_binary'].count('1') == 0 :
                     xor_sum_str = "0 (generator polynomial is zero)"
                print(f"      XOR Sum              : {xor_sum_str}")
                print(f"      Output Bit           : {gen_detail['output_bit']}")
                
            print(f"  Output for this step   : {event_data['output_bits_for_step']}")
            mem_after_str = "N/A"
            if mem_labels:
                 mem_after_str = str(event_data['memory_after']) + " (" + ' '.join(mem_labels) + ")"
            else:
                 mem_after_str = "[] (No memory elements for K=1)"
            print(f"  Memory (After)         : {mem_after_str}")
            print(f"  Accumulated Encoded    : {event_data['accumulated_encoded_data_total']} (before this step's output)")
        
        elif event_type == "MESSAGE_LOADED":
            print(f"  Message Loaded         : {event_data['message_loaded']}")
            print(f"  Length                 : {event_data['message_length']}")
            print(f"  Status                 : {event_data['status']}")
        
        elif event_type == "ENCODING_COMPLETE":
            print(f"  Original Message       : {event_data['message_processed']}")
            print(f"  Full Encoded Output    : {event_data['full_encoded_output']}")
            print(f"  Status                 : Encoding for the current message is complete.")
        
        elif event_type == "ENCODER_RESET":
            print(f"  Status                 : {event_data['status']}")
        
        else:
            print(f"  Raw Event Data         : {event_data}")
        print(f"🔔 --- End Notification --- 🔔")

    encoder.add_listener(comprehensive_listener)

    message_to_encode = [1, 0, 1, 1, 0, 0] # Example message with K-1 flush bits for K=3
    expected_output = [1,1, 1,0, 0,0, 0,1, 0,1, 1,1] # For K=3, G=["7","5"], M=101100

    print(f"\nLoading message for step-by-step encoding: {message_to_encode}")
    encoder.load_message(message_to_encode)

    step_count = 0
    while not encoder.is_current_message_fully_encoded():
        step_count += 1
        print(f"\n>>> Calling go_to_next_step() - Step {step_count} of {len(message_to_encode)} <<<")
        step_info = encoder.go_to_next_step()
        print(f">>> go_to_next_step() call returned status: '{step_info['status']}'")
        if step_info['output_bits'] is not None:
            print(f"    Output for this step (from return): {step_info['output_bits']}")
        # Listener will print more details if ENCODE_STEP or ENCODING_COMPLETE was triggered

    final_encoded_output_stepwise = encoder.get_current_encoded_data()
    print(f"\nFinal Encoded Output (from step-by-step): {final_encoded_output_stepwise}")
    print(f"Expected Output                           : {expected_output}")
    assert final_encoded_output_stepwise == expected_output, "Step-by-step output mismatch!"
    