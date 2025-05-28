# src/quantum/quantum_encoder.py
# from src.quantum.helpers import explain_generator_taps # Keep if you use it, else remove
from src.quantum.puncturer import Puncturer, PREDEFINED_PUNCTURING_SCHEMES

class ConvolutionalEncoder:
    """
    Convolutional encoder class.
    Configurable for constraint length and generator polynomials.
    Supports listeners for step-by-step encoding details and message lifecycle events.
    Integrates optional puncturing.
    """

    def __init__(self, constraint_length, generators_octal):
        if not isinstance(constraint_length, int) or constraint_length < 1:
            raise ValueError("Constraint length (K) must be a positive integer.")
        self.constraint_length = constraint_length
        
        if not isinstance(generators_octal, list) or not all(isinstance(g, str) for g in generators_octal):
            raise ValueError("Generators must be a list of octal strings.")
        if not generators_octal:
             raise ValueError("Generators list cannot be empty.")
        self.generators_octal = generators_octal

        self.binary_generators = [] # Initialize before try block
        try:
            for g_idx, g_oct_str in enumerate(self.generators_octal): # Iterate with index for better error msg
                self.binary_generators.append(
                    [int(bit) for bit in bin(int(g_oct_str, 8))[2:].zfill(self.constraint_length)]
                )
        except ValueError as e:
            problematic_generator = self.generators_octal[g_idx] if 'g_idx' in locals() else "unknown"
            raise ValueError(f"Invalid octal string ('{problematic_generator}') found in generators. Error: {e}")
        
        if not self.binary_generators and self.generators_octal: # Should be caught by try-except
             raise RuntimeError("Failed to parse binary generators from octal strings, though octal list was not empty.")

        self.num_output_streams = len(self.binary_generators)
        if self.num_output_streams == 0: # Should be caught by generators_octal check or binary_generators parsing
            raise ValueError("Encoder must have at least one generator, resulting in output streams.")

        self.memory_size = self.constraint_length - 1
        self.listeners = []
        
        # Puncturing related attributes
        self.puncturer = None 
        self.apply_puncturing_on_the_fly = True 
        self.unpunctured_symbols_for_current_message = []

        self._reset_encoding_state() # Initialize memory, buffer, and message state

    def _reset_encoding_state(self):
        """
        Resets memory, encoded data buffer, and current message tracking state.
        Called during full reset or when a new message is loaded.
        """
        self.memory = [0] * self.memory_size
        self.encoded_data_buffer = [] 
        self.unpunctured_symbols_for_current_message = [] # For batch puncturing mode
        self.current_message = []
        self.current_message_bit_idx = 0 
        self.is_message_loaded = False
        self.is_encoding_complete_for_current_message = False
        if self.puncturer and self.apply_puncturing_on_the_fly:
            self.puncturer.reset_period_tracker()


    def reset(self):
        """
        Performs a full reset of the encoder.
        Encoder configuration (K, G, listeners, puncturer, puncturing mode) are preserved.
        Only dynamic encoding state (memory, buffers, message progress) is reset.
        """
        self._reset_encoding_state() # This handles puncturer period index if on-the-fly
        self._notify_listeners({
            "type": "ENCODER_RESET",
            "status": "Encoder dynamic state reset (memory, buffer, message progress)."
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

    # --- Puncturing Specific Methods ---
    def set_puncturing_application_mode(self, on_the_fly: bool):
        changed = self.apply_puncturing_on_the_fly != on_the_fly
        self.apply_puncturing_on_the_fly = on_the_fly
        if self.apply_puncturing_on_the_fly and self.puncturer:
            self.puncturer.reset_period_tracker()
        if changed:
            self._notify_listeners({
                "type": "PUNCTURING_MODE_CHANGED",
                "on_the_fly": self.apply_puncturing_on_the_fly
            })

    def set_puncturer(self, puncturing_scheme_key=None, custom_matrix=None, custom_label="Custom"):
        old_puncturer_info = self.puncturer.get_info() if self.puncturer else None
        
        if puncturing_scheme_key and puncturing_scheme_key.upper() != "NONE" and puncturing_scheme_key in PREDEFINED_PUNCTURING_SCHEMES:
            scheme = PREDEFINED_PUNCTURING_SCHEMES[puncturing_scheme_key]
            if scheme["mother_code_output_streams"] != self.num_output_streams:
                raise ValueError(
                    f"Puncturing scheme '{scheme['label']}' expects {scheme['mother_code_output_streams']} "
                    f"mother code streams, but encoder has {self.num_output_streams}."
                )
            self.puncturer = Puncturer(scheme["matrix"], scheme["label"])
        elif custom_matrix:
            if len(custom_matrix) != self.num_output_streams:
                 raise ValueError(
                    f"Custom puncturing matrix rows ({len(custom_matrix)}) "
                    f"must match encoder output streams ({self.num_output_streams})."
                )
            self.puncturer = Puncturer(custom_matrix, custom_label)
        else: 
            self.puncturer = None
        
        if self.puncturer and self.apply_puncturing_on_the_fly:
            self.puncturer.reset_period_tracker()

        new_puncturer_info = self.puncturer.get_info() if self.puncturer else None
        
        config_changed = False
        if (old_puncturer_info is None and new_puncturer_info is not None) or \
           (old_puncturer_info is not None and new_puncturer_info is None):
            config_changed = True
        elif old_puncturer_info and new_puncturer_info and \
             (old_puncturer_info["matrix"] != new_puncturer_info["matrix"] or \
              old_puncturer_info["label"] != new_puncturer_info["label"]): # Check label too
            config_changed = True
        
        if config_changed:
            self._notify_listeners({
                "type": "PUNCTURER_CONFIG_CHANGED",
                "active": bool(self.puncturer),
                "scheme_label": self.puncturer.rate_label if self.puncturer else "None",
                "puncturing_matrix": self.puncturer.puncturing_matrix if self.puncturer else None
            })
        return self.puncturer.get_info() if self.puncturer else None

    # --- Core Encoding Logic ---
    def load_message(self, message_bits):
        if not isinstance(message_bits, list) or not all(bit in [0,1] for bit in message_bits):
            raise ValueError("Message bits must be a list of 0s or 1s.")
        
        self._reset_encoding_state() 

        self.current_message = list(message_bits) 
        self.is_message_loaded = True
        
        event_data = {
            "type": "MESSAGE_LOADED",
            "message_loaded": list(self.current_message),
            "message_length": len(self.current_message)
        }
        if not self.current_message: 
            self.is_encoding_complete_for_current_message = True 
            event_data["status"] = "Empty message loaded; encoding considered complete."
            # If batch mode and empty message, ensure final buffer is correct
            if self.puncturer and not self.apply_puncturing_on_the_fly:
                self.encoded_data_buffer = self.puncturer.puncture_stream(
                    self.unpunctured_symbols_for_current_message # which will be empty
                ) # Result is []
        else:
            event_data["status"] = "Message loaded; ready for encoding."
        
        self._notify_listeners(event_data)

    def _calculate_generator_output(self, effective_register_contents, generator_taps_binary):
        # This is your original _calculate_generator_output, it's fine.
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

    def _encode_single_input_bit_raw(self, input_bit):
        """Helper to get unpunctured bits and state changes for one input bit."""
        memory_before_update = list(self.memory) 
        effective_register_contents = [input_bit] + self.memory
        unpunctured_symbol_for_step = [] # Renamed from current_step_output_bits for clarity
        per_generator_calculation_details = []

        for g_idx, gen_taps_binary in enumerate(self.binary_generators):
            gen_octal_str = self.generators_octal[g_idx]
            output_for_this_gen, involved_bits = self._calculate_generator_output(
                effective_register_contents, gen_taps_binary
            )
            unpunctured_symbol_for_step.append(output_for_this_gen)
            per_generator_calculation_details.append({
                "generator_octal": gen_octal_str,
                "generator_binary": ''.join(map(str, gen_taps_binary)),
                "output_bit": output_for_this_gen,
                "involved_source_bits": involved_bits 
            })

        if self.memory_size > 0:
            self.memory = [input_bit] + self.memory[:-1]
        memory_after_update = list(self.memory)
        
        return (unpunctured_symbol_for_step, memory_before_update, 
                effective_register_contents, per_generator_calculation_details, memory_after_update)


    def encode_bit(self, input_bit): # This was your original encode_bit, now adapted
        if input_bit not in [0, 1]:
            raise ValueError("Input bit must be 0 or 1.")

        (unpunctured_symbol, mem_before, eff_reg, 
         gen_details, mem_after) = self._encode_single_input_bit_raw(input_bit)
        
        # Always store the unpunctured symbol for potential batch puncturing or logging
        self.unpunctured_symbols_for_current_message.append(list(unpunctured_symbol))

        punctured_bits_for_step = []
        accumulated_before_this_step = list(self.encoded_data_buffer) # Capture before modification

        if self.puncturer and self.apply_puncturing_on_the_fly:
            punctured_bits_for_step = self.puncturer.puncture_symbol(unpunctured_symbol)
            self.encoded_data_buffer.extend(punctured_bits_for_step)
        elif not self.puncturer: # No puncturer at all
            punctured_bits_for_step = list(unpunctured_symbol) # "Punctured" is same as unpunctured
            self.encoded_data_buffer.extend(unpunctured_symbol)
        # If batch mode (self.puncturer and not self.apply_puncturing_on_the_fly),
        # punctured_bits_for_step remains empty for this step's event, buffer not changed here.
        
        # Determine period index used for this step if on-the-fly
        idx_used = -1
        if self.puncturer and self.apply_puncturing_on_the_fly:
            # current_period_index was advanced by puncture_symbol.
            # So, the index *used* for this step was the one *before* it advanced.
            idx_used = self.puncturer.current_period_index -1 
            if idx_used < 0: # Wrapped around
                idx_used = self.puncturer.period - 1

        event_data = {
            "type": "ENCODE_STEP",
            "input_bit": input_bit,
            "memory_before": mem_before,
            "effective_register_contents": eff_reg,
            "generators_details": gen_details, # Based on unpunctured calculation
            "unpunctured_output_for_step": list(unpunctured_symbol),
            "punctured_output_for_step": list(punctured_bits_for_step), # Empty if batch mode for this step
            "output_bits_for_step": list(punctured_bits_for_step), # For compatibility, reflects what's added to buffer this step
            "memory_after": mem_after,
            "is_puncturing_active": bool(self.puncturer),
            "puncturing_on_the_fly": self.apply_puncturing_on_the_fly,
            "puncturer_period_index_before_step": idx_used,
            "accumulated_encoded_data_total": accumulated_before_this_step
        }
        self._notify_listeners(event_data)
        
        # Return what was (or would be, if on-the-fly) added to buffer this step
        return punctured_bits_for_step


    def go_to_next_step(self):
        if not self.is_message_loaded:
            return {"status": "no_message_loaded", "output_bits": None, "details": "No message loaded. Call load_message() first."}
        if self.is_encoding_complete_for_current_message:
            return {"status": "already_complete", "output_bits": None, "details": "Current message encoding is already complete."}

        if self.current_message_bit_idx < len(self.current_message):
            input_bit = self.current_message[self.current_message_bit_idx]
            # encode_bit handles listeners and returns the (potentially empty if batch mode) punctured bits for this step
            output_bits_for_this_step_event = self.encode_bit(input_bit)
            self.current_message_bit_idx += 1

            if self.current_message_bit_idx == len(self.current_message): # Message fully processed
                self.is_encoding_complete_for_current_message = True
                
                # If batch puncturing, apply it now to the collected unpunctured symbols
                if self.puncturer and not self.apply_puncturing_on_the_fly:
                    # Ensure puncturer's internal period tracker is reset before batch processing the stream
                    self.puncturer.reset_period_tracker()
                    final_punctured_output = self.puncturer.puncture_stream(
                        self.unpunctured_symbols_for_current_message
                    )
                    self.encoded_data_buffer = list(final_punctured_output) # Overwrite buffer with batch result
                # If on-the-fly or no puncturer, encoded_data_buffer is already correct.

                self._notify_listeners({
                    "type": "ENCODING_COMPLETE",
                    "message_processed": list(self.current_message),
                    "full_encoded_output": list(self.encoded_data_buffer), # Final buffer content
                    "full_unpunctured_output_pairs": list(self.unpunctured_symbols_for_current_message),
                    "is_puncturing_active": bool(self.puncturer),
                    "puncturing_on_the_fly": self.apply_puncturing_on_the_fly,
                    "puncturing_details": self.puncturer.get_info() if self.puncturer else None
                })
                return {"status": "encoding_complete", "output_bits": output_bits_for_this_step_event, "details": f"Processed bit {self.current_message_bit_idx}/{len(self.current_message)}. Encoding complete."}
            else:
                return {"status": "step_processed", "output_bits": output_bits_for_this_step_event, "details": f"Processed bit {self.current_message_bit_idx}/{len(self.current_message)}."}
        else: 
            self.is_encoding_complete_for_current_message = True
            return {"status": "already_complete", "output_bits": None, "details": "Internal: Index out of bounds but not marked complete."}

    def encode(self, message_bits, puncturing_scheme_key=None, custom_matrix=None, apply_on_the_fly=True):
        """
        Encodes a sequence of message bits (batch processing).
        Optionally sets a puncturer and its application mode for this encoding run.
        """
        if puncturing_scheme_key or custom_matrix: # Set or clear puncturer
            self.set_puncturer(puncturing_scheme_key, custom_matrix)
        # Set application mode regardless of whether a new puncturer was set,
        # it might be using an existing puncturer with a new mode.
        self.set_puncturing_application_mode(apply_on_the_fly)
        
        self.load_message(message_bits) # This resets state including puncturer period if on-the-fly

        if not self.current_message: # Empty message
            # load_message handles ENCODING_COMPLETE notification for empty message
            # and correct buffer state based on puncturing mode.
            return list(self.get_current_encoded_data())

        while not self.is_current_message_fully_encoded():
            step_result = self.go_to_next_step()
            # go_to_next_step will handle the final batch puncturing if mode is batch
            if step_result["status"] not in ["step_processed", "encoding_complete"]:
                # This should ideally not happen if logic is correct
                raise RuntimeError(f"Unexpected state during batch encode: {step_result['details']}")
            
        return list(self.get_current_encoded_data())

    # --- Getters ---
    def get_current_encoded_data(self):
        return list(self.encoded_data_buffer)

    def get_current_unpunctured_data_pairs(self):
        return list(self.unpunctured_symbols_for_current_message)

    def get_current_memory(self):
        return list(self.memory)

    def get_current_message(self):
        return list(self.current_message) if self.is_message_loaded else None

    def get_current_message_pointer(self):
        return self.current_message_bit_idx

    def is_current_message_fully_encoded(self):
        return self.is_encoding_complete_for_current_message
    
    def get_puncturer_info(self):
        return self.puncturer.get_info() if self.puncturer else None

    def get_puncturing_application_mode(self):
        return self.apply_puncturing_on_the_fly

# --- Main Execution (Example, keeping your original for testing your structure) ---
if __name__ == '__main__':
    target_constraint_length = 3
    target_generators_octal = ["7", "5"]

    print("\n===== Step-by-Step Encoding with Listener (Class-based) =====")
    encoder = ConvolutionalEncoder(constraint_length=target_constraint_length, 
                                   generators_octal=target_generators_octal)

    def comprehensive_listener(event_data):
        event_type = event_data.get("type", "UNKNOWN_EVENT")
        print(f"\n🔔 LISTENER NOTIFICATION ({event_type}) 🔔")

        if event_type == "ENCODE_STEP":
            print(f"  Input Bit                : {event_data['input_bit']}")
            mem_labels = [f"M_{i}" for i in range(encoder.constraint_length - 2, -1, -1)] if encoder.constraint_length > 1 else []
            mem_str = str(event_data['memory_before']) + " (" + ' '.join(mem_labels) + ")" if mem_labels else "[]"
            print(f"  Memory (Before)          : {mem_str}")
            print(f"  Effective Register       : {event_data['effective_register_contents']}")
            print(f"  Unpunctured Output (Step): {event_data['unpunctured_output_for_step']}")
            if event_data['is_puncturing_active']:
                print(f"  Puncturing Mode          : {'On-the-fly' if event_data['puncturing_on_the_fly'] else 'Batch (at end)'}")
                if event_data['puncturing_on_the_fly']:
                    print(f"  Punctured Output (Step)  : {event_data['punctured_output_for_step']} (idx: {event_data['puncturer_period_index_before_step']})")
                else:
                    print(f"  Punctured Output (Step)  : (deferred to end of message)")
            print(f"  Output for this step     : {event_data['output_bits_for_step']}") # What was added to buffer for this step
            mem_after_str = str(event_data['memory_after']) + " (" + ' '.join(mem_labels) + ")" if mem_labels else "[]"
            print(f"  Memory (After)           : {mem_after_str}")
            print(f"  Accumulated Encoded      : {event_data['accumulated_encoded_data_total']} (before this step's output added)")
        
        elif event_type == "MESSAGE_LOADED":
            print(f"  Message Loaded           : {event_data['message_loaded']}")
            print(f"  Length                   : {event_data['message_length']}")
            print(f"  Status                   : {event_data['status']}")
        
        elif event_type == "ENCODING_COMPLETE":
            print(f"  Original Message         : {event_data['message_processed']}")
            print(f"  Full Unpunctured Pairs   : {event_data['full_unpunctured_output_pairs']}")
            if event_data['is_puncturing_active']:
                print(f"  Puncturing Details       : {event_data['puncturing_details']}")
                print(f"  Puncturing Mode Used     : {'On-the-fly' if event_data['puncturing_on_the_fly'] else 'Batch'}")
            print(f"  Full Encoded Output (Final): {event_data['full_encoded_output']}")
        
        elif event_type == "ENCODER_RESET": print(f"  Status: {event_data['status']}")
        elif event_type == "PUNCTURER_CONFIG_CHANGED": print(f"  Puncturer set to: {event_data['scheme_label']}")
        elif event_type == "PUNCTURING_MODE_CHANGED": print(f"  Puncturing mode: {'On-the-fly' if event_data['on_the_fly'] else 'Batch'}")
        else: print(f"  Raw Event Data: {event_data}")
        print(f"🔔 --- End Notification --- 🔔")

    encoder.add_listener(comprehensive_listener)

    message_to_encode = [1, 0, 1, 1] # Example message K=3
    # K=3, G=["7","5"] = (111, 101)
    # 1 (00) -> 111,101 -> 1,1 (mem 10)
    # 0 (10) -> 011,010 -> 1,1 (mem 01)
    # 1 (01) -> 101,100 -> 0,0 (mem 10) -> Mistake in manual trace, G1=1^0^1=0, G2=1^0^0=1 -> (0,1)
    # Corrected manual trace K=3, G=(111,101) for input 1011
    # Input | Mem (M0,M1) | EffReg (I,M0,M1) | G1 (I^M0^M1) | G2 (I^M1) | Output | Next Mem
    # 1     | 00          | 100              | 1            | 1         | (1,1)  | 10
    # 0     | 10          | 010              | 1            | 0         | (1,0)  | 01
    # 1     | 01          | 101              | 0            | 1         | (0,1)  | 10
    # 1     | 10          | 110              | 0            | 1         | (0,1)  | 11
    # Unpunctured: 11100101
    expected_unpunctured = [1,1, 1,0, 0,1, 0,1] 

    # Test 1: No Puncturing
    print(f"\n--- Test: No Puncturing ---")
    encoder.set_puncturer("NONE") # or encoder.set_puncturer(None)
    encoder.set_puncturing_application_mode(True) # Mode doesn't matter much if no puncturer
    output1 = encoder.encode(list(message_to_encode)) # Use batch encode
    print(f"Output (No Puncturing)  : {output1}")
    assert output1 == expected_unpunctured

    # Test 2: Rate 2/3 Puncturing, On-the-fly
    # Matrix [[1,0],[1,1]], Period 2. Input bits 1,0,1,1
    # 1. (1,1) -> Punc idx 0 -> [1,1]
    # 2. (1,0) -> Punc idx 1 -> [0]
    # 3. (0,1) -> Punc idx 0 -> [0,1]
    # 4. (0,1) -> Punc idx 1 -> [1]
    # Expected: 110011
    expected_2_3_otf = [1,1, 0, 0,1, 1]
    print(f"\n--- Test: Rate 2/3 Puncturing, On-the-fly ---")
    encoder.set_puncturer("2/3")
    encoder.set_puncturing_application_mode(True)
    output2 = encoder.encode(list(message_to_encode))
    print(f"Output (2/3 On-the-fly): {output2}")
    assert output2 == expected_2_3_otf

    # Test 3: Rate 2/3 Puncturing, Batch
    print(f"\n--- Test: Rate 2/3 Puncturing, Batch ---")
    # Puncturer is already set to 2/3
    encoder.set_puncturing_application_mode(False)
    output3 = encoder.encode(list(message_to_encode)) # Batch mode in encode method
    print(f"Output (2/3 Batch)     : {output3}")
    assert output3 == expected_2_3_otf # Should be same for this case if Puncturer.puncture_stream is correct

    # Test with your original example from the question:
    # K=3, G=["7","5"], M=101100 (this looks like a message + flush bits for K=3)
    # Original message 1011, flush 00. Total input 101100.
    # Unpunctured: 111001010111
    message_k3_flush = [1,0,1,1,0,0]
    expected_unpunctured_k3_flush = [1,1, 1,0, 0,1, 0,1, 0,1, 1,1]
    print(f"\n--- Test: K=3 with flush, No Puncturing ---")
    encoder.set_puncturer("NONE")
    output_k3_flush = encoder.encode(list(message_k3_flush))
    print(f"Output K3 Flush No Punc: {output_k3_flush}")
    assert output_k3_flush == expected_unpunctured_k3_flush