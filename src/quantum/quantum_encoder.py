# src/quantum/quantum_encoder.py
# from src.quantum.helpers import explain_generator_taps # Assuming this exists
from src.quantum.puncturer import Puncturer, PREDEFINED_PUNCTURING_SCHEMES # Import Puncturer

class ConvolutionalEncoder:
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
        
        self.num_output_streams = len(self.binary_generators) # e.g., 2 for rate 1/2 base

        self.memory_size = self.constraint_length - 1
        self.listeners = []
        self.puncturer = None # Initialize puncturer as None
        self._reset_encoding_state()

    def _reset_encoding_state(self):
        self.memory = [0] * self.memory_size
        self.encoded_data_buffer = [] 
        self.unpunctured_data_for_current_message = [] # To store all unpunctured pairs
        self.current_message = []
        self.current_message_bit_idx = 0 
        self.is_message_loaded = False
        self.is_encoding_complete_for_current_message = False
        if self.puncturer:
            self.puncturer.reset_period_tracker() # Reset puncturer's internal state too

    def reset(self):
        self._reset_encoding_state()
        # self.puncturer remains as is, or could be reset to None by a separate clear_puncturer call
        self._notify_listeners({
            "type": "ENCODER_RESET",
            "status": "Full encoder state reset (memory, buffer, message progress)."
        })

    def set_puncturer(self, puncturing_scheme_key=None, custom_matrix=None, custom_label="Custom"):
        """
        Sets or clears the puncturer for the encoder.
        Args:
            puncturing_scheme_key (str, optional): Key from PREDEFINED_PUNCTURING_SCHEMES.
                                                   If "NONE" or None, puncturer is cleared.
            custom_matrix (list of lists, optional): A custom puncturing matrix.
                                                     Ignored if puncturing_scheme_key is valid.
            custom_label (str, optional): Label for custom matrix.
        """
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
            # Basic validation for custom_matrix compatibility
            if len(custom_matrix) != self.num_output_streams:
                 raise ValueError(
                    f"Custom puncturing matrix rows ({len(custom_matrix)}) "
                    f"must match encoder output streams ({self.num_output_streams})."
                )
            self.puncturer = Puncturer(custom_matrix, custom_label)
        else: # Clear puncturer
            self.puncturer = None
        
        if self.puncturer:
            self.puncturer.reset_period_tracker()

        new_puncturer_info = self.puncturer.get_info() if self.puncturer else None
        
        # Notify only if there's a change in puncturing status or configuration
        if (old_puncturer_info is None and new_puncturer_info is not None) or \
           (old_puncturer_info is not None and new_puncturer_info is None) or \
           (old_puncturer_info and new_puncturer_info and old_puncturer_info["matrix"] != new_puncturer_info["matrix"]):
            self._notify_listeners({
                "type": "PUNCTURER_CONFIG_CHANGED",
                "active": bool(self.puncturer),
                "scheme_label": self.puncturer.rate_label if self.puncturer else "None",
                "puncturing_matrix": self.puncturer.puncturing_matrix if self.puncturer else None
            })
        return self.puncturer.get_info() if self.puncturer else None


    def add_listener(self, listener): # Unchanged
        if not callable(listener):
            raise ValueError("Listener must be a callable function or method.")
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener): # Unchanged
        try:
            self.listeners.remove(listener)
        except ValueError:
            pass 

    def _notify_listeners(self, event_data): # Unchanged
        for listener in self.listeners:
            listener(event_data)

    def load_message(self, message_bits): # Unchanged in core logic, _reset_encoding_state handles puncturer period
        if not isinstance(message_bits, list) or not all(bit in [0,1] for bit in message_bits):
            raise ValueError("Message bits must be a list of 0s or 1s.")
        
        self._reset_encoding_state() 

        self.current_message = list(message_bits)
        self.is_message_loaded = True
        self.unpunctured_data_for_current_message = [] # Reset for new message
        
        event_data = {
            "type": "MESSAGE_LOADED",
            "message_loaded": list(self.current_message),
            "message_length": len(self.current_message)
        }
        # ... rest of load_message is the same ...
        if not self.current_message:
            self.is_encoding_complete_for_current_message = True
            event_data["status"] = "Empty message loaded; encoding considered complete."
        else:
            event_data["status"] = "Message loaded; ready for encoding."
        self._notify_listeners(event_data)


    def _calculate_generator_output(self, effective_register_contents, generator_taps_binary): # Unchanged
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
        """
        Performs raw encoding for one bit, returning unpunctured output and details.
        This is the core encoding logic before puncturing.
        """
        memory_before_update = list(self.memory) 
        effective_register_contents = [input_bit] + self.memory
        unpunctured_output_for_step = []
        per_generator_calculation_details = []

        for g_idx, gen_taps_binary in enumerate(self.binary_generators):
            gen_octal_str = self.generators_octal[g_idx]
            output_for_this_gen, involved_bits = self._calculate_generator_output(
                effective_register_contents, gen_taps_binary
            )
            unpunctured_output_for_step.append(output_for_this_gen)
            per_generator_calculation_details.append({
                "generator_octal": gen_octal_str,
                "generator_binary": ''.join(map(str, gen_taps_binary)),
                "output_bit": output_for_this_gen,
                "involved_source_bits": involved_bits 
            })

        if self.memory_size > 0:
            self.memory = [input_bit] + self.memory[:-1]
        memory_after_update = list(self.memory)
        
        return (unpunctured_output_for_step, memory_before_update, 
                effective_register_contents, per_generator_calculation_details, memory_after_update)

    def encode_bit(self, input_bit):
        """
        Encodes a single input bit, applies puncturing if active, and notifies listeners.
        Args:
            input_bit (int): The current input bit (0 or 1).
        Returns:
            list: The *punctured* output bits generated for this input_bit.
        """
        if input_bit not in [0, 1]:
            raise ValueError("Input bit must be 0 or 1.")

        (unpunctured_bits, mem_before, eff_reg, 
         gen_details, mem_after) = self._encode_single_input_bit_raw(input_bit)
        
        self.unpunctured_data_for_current_message.append(list(unpunctured_bits)) # Store unpunctured pair

        punctured_bits_for_step = []
        if self.puncturer:
            punctured_bits_for_step = self.puncturer.puncture_symbol(unpunctured_bits)
        else:
            punctured_bits_for_step = list(unpunctured_bits) # No puncturing, keep all

        event_data = {
            "type": "ENCODE_STEP",
            "input_bit": input_bit,
            "memory_before": mem_before,
            "effective_register_contents": eff_reg,
            "generators_details": gen_details, # Details of unpunctured calculation
            "unpunctured_output_for_step": list(unpunctured_bits), # Explicitly add unpunctured
            "punctured_output_for_step": list(punctured_bits_for_step),
            "output_bits_for_step": list(punctured_bits_for_step), # For backward compatibility if something expects this key
            "memory_after": mem_after,
            "is_puncturing_active": bool(self.puncturer),
            "puncturer_period_index_before_step": self.puncturer.current_period_index -1 if self.puncturer else -1, # Index used for this step
            "accumulated_encoded_data_total": list(self.encoded_data_buffer) # Before this step's output added
        }
        if self.puncturer and event_data["puncturer_period_index_before_step"] < 0 : # Correct for wrap around
             event_data["puncturer_period_index_before_step"] = self.puncturer.period -1


        self._notify_listeners(event_data)
        self.encoded_data_buffer.extend(punctured_bits_for_step) # Add punctured bits to main buffer
        return punctured_bits_for_step

    def go_to_next_step(self): # Logic remains largely the same, calls the new encode_bit
        if not self.is_message_loaded:
            return {"status": "no_message_loaded", "output_bits": None, "details": "No message loaded. Call load_message() first."}
        if self.is_encoding_complete_for_current_message:
            return {"status": "already_complete", "output_bits": None, "details": "Current message encoding is already complete."}

        if self.current_message_bit_idx < len(self.current_message):
            input_bit = self.current_message[self.current_message_bit_idx]
            # 'encode_bit' now handles puncturing internally and returns punctured bits
            output_bits_for_step = self.encode_bit(input_bit) 
            self.current_message_bit_idx += 1

            if self.current_message_bit_idx == len(self.current_message):
                self.is_encoding_complete_for_current_message = True
                self._notify_listeners({
                    "type": "ENCODING_COMPLETE",
                    "message_processed": list(self.current_message),
                    "full_encoded_output": list(self.encoded_data_buffer), # This is now the punctured output
                    "full_unpunctured_output_pairs": list(self.unpunctured_data_for_current_message), # Store all raw pairs
                    "is_puncturing_active": bool(self.puncturer),
                    "puncturing_details": self.puncturer.get_info() if self.puncturer else None
                })
                return {"status": "encoding_complete", "output_bits": output_bits_for_step, "details": f"Processed bit {self.current_message_bit_idx}/{len(self.current_message)}. Encoding complete."}
            else:
                return {"status": "step_processed", "output_bits": output_bits_for_step, "details": f"Processed bit {self.current_message_bit_idx}/{len(self.current_message)}."}
        else: 
            self.is_encoding_complete_for_current_message = True
            return {"status": "already_complete", "output_bits": None, "details": "Internal: Index out of bounds but not marked complete."}

    def encode(self, message_bits, puncturing_scheme_key=None, custom_matrix=None): # Allow setting puncturer during batch encode
        """
        Encodes a sequence of message bits (batch processing).
        Optionally sets a puncturer for this encoding run.
        """
        if puncturing_scheme_key or custom_matrix: # If puncturing provided for this batch call
            self.set_puncturer(puncturing_scheme_key, custom_matrix)
        
        self.load_message(message_bits) 

        if not self.current_message:
            return list(self.get_current_encoded_data())

        while not self.is_current_message_fully_encoded():
            step_result = self.go_to_next_step()
            if step_result["status"] not in ["step_processed", "encoding_complete"]:
                raise RuntimeError(f"Unexpected state during batch encode: {step_result['details']}")
            
        return list(self.get_current_encoded_data())

    # Getters remain mostly the same
    def get_current_encoded_data(self): # This now returns (potentially) punctured data
        return list(self.encoded_data_buffer)

    def get_current_unpunctured_data_pairs(self): # New getter for UI or analysis
        return list(self.unpunctured_data_for_current_message)

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

# --- Main Execution (Example update) ---
if __name__ == '__main__':
    target_constraint_length = 7 
    target_generators_octal = ["171", "133"] 

    encoder = ConvolutionalEncoder(constraint_length=target_constraint_length, 
                                   generators_octal=target_generators_octal)
    # ... (listener setup can be similar, but now check for puncturing fields in event_data) ...
    
    def simple_listener(event_data):
        print(f"Event: {event_data['type']}")
        if event_data['type'] == 'ENCODE_STEP':
            print(f"  Input: {event_data['input_bit']}")
            print(f"  Unpunctured: {event_data['unpunctured_output_for_step']}")
            if event_data['is_puncturing_active']:
                print(f"  Punctured  : {event_data['punctured_output_for_step']} (idx {event_data['puncturer_period_index_before_step']})")
            print(f"  Mem After: {event_data['memory_after']}")
        elif event_data['type'] == 'ENCODING_COMPLETE':
            print(f"  Final Punctured Output: {event_data['full_encoded_output']}")
            if event_data['is_puncturing_active']:
                 print(f"  Puncturing Details: {event_data['puncturing_details']}")
        elif event_data['type'] == 'PUNCTURER_CONFIG_CHANGED':
            print(f"  Puncturer Active: {event_data['active']}, Label: {event_data['scheme_label']}")


    encoder.add_listener(simple_listener)
    message_to_encode = [1,0,1,1,0] # 5 bits

    print("\n--- Encoding WITHOUT Puncturing ---")
    encoder.set_puncturer("NONE") # Or encoder.set_puncturer()
    encoder.load_message(list(message_to_encode))
    while not encoder.is_current_message_fully_encoded():
        encoder.go_to_next_step()
    print(f"Final (unpunctured) output: {encoder.get_current_encoded_data()}")


    print("\n--- Encoding WITH Rate 2/3 Puncturing ---")
    encoder.set_puncturer("2/3")
    encoder.load_message(list(message_to_encode)) # Reload message to reset state
    while not encoder.is_current_message_fully_encoded():
        encoder.go_to_next_step()
    final_punctured_output = encoder.get_current_encoded_data()
    print(f"Final (2/3 punctured) output: {final_punctured_output}")
    # Expected for Rate 2/3 (Matrix: [[1,0],[1,1]], Period 2, 3 ones) for 5 input bits:
    # Input 1 -> Unp (v1a,v2a) -> Punc [v1a, v2a] (idx 0 of matrix [[1],[1]])
    # Input 0 -> Unp (v1b,v2b) -> Punc [v2b]       (idx 1 of matrix [[0],[1]])
    # Input 1 -> Unp (v1c,v2c) -> Punc [v1c, v2c] (idx 0 of matrix [[1],[1]])
    # Input 1 -> Unp (v1d,v2d) -> Punc [v2d]       (idx 1 of matrix [[0],[1]])
    # Input 0 -> Unp (v1e,v2e) -> Punc [v1e, v2e] (idx 0 of matrix [[1],[1]])
    # Total output bits = 2+1+2+1+2 = 8 bits. If message was 4 bits, output is 2+1+2+1 = 6 bits.

    # Example using batch encode with puncturing
    print("\n--- Batch Encoding WITH Rate 4/5 Puncturing ---")
    batch_output = encoder.encode(list(message_to_encode), puncturing_scheme_key="4/5")
    print(f"Final (4/5 punctured) output from batch: {batch_output}")