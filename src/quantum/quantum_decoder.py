import math # For float('inf')
import random # For testing

from quantum_encoder import ConvolutionalEncoder

# --- Simplified Viterbi Decoder for K=7, G=["171", "133"] ---
class ViterbiDecoderK7G171_133_Simple:
    """
    Simplified Hard-Decision Viterbi Decoder specifically for:
    - Constraint Length (K) = 7
    - Rate = 1/2
    - Generator Polynomials (octal) = ["171", "133"]
    This version has listeners and most step-by-step UI helpers removed
    for a more straightforward understanding of the core batch decoding process.
    """

    def __init__(self):
        # Fixed parameters for this specific code
        self.constraint_length = 7
        self.generators_octal = ["171", "133"]
        
        self.binary_generators = []
        for g_oct_str in self.generators_octal:
            binary_str = bin(int(g_oct_str, 8))[2:].zfill(self.constraint_length)
            self.binary_generators.append([int(bit) for bit in binary_str])
        
        self.memory_size = self.constraint_length - 1  # 6
        self.num_states = 2**self.memory_size      # 64
        
        # Precompute state transitions (fixed for the code)
        self._state_transitions_table = [{} for _ in range(self.num_states)] 
        self._precompute_state_transitions()

        self._reset_decoding_state() # Initialize dynamic decoding variables

    def _reset_decoding_state(self):
        """Resets dynamic state variables for a new decoding operation."""
        self.current_received_sequence = []
        self.num_original_message_bits = 0
        self.current_trellis_stage_idx = 0
        self.T_stages_total = 0

        self.path_metrics = []
        self.traceback_pointers = []
        
        self.is_sequence_loaded = False
        self.is_acs_complete = False
        self.is_traceback_complete = False
        self.decoded_message_final = []
        # No listener notification here

    def _state_to_memory_bits(self, state_int):
        return [int(bit) for bit in bin(state_int)[2:].zfill(self.memory_size)]

    def _memory_bits_to_state(self, memory_bits_list):
        return int("".join(map(str, memory_bits_list)), 2)

    def _precompute_state_transitions(self):
        for current_state_int in range(self.num_states):
            current_memory_content = self._state_to_memory_bits(current_state_int)
            for input_bit_u in [0, 1]:
                effective_register = [input_bit_u] + current_memory_content
                output_c0 = 0
                for i in range(self.constraint_length):
                    if self.binary_generators[0][i] == 1: output_c0 ^= effective_register[i]
                output_c1 = 0
                for i in range(self.constraint_length):
                    if self.binary_generators[1][i] == 1: output_c1 ^= effective_register[i]
                expected_output_pair = (output_c0, output_c1)
                next_memory_content = ([input_bit_u] + current_memory_content)[:self.memory_size]
                next_state_int = self._memory_bits_to_state(next_memory_content)
                self._state_transitions_table[current_state_int][input_bit_u] = (next_state_int, expected_output_pair)

    def _hamming_distance(self, bits1_tuple, bits2_tuple):
        dist = 0
        if bits1_tuple[0] != bits2_tuple[0]: dist += 1
        if bits1_tuple[1] != bits2_tuple[1]: dist += 1
        return dist

    def _load_received_sequence(self, received_sequence_bits, num_original_message_bits):
        if not isinstance(received_sequence_bits, list) or not all(bit in [0,1] for bit in received_sequence_bits):
            raise ValueError("Received sequence must be a list of 0s or 1s.")
        if len(received_sequence_bits) % 2 != 0:
            raise ValueError("Received sequence length must be a multiple of 2.")
        if not isinstance(num_original_message_bits, int) or num_original_message_bits < 0:
            raise ValueError("Number of original message bits must be a non-negative integer.")

        self._reset_decoding_state() 

        self.current_received_sequence = list(received_sequence_bits)
        self.num_original_message_bits = num_original_message_bits
        self.T_stages_total = len(self.current_received_sequence) // 2
        
        if self.T_stages_total == 0:
            self.is_sequence_loaded = True
            self.is_acs_complete = True
            self.is_traceback_complete = True
            return

        self.path_metrics = [0.0] + [float('inf')] * (self.num_states - 1)
        self.traceback_pointers = [[(0,0) for _ in range(self.num_states)] for _ in range(self.T_stages_total)]
        self.is_sequence_loaded = True
        
    def _go_to_next_acs_step(self):
        # This is now an internal method, called by decode()
        if not self.is_sequence_loaded:
            return {"status": "error", "details": "Internal: Sequence not loaded before ACS."}
        if self.is_acs_complete:
            return {"status": "error", "details": "Internal: ACS already complete."}
        
        t = self.current_trellis_stage_idx
        current_received_pair = (self.current_received_sequence[t*2], self.current_received_sequence[t*2 + 1])
        
        path_metrics_at_stage_start = list(self.path_metrics)
        new_path_metrics_for_next_stage = [float('inf')] * self.num_states

        # Check for each possible next state if there is a "way" to get to it
        # based on the previous (current) state on both possible inputs (0 or 1)
        for next_s_idx in range(self.num_states):
            for prev_s_idx in range(self.num_states):
                if path_metrics_at_stage_start[prev_s_idx] == float('inf'):
                    continue

                for input_bit_u in [0, 1]:
                    potential_next_state, expected_out = self._state_transitions_table[prev_s_idx][input_bit_u]
                    
                    # Check if there is a way to the next_state => potential_next is next_s
                    if potential_next_state == next_s_idx:
                        # if next_state can be formed based on previous compute distance of received message with what the transition offers
                        branch_metric = self._hamming_distance(current_received_pair, expected_out)
                        candidate_total_metric = path_metrics_at_stage_start[prev_s_idx] + branch_metric
                        
                        if candidate_total_metric < new_path_metrics_for_next_stage[next_s_idx]:
                            new_path_metrics_for_next_stage[next_s_idx] = candidate_total_metric
                            self.traceback_pointers[t][next_s_idx] = (prev_s_idx, input_bit_u)
            
        self.path_metrics = new_path_metrics_for_next_stage

        self.current_trellis_stage_idx += 1
        if self.current_trellis_stage_idx == self.T_stages_total:
            self.is_acs_complete = True
            return {"status": "acs_complete"}
        else:
            return {"status": "acs_step_processed"}

    def _perform_traceback(self, assume_zero_terminated=True):
        if not self.is_sequence_loaded or not self.is_acs_complete:
             return {"status": "error", "decoded_message": [], "details": "Internal: Load/ACS issue before traceback."}
        if self.is_traceback_complete:
            return {"status": "success", "decoded_message": list(self.decoded_message_final)}

        if self.T_stages_total == 0:
            self.decoded_message_final = []
            self.is_traceback_complete = True
            return {"status": "success", "decoded_message": []}

        decoded_sequence_with_tail = [0] * self.T_stages_total
        final_state_at_trellis_end = 0
        min_final_metric = float('inf')

        if assume_zero_terminated:
            # Typically, for a zero-terminated sequence, we expect to end in state 0.
            # If state 0 is unreachable (inf metric), it indicates a problem,
            # but we might still pick the overall minimum metric state as a fallback.
            final_state_at_trellis_end = 0
            min_final_metric = self.path_metrics[0]
            if min_final_metric == float('inf'):
                print("Warning: Assumed zero termination, but state 0 has infinite metric. Searching for best available final state.")
                for s_idx, metric_val in enumerate(self.path_metrics):
                    if metric_val < min_final_metric: # This will find a state if any has non-inf metric
                        min_final_metric = metric_val
                        final_state_at_trellis_end = s_idx
        else: # Not assuming zero termination, find the state with the overall minimum path metric
            for s_idx, metric_val in enumerate(self.path_metrics):
                if metric_val < min_final_metric:
                    min_final_metric = metric_val
                    final_state_at_trellis_end = s_idx
        
        if min_final_metric == float('inf'):
            print("Error: All final states have infinite path metrics. Decoding will likely fail or be arbitrary.")
            final_state_at_trellis_end = 0

        current_state_in_traceback = final_state_at_trellis_end
        for t_idx in range(self.T_stages_total - 1, -1, -1):
            prev_state, input_bit = self.traceback_pointers[t_idx][current_state_in_traceback]
            decoded_sequence_with_tail[t_idx] = input_bit
            current_state_in_traceback = prev_state
        
        self.decoded_message_final = decoded_sequence_with_tail[:self.num_original_message_bits]
        self.is_traceback_complete = True
        return {"status": "success", "decoded_message": list(self.decoded_message_final)}

    def decode(self, received_sequence_bits, num_original_message_bits, assume_zero_terminated=True):
        """
        Performs full Viterbi decoding on the received sequence.
        Args:
            received_sequence_bits (list): The sequence of 0s and 1s received from the channel.
            num_original_message_bits (int): The number of actual message bits (excluding tail bits).
            assume_zero_terminated (bool): If True, assumes the encoder was flushed with zeros,
                                           so traceback prefers ending in state 0.
        Returns:
            list: The decoded sequence of message bits.
        """
        self._load_received_sequence(received_sequence_bits, num_original_message_bits)
        
        if self.T_stages_total == 0:
            return []

        while not self.is_acs_complete:
            result = self._go_to_next_acs_step()
            if result["status"] == "error":
                raise RuntimeError(f"Error during ACS phase: {result.get('details','')}")

        traceback_result = self._perform_traceback(assume_zero_terminated)
        if traceback_result["status"] == "success":
            return traceback_result["decoded_message"]
        else:
            raise RuntimeError(f"Error during traceback phase: {traceback_result.get('details','')}")



def run_random_example():
    # 1. Define an original 128-bit message
    num_message_bits = 128
    random.seed(40) # For reproducibility
    original_message = [random.randint(0, 1) for _ in range(num_message_bits)]
    print(f"Original Message ({num_message_bits} bits): \n{''.join(map(str,original_message))}\n")

    # 2. Setup Convolutional Encoder (K=7, G=["171", "133"])
    # This assumes ConvolutionalEncoder class is available and works.
    # The import `from src.quantum.quantum_encoder import ConvolutionalEncoder` might need adjustment
    try:
        encoder = ConvolutionalEncoder(constraint_length=7, generators_octal=["171", "133"])
    except Exception as e:
        print(f"Error initializing ConvolutionalEncoder: {e}")
        exit()

    # 3. Add K-1 tail bits (zeros) for flushing the encoder
    num_tail_bits = encoder.constraint_length - 1 # K-1 = 6
    message_with_tail = original_message + [0] * num_tail_bits
    
    # 4. Encode the tailed message
    encoded_sequence = encoder.encode(message_with_tail)
    print(f"Encoded Sequence ({len(encoded_sequence)} bits, Rate 1/2): \n{''.join(map(str,encoded_sequence[:256]))}...\n") # Print first 256 bits

    received_sequence = list(encoded_sequence) # Make a copy
    
    # CAN BE UNCOMMENTED TO ADD NOISE
    # ber = 0.01 # Bit Error Rate of 1%
    # num_errors_to_introduce = int(len(received_sequence) * ber)
    # error_positions = random.sample(range(len(received_sequence)), num_errors_to_introduce)
    # for pos in error_positions:
    #     received_sequence[pos] = 1 - received_sequence[pos] # Flip the bit
    
    # print(f"Introduced {num_errors_to_introduce} errors.")
    # print(f"Received Sequence (with noise, {len(received_sequence)} bits): \n{''.join(map(str,received_sequence[:256]))}...\n")
    
    #For this example, let's run with NO noise first to verify correctness
    print("Running decoder with NO noise on the encoded sequence.\n")


    # 6. Instantiate the simplified Viterbi Decoder
    decoder = ViterbiDecoderK7G171_133_Simple()

    # 7. Decode the (possibly noisy) sequence
    # We tell the decoder the number of *original* message bits, not including the tail.
    # assume_zero_terminated=True is important because we added tail bits to the encoder.
    try:
        decoded_message = decoder.decode(received_sequence, num_message_bits, assume_zero_terminated=True)
        print(f"Decoded Message ({len(decoded_message)} bits): \n{''.join(map(str,decoded_message))}\n")

        # 8. Compare decoded message with the original
        if decoded_message == original_message:
            print("SUCCESS: Decoded message matches the original message!")
    except Exception as e:
        print(f"An error occurred during decoding: {e}")
        import traceback
        traceback.print_exc()


def run_short_example():
        # 1. Define an original 128-bit message
    num_message_bits = 128
    random.seed(42) # For reproducibility
    original_message = [random.randint(0, 1) for _ in range(num_message_bits)]
    print(f"Original Message ({num_message_bits} bits): \n{''.join(map(str,original_message))}\n")

    # 2. Setup Convolutional Encoder (K=7, G=["171", "133"])
    # This assumes ConvolutionalEncoder class is available and works.
    # The import `from src.quantum.quantum_encoder import ConvolutionalEncoder` might need adjustment
    try:
        encoder = ConvolutionalEncoder(constraint_length=7, generators_octal=["171", "133"])
    except Exception as e:
        print(f"Error initializing ConvolutionalEncoder: {e}")
        exit()

    # 3. Add K-1 tail bits (zeros) for flushing the encoder
    num_tail_bits = encoder.constraint_length - 1 # K-1 = 6
    message_with_tail = original_message + [0] * num_tail_bits
    
    # 4. Encode the tailed message
    encoded_sequence = encoder.encode(message_with_tail)
    print(f"Encoded Sequence ({len(encoded_sequence)} bits, Rate 1/2): \n{''.join(map(str,encoded_sequence[:256]))}...\n") # Print first 256 bits

    received_sequence = list(encoded_sequence) # Make a copy
    decoder = ViterbiDecoderK7G171_133_Simple()
    
    print("\n--- Example with a short, known (hardcoded) sequence for demonstration ---")
    # Say, the first 20 bits of the encoded sequence for the 128-bit message (10 original bits represented)
    if len(encoded_sequence) >= 20: # Ensure we have enough bits
        hardcoded_received_segment = encoded_sequence[:20] # Takes 10 input bits to produce 20 output bits
        num_orig_bits_for_segment = 10

        print(f"Hardcoded received segment (first 20 bits of previous encoding): {''.join(map(str, hardcoded_received_segment))}")
        print(f"Number of original message bits this segment represents: {num_orig_bits_for_segment}")

        try:
            decoded_segment = decoder.decode(hardcoded_received_segment, num_orig_bits_for_segment, assume_zero_terminated=False)
            print(f"Decoded segment: {''.join(map(str,decoded_segment))}")
            original_segment_part = original_message[:num_orig_bits_for_segment]
            print(f"Original part  : {''.join(map(str,original_segment_part))}")

            if decoded_segment == original_segment_part:
                print("SUCCESS: Decoded segment matches the original part of the message!")
            else:
                print("FAILURE: Decoded segment does NOT match. This can be expected for short, non-terminated segments.")
        except Exception as e:
            print(f"An error occurred during segment decoding: {e}")
    
# --- Main Execution Example ---
if __name__ == '__main__':
    run_random_example()
    #run_short_example()
