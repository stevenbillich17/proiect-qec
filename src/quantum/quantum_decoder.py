import math # For float('inf')
import random # For testing

# Assuming ConvolutionalEncoder is in a sibling directory or correctly pathed for the examples
# from quantum_encoder import ConvolutionalEncoder 
# For the class itself, this import is not directly needed.

class ViterbiDecoderK7G171_133_Stepwise:
    """
    Stepwise Hard-Decision Viterbi Decoder specifically for:
    - Constraint Length (K) = 7
    - Rate = 1/2
    - Generator Polynomials (octal) = ["171", "133"]
    This version supports step-by-step execution and event listeners for UI integration.
    """

    def __init__(self):
        self.constraint_length = 7
        self.generators_octal = ["171", "133"]
        
        self.binary_generators = []
        self.memory_size = self.constraint_length - 1
        self.num_states = 2**self.memory_size
        self._state_transitions_table = [{} for _ in range(self.num_states)]

        self._precompute_state_transitions() # Done once at init

        self.listeners = []
        self._reset_decoding_state() # Initialize dynamic state and fires DECODER_RESET

    def add_listener(self, callback_fn):
        if callback_fn not in self.listeners:
            self.listeners.append(callback_fn)

    def _notify_listeners(self, event_data):
        for listener in self.listeners:
            listener(event_data)

    def _reset_decoding_state(self):
        self.current_received_sequence = []
        self.num_original_message_bits = 0
        self.current_trellis_stage_idx = 0 # Tracks the *next* stage to be processed (0 to T-1)
        self.T_stages_total = 0

        self.path_metrics_history = [] # Stores PMs at each stage: history[stage_idx_end] has PMs for end of that stage
        self.path_metrics = [] # Current path metrics (PMs at current_trellis_stage_idx)
        self.traceback_pointers = [] # List of lists of tuples: TP[stage_idx][next_state] = (prev_state, input_bit)
        
        self.is_sequence_loaded = False
        self.is_acs_complete = False
        self.is_traceback_complete = False
        self.decoded_message_final = []
        
        self._notify_listeners({
            "type": "DECODER_RESET",
            "status": "Decoder state has been reset."
        })

    def _state_to_memory_bits(self, state_int):
        return [int(bit) for bit in bin(state_int)[2:].zfill(self.memory_size)]

    def _memory_bits_to_state(self, memory_bits_list):
        if not memory_bits_list: return 0
        return int("".join(map(str, memory_bits_list)), 2)

    def _precompute_state_transitions(self):
        # Initialize binary_generators if not already done (e.g. by constructor directly)
        if not self.binary_generators:
            for g_oct_str in self.generators_octal:
                binary_str = bin(int(g_oct_str, 8))[2:].zfill(self.constraint_length)
                self.binary_generators.append([int(bit) for bit in binary_str])

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

    def load_received_sequence(self, received_sequence_bits, num_original_message_bits):
        if not isinstance(received_sequence_bits, list) or not all(bit in [0,1] for bit in received_sequence_bits):
            raise ValueError("Received sequence must be a list of 0s or 1s.")
        if len(received_sequence_bits) % 2 != 0:
            raise ValueError("Received sequence length must be a multiple of 2 for rate 1/2 code.")
        if not isinstance(num_original_message_bits, int) or num_original_message_bits < 0:
            raise ValueError("Number of original message bits must be a non-negative integer.")

        self._reset_decoding_state() # Fires DECODER_RESET

        self.current_received_sequence = list(received_sequence_bits)
        self.num_original_message_bits = num_original_message_bits
        self.T_stages_total = len(self.current_received_sequence) // 2
        
        if self.T_stages_total == 0: # Empty sequence
            self.is_sequence_loaded = True
            self.is_acs_complete = True 
            self.is_traceback_complete = True
            self.path_metrics = [0.0] + [float('inf')] * (self.num_states - 1) # Default initial if no stages
            self.path_metrics_history.append(list(self.path_metrics))
            self._notify_listeners({
                "type": "DECODER_RECEIVED_SEQUENCE_LOADED",
                "sequence_loaded": self.current_received_sequence,
                "num_original_bits": self.num_original_message_bits,
                "total_stages": self.T_stages_total,
                "initial_path_metrics": self.path_metrics
            })
            return {"status": "success", "details": "Sequence loaded (empty). Decoding complete."}

        # Initialize path_metrics for start of stage 0 (time t=0)
        self.path_metrics = [0.0] + [float('inf')] * (self.num_states - 1)
        self.path_metrics_history = [list(self.path_metrics)] # PMs before any ACS step (at t=0)

        self.traceback_pointers = [[(0,0) for _ in range(self.num_states)] for _ in range(self.T_stages_total)]
        
        self.is_sequence_loaded = True
        self.current_trellis_stage_idx = 0 # Next stage to process is stage 0
        
        self._notify_listeners({
            "type": "DECODER_RECEIVED_SEQUENCE_LOADED",
            "sequence_loaded": self.current_received_sequence,
            "num_original_bits": self.num_original_message_bits,
            "total_stages": self.T_stages_total,
            "initial_path_metrics": list(self.path_metrics)
        })
        return {"status": "success", "details": f"Sequence loaded. {self.T_stages_total} trellis stages."}

    def go_to_next_decode_step(self): # Performs one ACS step
        if not self.is_sequence_loaded:
            return {"status": "error", "details": "Sequence not loaded."}
        if self.is_acs_complete:
            return {"status": "error", "details": "ACS already complete."}
        
        t = self.current_trellis_stage_idx # This is the stage being processed (e.g., t=0 is the first ACS step)
        
        # received_pair is for stage t (input symbols y_t)
        current_received_pair = (self.current_received_sequence[t*2], self.current_received_sequence[t*2 + 1])
        
        # path_metrics are PMs at the *beginning* of stage t (i.e., end of t-1, or initial for t=0)
        path_metrics_at_stage_t_start = list(self.path_metrics) 
        # new_path_metrics will be PMs at the *end* of stage t (i.e., start of t+1)
        new_path_metrics_for_stage_t_end = [float('inf')] * self.num_states
        
        updated_pointers_for_stage_t = {} # {next_state_idx: (prev_state_idx, input_bit)}

        # For each state S_j at time t+1 (indexed by next_s_idx)
        for next_s_idx in range(self.num_states):
            # Consider all S_i at time t (indexed by prev_s_idx) that could transition to S_j
            for prev_s_idx in range(self.num_states):
                if path_metrics_at_stage_t_start[prev_s_idx] == float('inf'):
                    continue # No valid path to this prev_s_idx at time t

                for input_bit_u in [0, 1]: # Possible input bit u_t
                    # Check if transition (prev_s_idx, input_bit_u) leads to next_s_idx
                    potential_next_state_from_table, expected_out_pair = self._state_transitions_table[prev_s_idx][input_bit_u]
                    
                    if potential_next_state_from_table == next_s_idx:
                        branch_metric = self._hamming_distance(current_received_pair, expected_out_pair)
                        candidate_total_metric = path_metrics_at_stage_t_start[prev_s_idx] + branch_metric
                        
                        if candidate_total_metric < new_path_metrics_for_stage_t_end[next_s_idx]:
                            new_path_metrics_for_stage_t_end[next_s_idx] = candidate_total_metric
                            # Store pointer: at stage t, to reach next_s_idx, we came from prev_s_idx via input_bit_u
                            self.traceback_pointers[t][next_s_idx] = (prev_s_idx, input_bit_u)
                            updated_pointers_for_stage_t[next_s_idx] = (prev_s_idx, input_bit_u)
            
        self.path_metrics = new_path_metrics_for_stage_t_end # Update current PMs to end of stage t
        self.path_metrics_history.append(list(self.path_metrics))

        self._notify_listeners({
            "type": "DECODER_ACS_STEP",
            "stage_processed_idx": t, 
            "received_pair_for_stage": current_received_pair,
            "path_metrics_at_stage_start": path_metrics_at_stage_t_start, # PMs(t)
            "path_metrics_at_stage_end": list(self.path_metrics),         # PMs(t+1)
            "traceback_pointers_set_for_stage": updated_pointers_for_stage_t 
        })
        
        self.current_trellis_stage_idx += 1 # Advance to next stage index
        
        if self.current_trellis_stage_idx == self.T_stages_total:
            self.is_acs_complete = True
            self._notify_listeners({
                "type": "DECODER_ACS_COMPLETE",
                "total_stages_processed": self.T_stages_total,
                "final_path_metrics_at_T": list(self.path_metrics)
            })
            return {"status": "acs_complete", "details": f"ACS complete. Processed {self.T_stages_total} stages."}
        else:
            return {"status": "acs_step_processed", "details": f"ACS step for stage {t} processed. Next stage is {self.current_trellis_stage_idx}."}

    def perform_traceback(self, assume_zero_terminated=True):
        if not self.is_sequence_loaded:
             return {"status": "error", "details": "Sequence not loaded."}
        if not self.is_acs_complete:
             return {"status": "error", "details": "ACS not complete. Cannot perform traceback."}
        # Allow re-running traceback if desired, e.g. with different termination assumption
        # self.is_traceback_complete = False # Reset if re-running
        # self.decoded_message_final = [] 

        if self.T_stages_total == 0:
            self.decoded_message_final = []
            self.is_traceback_complete = True
            self._notify_listeners({
                "type": "DECODER_TRACEBACK_COMPLETE",
                "decoded_message": [],
                "traceback_path_states": [0], # Start and end in state 0 for empty
                "traceback_path_inputs": []
            })
            return {"status": "success", "decoded_message": [], "details": "Traceback on empty sequence."}

        # Determine starting state for traceback at time T (end of trellis)
        final_state_at_trellis_end = 0
        min_final_metric = float('inf')

        if assume_zero_terminated:
            final_state_at_trellis_end = 0 # Expect to end in state 0
            min_final_metric = self.path_metrics[0] # PMs are for end of T_stages_total-1, i.e. time T
            if min_final_metric == float('inf'):
                # Fallback: if state 0 is unreachable, find the best actual state.
                for s_idx, metric_val in enumerate(self.path_metrics):
                    if metric_val < min_final_metric:
                        min_final_metric = metric_val
                        final_state_at_trellis_end = s_idx
        else: # Not assuming zero termination, find the state with the overall minimum path metric at time T
            for s_idx, metric_val in enumerate(self.path_metrics):
                if metric_val < min_final_metric:
                    min_final_metric = metric_val
                    final_state_at_trellis_end = s_idx
        
        if min_final_metric == float('inf') and self.T_stages_total > 0 :
             # This means no path through trellis, something is wrong or all paths impossible.
             # Default to state 0, but result will likely be garbage.
             final_state_at_trellis_end = 0 
             # Fire a warning or handle this? For now, proceed.

        decoded_sequence_with_tail = [0] * self.T_stages_total
        # Store states S_T, S_{T-1}, ..., S_0
        traceback_path_states = [0] * (self.T_stages_total + 1) 
        # Store inputs u_{T-1}, u_{T-2}, ..., u_0
        traceback_path_inputs = [0] * self.T_stages_total      

        current_state_in_traceback = final_state_at_trellis_end
        traceback_path_states[self.T_stages_total] = current_state_in_traceback

        # Trace back from t = T_stages_total-1 down to t = 0
        for t_idx in range(self.T_stages_total - 1, -1, -1):
            # self.traceback_pointers[t_idx][state_at_t_plus_1] gives (state_at_t, input_bit_u_at_t)
            prev_state, input_bit = self.traceback_pointers[t_idx][current_state_in_traceback]
            
            decoded_sequence_with_tail[t_idx] = input_bit # This is u_t
            current_state_in_traceback = prev_state       # This is S_t
            
            traceback_path_states[t_idx] = current_state_in_traceback
            traceback_path_inputs[t_idx] = input_bit
        
        self.decoded_message_final = decoded_sequence_with_tail[:self.num_original_message_bits]
        self.is_traceback_complete = True

        self._notify_listeners({
            "type": "DECODER_TRACEBACK_COMPLETE",
            "decoded_message": list(self.decoded_message_final),
            "full_decoded_sequence_with_tail": decoded_sequence_with_tail,
            "traceback_path_states_chronological": traceback_path_states, # [S_0, S_1, ..., S_T]
            "traceback_path_inputs_chronological": traceback_path_inputs  # [u_0, u_1, ..., u_{T-1}]
        })
        return {"status": "success", "decoded_message": list(self.decoded_message_final), "details": "Traceback complete."}

    # Getter methods for Flask app state
    def get_current_trellis_stage(self): return self.current_trellis_stage_idx
    def get_total_trellis_stages(self): return self.T_stages_total
    def get_current_path_metrics(self): return list(self.path_metrics) # Return a copy


# --- Example Usage (kept from original, ensure ConvolutionalEncoder is available if running this part) ---
# Needs: from src.quantum.quantum_encoder import ConvolutionalEncoder
# if __name__ == '__main__':
    # def run_random_example():
    #     # ... (Original example code can remain here for testing the class)
    #     pass
    # run_random_example()