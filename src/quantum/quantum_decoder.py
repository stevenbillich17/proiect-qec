import math # For float('inf')
import random # For testing
from quantum_encoder import ConvolutionalEncoder

# --- Viterbi Decoder for K=7, G=["171", "133"] with Step-by-Step and Listeners ---
class ViterbiDecoderK7G171_133_Stepwise:
    """
    Hard-Decision Viterbi Decoder specifically for:
    - Constraint Length (K) = 7
    - Rate = 1/2
    - Generator Polynomials (octal) = ["171", "133"]
    Supports step-by-step decoding and listeners for UI integration.
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

        self.listeners = []
        
        # Precompute state transitions (fixed for the code)
        # self.state_transitions[current_state_int][input_bit_u] = (next_state_int, (out_c0, out_c1))
        self._state_transitions_table = [{} for _ in range(self.num_states)] 
        self._precompute_state_transitions()

        self._reset_decoding_state() # Initialize dynamic decoding variables

    def _reset_decoding_state(self):
        """Resets dynamic state variables for a new decoding operation."""
        self.current_received_sequence = []
        self.num_original_message_bits = 0
        self.current_trellis_stage_idx = 0  # Index of the *next* ACS stage to process
        self.T_stages_total = 0             # Total number of trellis stages for the loaded sequence

        self.path_metrics = []              # List of path metrics to each state at current stage
        self.traceback_pointers = []        # 2D list: [stage_idx][next_state_idx] -> (prev_state, input_bit)
        
        self.is_sequence_loaded = False
        self.is_acs_complete = False        # Add-Compare-Select phase
        self.is_traceback_complete = False
        self.decoded_message_final = []

        self._notify_listeners({
            "type": "DECODER_RESET",
            "status": "Decoder state reset for new sequence."
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

    def _state_to_memory_bits(self, state_int):
        return [int(bit) for bit in bin(state_int)[2:].zfill(self.memory_size)]

    def _memory_bits_to_state(self, memory_bits_list):
        return int("".join(map(str, memory_bits_list)), 2)

    def _precompute_state_transitions(self):
        """
        This function computes the transitions (state vector machine)
        Slide 36 -> The one on top left
        """
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
                expected_output_pair = (output_c0, output_c1) # the YY from X/YY YY->will represent the received value, X is input_u
                next_memory_content = ([input_bit_u] + current_memory_content)[:self.memory_size] #[1+011]=>[101] // the new mem state
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
            raise ValueError("Received sequence length must be a multiple of 2.")
        if not isinstance(num_original_message_bits, int) or num_original_message_bits < 0:
            raise ValueError("Number of original message bits must be a non-negative integer.")

        self._reset_decoding_state() # Clear previous state

        self.current_received_sequence = list(received_sequence_bits)
        self.num_original_message_bits = num_original_message_bits
        self.T_stages_total = len(self.current_received_sequence) // 2
        
        if self.T_stages_total == 0:
            self.is_sequence_loaded = True # Can be considered loaded but ACS will be skipped
            self.is_acs_complete = True
            self.is_traceback_complete = True # No traceback needed
            self._notify_listeners({
                "type": "DECODER_RECEIVED_SEQUENCE_LOADED",
                "received_sequence_length": 0,
                "num_original_message_bits": self.num_original_message_bits,
                "total_trellis_stages": 0,
                "status": "Empty sequence loaded; decoding considered complete."
            })
            return

        # INITIALIZE VITERBI ALGORITHM'S FORWARD PASS (ACS)
        # Always starts from state 0 and has cost 0 and for the other 63 paths we set the cost to infinity
        self.path_metrics = [0.0] + [float('inf')] * (self.num_states - 1) # Start at state 0
        # Initialize a 2D structure trac_pointers[t][next_state] each entry (prev_state_idx, input_bit) t=(stageNumber) [nextStateIdx]=bestPathToNextState at stage t
        # All the entries are initialized with (0,0)
        self.traceback_pointers = [[(0,0) for _ in range(self.num_states)] for _ in range(self.T_stages_total)]
        self.is_sequence_loaded = True

        self._notify_listeners({
            "type": "DECODER_RECEIVED_SEQUENCE_LOADED",
            "received_sequence_length": len(self.current_received_sequence),
            "num_original_message_bits": self.num_original_message_bits,
            "total_trellis_stages": self.T_stages_total,
            "initial_path_metrics": list(self.path_metrics),
            "status": "Sequence loaded; ready for ACS."
        })
        
    def go_to_next_decode_step(self):
        """Performs one Add-Compare-Select (ACS) step for the current trellis stage."""
        if not self.is_sequence_loaded:
            return {"status": "no_sequence_loaded", "details": "No sequence loaded. Call load_received_sequence() first."}
        if self.is_acs_complete:
            return {"status": "acs_already_complete", "details": "ACS phase is already complete. Perform traceback."}
        if self.current_trellis_stage_idx >= self.T_stages_total : # Should be caught by is_acs_complete
             return {"status": "error_acs_overflow", "details": "Internal error: trellis stage index out of bounds."}


        t = self.current_trellis_stage_idx
        current_received_pair = (self.current_received_sequence[t*2], self.current_received_sequence[t*2 + 1])
        
        path_metrics_at_stage_start = list(self.path_metrics) # Metrics for states at time t
        new_path_metrics_for_next_stage = [float('inf')] * self.num_states # Will be metrics for states at time t+1

        # For listener: detailed breakdown of calculations for this stage
        stage_state_updates_info = []

        for next_s_idx in range(self.num_states):
            candidate_info_for_next_s = {
                "next_state_index": next_s_idx,
                "candidates_considered": [],
                "surviving_metric": float('inf'),
                "surviving_path_from_state": None,
                "surviving_path_input_bit": None
            }
            # For each next_state, find which previous_state + input_bit leads to it
            for prev_s_idx in range(self.num_states):
                if path_metrics_at_stage_start[prev_s_idx] == float('inf'):
                    continue # Path to prev_s_idx was not viable

                for input_bit_u in [0, 1]:
                    # Check if (prev_s_idx, input_bit_u) transitions to next_s_idx
                    potential_next_state, expected_out = self._state_transitions_table[prev_s_idx][input_bit_u]
                    
                    if potential_next_state == next_s_idx:
                        branch_metric = self._hamming_distance(current_received_pair, expected_out)
                        candidate_total_metric = path_metrics_at_stage_start[prev_s_idx] + branch_metric
                        
                        candidate_info_for_next_s["candidates_considered"].append({
                            "previous_state_index": prev_s_idx,
                            "input_bit": input_bit_u,
                            "expected_output": expected_out,
                            "branch_metric": branch_metric,
                            "path_metric_from_prev": path_metrics_at_stage_start[prev_s_idx],
                            "candidate_total_metric": candidate_total_metric
                        })

                        if candidate_total_metric < new_path_metrics_for_next_stage[next_s_idx]:
                            new_path_metrics_for_next_stage[next_s_idx] = candidate_total_metric
                            self.traceback_pointers[t][next_s_idx] = (prev_s_idx, input_bit_u)
                            
                            # Update surviving path info for listener for this next_s_idx
                            candidate_info_for_next_s["surviving_metric"] = candidate_total_metric
                            candidate_info_for_next_s["surviving_path_from_state"] = prev_s_idx
                            candidate_info_for_next_s["surviving_path_input_bit"] = input_bit_u
            
            if candidate_info_for_next_s["candidates_considered"]: # Only add if there were actual candidates
                 stage_state_updates_info.append(candidate_info_for_next_s)


        self.path_metrics = new_path_metrics_for_next_stage
        
        self._notify_listeners({
            "type": "DECODER_ACS_STEP",
            "stage_index": t,
            "received_pair_for_stage": current_received_pair,
            "path_metrics_at_stage_start": path_metrics_at_stage_start, # Metrics for states at time t
            "state_updates_details": stage_state_updates_info, # Detailed calculations for each next_state
            "path_metrics_at_stage_end": list(self.path_metrics), # Metrics for states at time t+1
            "traceback_pointers_set_for_stage": list(self.traceback_pointers[t]) if t < self.T_stages_total else []
        })

        self.current_trellis_stage_idx += 1
        if self.current_trellis_stage_idx == self.T_stages_total:
            self.is_acs_complete = True
            self._notify_listeners({
                "type": "DECODER_ACS_COMPLETE",
                "total_stages_processed": self.T_stages_total,
                "final_path_metrics": list(self.path_metrics) # Path metrics after all stages
            })
            return {"status": "acs_complete", "details": f"ACS complete after processing stage {t}."}
        else:
            return {"status": "acs_step_processed", "details": f"Processed ACS for stage {t}."}

    def perform_traceback(self, assume_zero_terminated=True):
        if not self.is_sequence_loaded:
            return {"status": "no_sequence_loaded", "decoded_message": [], "details": "Load sequence first."}
        if not self.is_acs_complete:
            return {"status": "acs_not_complete", "decoded_message": [], "details": "ACS phase not complete."}
        if self.is_traceback_complete:
            return {"status": "traceback_already_done", "decoded_message": list(self.decoded_message_final), 
                    "details": "Traceback already performed."}
        
        if self.T_stages_total == 0: # Handle empty sequence
            self.decoded_message_final = []
            self.is_traceback_complete = True
            self._notify_listeners({
                "type": "DECODER_TRACEBACK_COMPLETE",
                "final_state_chosen": 0, # Or None
                "traceback_path_details": [],
                "decoded_sequence_with_tail": [],
                "decoded_message": []
            })
            return {"status": "success", "decoded_message": []}

        decoded_sequence_with_tail = [0] * self.T_stages_total
        
        final_state_at_trellis_end = 0
        min_final_metric = float('inf')

        if assume_zero_terminated:
            final_state_at_trellis_end = 0
            min_final_metric = self.path_metrics[0]
            if min_final_metric == float('inf'):
                 self._notify_listeners({"type":"DECODER_TRACEBACK_WARNING", "message": "Assumed zero term, but state 0 unreachable. Fallback."})
                 # Fallback to min metric state if state 0 is unreachable
                 for s_idx, metric_val in enumerate(self.path_metrics):
                    if metric_val < min_final_metric:
                        min_final_metric = metric_val
                        final_state_at_trellis_end = s_idx
        else:
            for s_idx, metric_val in enumerate(self.path_metrics):
                if metric_val < min_final_metric:
                    min_final_metric = metric_val
                    final_state_at_trellis_end = s_idx
        
        if min_final_metric == float('inf'):
            self._notify_listeners({"type":"DECODER_TRACEBACK_ERROR", "message": "All final states have infinite path metrics. Decoding likely failed."})
            # Still proceed with traceback from state 0, but result will be unreliable.
            final_state_at_trellis_end = 0 


        current_state_in_traceback = final_state_at_trellis_end
        traceback_path_details_for_listener = []

        for t_idx in range(self.T_stages_total - 1, -1, -1):
            prev_state, input_bit = self.traceback_pointers[t_idx][current_state_in_traceback]
            decoded_sequence_with_tail[t_idx] = input_bit
            traceback_path_details_for_listener.append({
                "stage_index": t_idx,
                "state_at_stage_t_plus_1": current_state_in_traceback, # State being traced from
                "decoded_input_bit_at_t": input_bit,
                "previous_state_at_stage_t": prev_state       # State traced to
            })
            current_state_in_traceback = prev_state
        
        traceback_path_details_for_listener.reverse() # Chronological order for listener
        self.decoded_message_final = decoded_sequence_with_tail[:self.num_original_message_bits]
        self.is_traceback_complete = True

        self._notify_listeners({
            "type": "DECODER_TRACEBACK_COMPLETE",
            "final_state_chosen": final_state_at_trellis_end,
            "min_final_metric": min_final_metric,
            "traceback_path_details": traceback_path_details_for_listener,
            "decoded_sequence_with_tail": list(decoded_sequence_with_tail),
            "decoded_message": list(self.decoded_message_final)
        })
        return {"status": "success", "decoded_message": list(self.decoded_message_final)}

    def decode(self, received_sequence_bits, num_original_message_bits, assume_zero_terminated=True):
        """Performs full decoding (load, ACS, traceback) in batch."""
        self.load_received_sequence(received_sequence_bits, num_original_message_bits)
        
        if self.T_stages_total == 0: # Empty sequence handled by load
            return []

        while not self.is_acs_complete:
            result = self.go_to_next_decode_step()
            if result["status"] not in ["acs_step_processed", "acs_complete"]:
                # This indicates an issue, like sequence not loaded or error
                raise RuntimeError(f"Unexpected status during batch ACS: {result['status']} - {result.get('details','')}")

        traceback_result = self.perform_traceback(assume_zero_terminated)
        if traceback_result["status"] == "success":
            return traceback_result["decoded_message"]
        else:
            # Handle cases where traceback might not be 'success' due to earlier issues
            raise RuntimeError(f"Traceback failed during batch decode: {traceback_result['status']} - {traceback_result.get('details','')}")

    # --- Getter methods for UI ---
    def get_current_path_metrics(self): return list(self.path_metrics) if self.is_sequence_loaded else []
    def get_traceback_pointers_all_stages(self): return list(self.traceback_pointers) if self.is_sequence_loaded else []
    def get_current_trellis_stage(self): return self.current_trellis_stage_idx
    def get_total_trellis_stages(self): return self.T_stages_total
    def is_current_sequence_fully_decoded(self): return self.is_traceback_complete


# --- Example Listener and Main Execution ---
if __name__ == '__main__':
    encoder = ConvolutionalEncoder(constraint_length=7, generators_octal=["171", "133"])
    decoder = ViterbiDecoderK7G171_133_Stepwise()

    def comprehensive_decoder_listener(event_data):
        event_type = event_data.get("type", "UNKNOWN_EVENT")
        print(f"\n🔔 DECODER LISTENER ({event_type}) 🔔")

        if event_type == "DECODER_RESET":
            print(f"  Status: {event_data['status']}")
        elif event_type == "DECODER_RECEIVED_SEQUENCE_LOADED":
            print(f"  Sequence Length : {event_data['received_sequence_length']}")
            print(f"  Original Msg Bits: {event_data['num_original_message_bits']}")
            print(f"  Total Stages    : {event_data['total_trellis_stages']}")
            if "initial_path_metrics" in event_data:
                 print(f"  Initial Path Metrics (first 5): {event_data['initial_path_metrics'][:5]}...")
            print(f"  Status          : {event_data['status']}")
        elif event_type == "DECODER_ACS_STEP":
            print(f"  ACS for Stage Index : {event_data['stage_index']}")
            print(f"  Received Pair       : {event_data['received_pair_for_stage']}")
            print(f"  Path Metrics (Start): {event_data['path_metrics_at_stage_start'][:4]}... (to) ...{event_data['path_metrics_at_stage_start'][-4:]}")
            # For brevity, we won't print all state_updates_details here
            # In a UI, you'd iterate through event_data['state_updates_details']
            # and for each next_state, show its event_data['state_updates_details'][j]['candidates_considered']
            # and which one survived.
            num_updated_states = len([s_info for s_info in event_data['state_updates_details'] if s_info['surviving_metric'] != float('inf')])
            print(f"  State Updates       : {num_updated_states} states updated (details omitted for brevity)")
            print(f"  Path Metrics (End)  : {event_data['path_metrics_at_stage_end'][:4]}... (to) ...{event_data['path_metrics_at_stage_end'][-4:]}")
            # print(f"  Traceback Pointers  : {event_data['traceback_pointers_set_for_stage']}") # Can be very long
        elif event_type == "DECODER_ACS_COMPLETE":
            print(f"  ACS Phase Complete!")
            print(f"  Total Stages Processed: {event_data['total_stages_processed']}")
            print(f"  Final Path Metrics (first 5): {event_data['final_path_metrics'][:5]}...")
        elif event_type == "DECODER_TRACEBACK_WARNING" or event_type == "DECODER_TRACEBACK_ERROR":
            print(f"  Message: {event_data['message']}")
        elif event_type == "DECODER_TRACEBACK_COMPLETE":
            print(f"  Traceback Complete!")
            print(f"  Final State Chosen for Traceback: {event_data['final_state_chosen']} (Metric: {event_data.get('min_final_metric', 'N/A')})")
            # Print first few steps of traceback path for brevity
            print(f"  Traceback Path (first 3 steps):")
            for i, step in enumerate(event_data['traceback_path_details'][:3]):
                print(f"    t={step['stage_index']}: PrevS={step['previous_state_at_stage_t']}, Bit={step['decoded_input_bit_at_t']}, NextS={step['state_at_stage_t_plus_1']}")
            if len(event_data['traceback_path_details']) > 3: print("    ...")
            print(f"  Decoded (with tail) : {''.join(map(str,event_data['decoded_sequence_with_tail'][:32]))}...")
            print(f"  Final Decoded Msg   : {''.join(map(str,event_data['decoded_message'][:32]))}...")
        else:
            print(f"  Raw Event Data: {event_data}")
        print(f"🔔 --- End Decoder Notification --- 🔔")

    decoder.add_listener(comprehensive_decoder_listener)

    # --- Test with a 128-bit message ---
    num_message_bits = 128 # As per your task
    # num_message_bits = 10 # Smaller for quicker manual check of listener output
    random.seed(42)
    original_message = [random.randint(0, 1) for _ in range(num_message_bits)]
    
    num_tail_bits = encoder.constraint_length - 1 # K-1 = 6
    message_with_tail = original_message + [0] * num_tail_bits
    
    print(f"\nOriginal Message ({num_message_bits} bits): {''.join(map(str,original_message[:32]))}...")
    
    encoded_sequence = encoder.encode(message_with_tail)
    print(f"Encoded Sequence Length: {len(encoded_sequence)}")

    # --- Step-by-step decoding ---
    print("\n--- Initiating Step-by-Step Decoding ---")
    decoder.load_received_sequence(encoded_sequence, num_message_bits)

    step_count = 0
    if decoder.T_stages_total > 0: # Only proceed if there are stages
        while not decoder.is_acs_complete:
            step_count += 1
            print(f"\n>>> Calling go_to_next_decode_step() - ACS Stage {decoder.get_current_trellis_stage()} <<<")
            step_info = decoder.go_to_next_decode_step()
            print(f">>> go_to_next_decode_step() returned: {step_info}")
            if step_count > decoder.T_stages_total + 5: # Safety break for debugging
                print("ERROR: Exceeded expected number of ACS steps.")
                break
    else: # Handle empty sequence case properly
        print("Sequence is empty, ACS phase skipped.")


    print("\n>>> Performing Traceback <<<")
    traceback_info = decoder.perform_traceback(assume_zero_terminated=True)
    print(f">>> perform_traceback() returned: {traceback_info['status']}")
    
    final_decoded_stepwise = traceback_info["decoded_message"]

    if final_decoded_stepwise == original_message:
        print("\nSUCCESS (Step-by-step): Decoded message matches original.")
    else:
        print("\nFAILURE (Step-by-step): Decoded message does NOT match original.")
        err_count = sum(1 for i in range(num_message_bits) if final_decoded_stepwise[i] != original_message[i])
        print(f"  Errors: {err_count}/{num_message_bits}")

    # --- Batch decoding test (resets and uses internal loop) ---
    print("\n--- Initiating Batch Decoding Test ---")
    # Decoder will be reset by the .decode() call implicitly via load_received_sequence
    decoder_batch = ViterbiDecoderK7G171_133_Stepwise() # Fresh instance or call reset
    decoder_batch.add_listener(comprehensive_decoder_listener) # Add listener if you want to see its events

    final_decoded_batch = decoder_batch.decode(encoded_sequence, num_message_bits, assume_zero_terminated=True)

    if final_decoded_batch == original_message:
        print("\nSUCCESS (Batch): Decoded message matches original.")
    else:
        print("\nFAILURE (Batch): Decoded message does NOT match original.")
        err_count = sum(1 for i in range(num_message_bits) if final_decoded_batch[i] != original_message[i])
        print(f"  Errors: {err_count}/{num_message_bits}")