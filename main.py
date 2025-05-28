import os
import json
import math

from flask import Flask, send_file, request, jsonify

# Assuming these are correctly placed and updated
from src.quantum.quantum_encoder import ConvolutionalEncoder
from src.quantum.quantum_decoder import ViterbiDecoderK7G171_133_Stepwise
from src.quantum.puncturer import PREDEFINED_PUNCTURING_SCHEMES # Import predefined schemes


app = Flask(__name__)

# --- Encoder Globals ---
encoder = None # Will be initialized by set_generator
all_encoder_events = []

# --- Decoder Globals ---
viterbi_decoder_instance = None
all_decoder_events = []

# --- Helper to get or create encoder instance ---
def get_or_create_encoder(constraint_length=None, generators_octal=None):
    global encoder
    if encoder is None:
        if constraint_length is None or generators_octal is None:
            # Default or raise error if essential params missing for first creation
            # For now, let's assume set_generator is called first
            raise ValueError("Encoder must be initialized via /set_generator first if not already existing.")
        encoder = ConvolutionalEncoder(constraint_length=constraint_length, generators_octal=generators_octal)
        encoder.add_listener(flask_encoder_listener)
    # If K or G changes, we should re-initialize. set_generator route handles this.
    return encoder

# --- Listeners (Unchanged) ---
def flask_encoder_listener(event_data):
    global all_encoder_events
    # Sanitize known path metrics fields IF encoder ever emits them (unlikely for current encoder)
    # This is more relevant for the decoder listener.
    all_encoder_events.append(event_data)

def sanitize_path_metrics_for_json(metrics_list):
    if metrics_list is None: return None
    return [None if m is not None and (math.isinf(m) or math.isnan(m)) else m for m in metrics_list]

def flask_decoder_listener(event_data):
    global all_decoder_events
    for key in ['initial_path_metrics', 'path_metrics_at_stage_start', 
                'path_metrics_at_stage_end', 'final_path_metrics_at_T']:
        if key in event_data and event_data[key] is not None:
            event_data[key] = sanitize_path_metrics_for_json(event_data[key])
    all_decoder_events.append(event_data)


# --- State Dictionary Functions ---
def get_current_encoder_state_dict(enc_inst):
    if not enc_inst:
        return {
            "is_initialized": False,
            "constraint_length": 0, "generators_octal": [], "binary_generators": [],
            "is_message_loaded": False, "original_message_length": 0, "padded_message_length": 0,
            "memory": [], "accumulated_output": [], "message_pointer": 0, "is_complete": True,
            "puncturer_info": None, "predefined_puncturing_schemes_available": PREDEFINED_PUNCTURING_SCHEMES
        }
    
    padded_msg = enc_inst.get_current_message()
    punct_info = enc_inst.get_puncturer_info()

    return {
        "is_initialized": True,
        "constraint_length": enc_inst.constraint_length,
        "generators_octal": enc_inst.generators_octal,
        "binary_generators": [''.join(map(str,bg)) for bg in enc_inst.binary_generators],
        "is_message_loaded": enc_inst.is_message_loaded,
        "original_message_length": 0, # This was specific to set_generator, not stored in encoder directly.
                                      # We can infer padded_message_length if needed.
        "padded_message_length": len(padded_msg) if padded_msg else 0,
        "memory": enc_inst.get_current_memory(),
        "accumulated_output": enc_inst.get_current_encoded_data(), # This is now the (potentially) punctured output
        "message_pointer": enc_inst.get_current_message_pointer(),
        "is_complete": enc_inst.is_current_message_fully_encoded(),
        "puncturer_info": punct_info, # Will be None if no puncturer is set
        "predefined_puncturing_schemes_available": PREDEFINED_PUNCTURING_SCHEMES # Send available schemes to UI
    }

def get_current_decoder_state_dict(decoder_inst): # Unchanged from previous version
    if not decoder_inst:
        default_num_states = 64
        return {
            "is_sequence_loaded": False, "current_trellis_stage_idx": 0, "T_stages_total": 0,
            "is_acs_complete": False, "is_traceback_complete": False,
            "path_metrics": sanitize_path_metrics_for_json([0.0] + [float('inf')] * (default_num_states - 1)),
            "decoded_message_final": [], "num_original_message_bits": 0,
            "fixed_params": {"K": 7, "G_octal": ["171", "133"], "num_states": default_num_states}
        }
    return {
        "is_sequence_loaded": decoder_inst.is_sequence_loaded,
        "current_trellis_stage_idx": decoder_inst.get_current_trellis_stage(),
        "T_stages_total": decoder_inst.get_total_trellis_stages(),
        "is_acs_complete": decoder_inst.is_acs_complete,
        "is_traceback_complete": decoder_inst.is_traceback_complete,
        "path_metrics": sanitize_path_metrics_for_json(decoder_inst.get_current_path_metrics()), 
        "decoded_message_final": list(decoder_inst.decoded_message_final),
        "num_original_message_bits": decoder_inst.num_original_message_bits,
        "fixed_params": {
            "K": decoder_inst.constraint_length, "G_octal": decoder_inst.generators_octal,
            "num_states": decoder_inst.num_states
        }
    }

# --- Routes ---
@app.route("/")
def index():
    return send_file('src/index.html')

@app.route('/get_initial_encoder_config', methods=['GET'])
def get_initial_encoder_config():
    # This route can provide predefined puncturing schemes to the UI on load
    # and current encoder state if one exists from a previous session (not implemented here)
    # For now, just returns schemes and a default uninitialized state
    global encoder
    return jsonify(get_current_encoder_state_dict(encoder))


@app.route('/set_generator', methods=['POST'])
def set_generator():
    global encoder, all_encoder_events
    data = request.get_json()
    # ... (validation for K, generators, message as before) ...
    generators_str = data.get('generators', '')
    message_str = data.get('message', '')
    try:
        constraint_length = int(data.get('constraint_length', 3))
        if constraint_length < 1: raise ValueError("K must be >= 1")
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid Constraint Length (K)."}), 400

    try:
        generators = [g.strip() for g in generators_str.split(',') if g.strip()]
        if not generators: raise ValueError("Generators cannot be empty.")
        # Further validation of octal format happens in Encoder init

        parsed_message_bits = []
        if message_str.strip():
            raw_bits = message_str.replace(' ', '')
            if not all(c in '01' for c in raw_bits):
                raise ValueError("Message bits must only contain 0s and 1s.")
            parsed_message_bits = [int(bit) for bit in raw_bits]

        message_to_encode_padded = list(parsed_message_bits)
        if constraint_length > 1: # K=1 has no memory, no padding needed for flushing
            padding_length = constraint_length - 1
            message_to_encode_padded.extend([0] * padding_length)
        
        all_encoder_events = [] 
        # Create or re-create encoder instance
        encoder = ConvolutionalEncoder(constraint_length=constraint_length, generators_octal=generators)
        encoder.add_listener(flask_encoder_listener)
        # Preserve existing puncturer if any, or user has to set it again.
        # For simplicity, setting generator clears puncturer unless explicitly managed.
        # Or, try to re-apply current puncturer if compatible.
        # Current encoder.set_puncturer will raise error if streams don't match.
        # Let's assume setting generator requires re-setting puncturer via UI.
        # Or we can try:
        current_puncturer_info = encoder.get_puncturer_info() # Get before load_message resets it in a way
        
        encoder.load_message(message_to_encode_padded) # This resets puncturer's period_index

        if current_puncturer_info: # Try to re-apply if it was set
            try:
                # We need the key or matrix. Let's assume for now UI sends it again.
                # This part needs robust handling if we want to auto-preserve.
                # For now, if K or G changes, puncturer compatibility might change.
                # User should re-select puncturing.
                pass # encoder.set_puncturer(key_from_current_puncturer_info_if_available)
            except ValueError as pe:
                app.logger.warning(f"Could not re-apply previous puncturer after K/G change: {pe}")


        initial_event = None
        if all_encoder_events:
            for e in all_encoder_events: # Find MESSAGE_LOADED
                if e.get("type") == "MESSAGE_LOADED":
                    initial_event = e
                    break
            if not initial_event: initial_event = all_encoder_events[0]
        
        # Return the full encoder state
        return jsonify({
            "status": "success",
            "message": "Encoder initialized and message loaded.",
            "initial_event": initial_event,
            "encoder_state": get_current_encoder_state_dict(encoder) 
        })
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in /set_generator: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/set_puncturing', methods=['POST'])
def set_puncturing_route():
    global encoder, all_encoder_events
    if encoder is None:
        return jsonify({"status": "error", "message": "Encoder not initialized. Set K and G first."}), 400

    data = request.get_json()
    scheme_key = data.get('puncturing_scheme_key') # e.g., "2/3", "NONE", etc.
    # custom_matrix = data.get('custom_matrix') # For future extension

    try:
        num_events_before = len(all_encoder_events)
        encoder.set_puncturer(puncturing_scheme_key=scheme_key) # custom_matrix=custom_matrix)
        
        # Check if PUNCTURER_CONFIG_CHANGED event was fired
        config_event = None
        if len(all_encoder_events) > num_events_before:
            for e in reversed(all_encoder_events[num_events_before:]):
                if e.get("type") == "PUNCTURER_CONFIG_CHANGED":
                    config_event = e
                    break
        
        # If a message is already loaded, changing puncturing might invalidate previous steps.
        # The encoder's internal buffers are NOT automatically re-processed by set_puncturer.
        # A full message reload or encoder reset might be advisable for user clarity.
        # For now, we assume user understands this or will reload message.
        # The `load_message` in encoder now resets the puncturer's period index.

        return jsonify({
            "status": "success",
            "message": f"Puncturing scheme set to: {encoder.get_puncturer_info()['label'] if encoder.get_puncturer_info() else 'None'}.",
            "event": config_event,
            "encoder_state": get_current_encoder_state_dict(encoder)
        })
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e), "encoder_state": get_current_encoder_state_dict(encoder)}), 400
    except Exception as e:
        app.logger.error(f"Error setting puncturer: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}", "encoder_state": get_current_encoder_state_dict(encoder)}), 500


@app.route('/next_step', methods=['POST'])
def next_step():
    global encoder, all_encoder_events
    if encoder is None:
        return jsonify({"status": "error", "message": "Encoder not initialized."}), 400

    num_events_before_step = len(all_encoder_events)
    step_outcome = encoder.go_to_next_step() # This now handles puncturing
    
    newly_generated_events = all_encoder_events[num_events_before_step:]
    primary_event_for_step = None
    # Find ENCODE_STEP or ENCODING_COMPLETE
    if newly_generated_events:
        for e in reversed(newly_generated_events):
            if e.get("type") in ["ENCODE_STEP", "ENCODING_COMPLETE"]:
                primary_event_for_step = e
                break
        if not primary_event_for_step: primary_event_for_step = newly_generated_events[-1]
            
    return jsonify({
        "status_from_method": step_outcome["status"], 
        "details_from_method": step_outcome.get("details"),
        "output_bits_from_method": step_outcome.get("output_bits"), # This is the punctured output for the step
        "primary_event": primary_event_for_step, # This event will contain both unpunctured and punctured
        "encoder_state": get_current_encoder_state_dict(encoder)
    })

# --- Decoder Routes (Largely Unchanged from previous version) ---
# ... (keep existing decoder routes: /decoder/load_sequence, /decoder/next_acs_step, etc.)
@app.route('/decoder/load_sequence', methods=['POST'])
def decoder_load_sequence():
    global viterbi_decoder_instance, all_decoder_events
    data = request.get_json()
    received_sequence_str = data.get('received_sequence', '')
    num_original_bits_str = data.get('num_original_message_bits', '0')

    try:
        num_original_bits = int(num_original_bits_str)
        if num_original_bits < 0:
            raise ValueError("Number of original message bits cannot be negative.")

        parsed_sequence = []
        if received_sequence_str.strip():
            raw_bits = received_sequence_str.replace(' ', '').replace(',', '')
            if not all(c in '01' for c in raw_bits):
                raise ValueError("Received sequence must only contain 0s and 1s.")
            parsed_sequence = [int(bit) for bit in raw_bits]
        
        all_decoder_events = [] 
        decoder = get_or_create_decoder() 
        
        load_outcome = decoder.load_received_sequence(parsed_sequence, num_original_bits)
        
        final_event_for_response = None
        for e in reversed(all_decoder_events): 
            if e.get("type") == "DECODER_RECEIVED_SEQUENCE_LOADED":
                final_event_for_response = e
                break
        if not final_event_for_response and all_decoder_events: 
            final_event_for_response = all_decoder_events[-1] 

        return jsonify({
            "status": load_outcome.get("status", "error"),
            "message": load_outcome.get("details", "Sequence load processed."),
            "event": final_event_for_response,
            "decoder_state": get_current_decoder_state_dict(decoder)
        })
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e), "decoder_state": get_current_decoder_state_dict(viterbi_decoder_instance)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in /decoder/load_sequence: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}", "decoder_state": get_current_decoder_state_dict(viterbi_decoder_instance)}), 500

@app.route('/decoder/next_acs_step', methods=['POST'])
def decoder_next_acs_step():
    global viterbi_decoder_instance, all_decoder_events # Corrected global name
    decoder = get_or_create_decoder()
    if not decoder.is_sequence_loaded:
        return jsonify({"status": "error", "message": "No sequence loaded.", "decoder_state": get_current_decoder_state_dict(decoder)}), 400
    if decoder.is_acs_complete:
        return jsonify({"status": "info", "message": "ACS already complete.", "event": None, "decoder_state": get_current_decoder_state_dict(decoder)}), 200

    num_events_before_step = len(all_decoder_events)
    step_outcome = decoder.go_to_next_decode_step() 

    newly_generated_events = all_decoder_events[num_events_before_step:]
    primary_event = None
    if newly_generated_events:
        for e in reversed(newly_generated_events): 
            if e.get("type") in ["DECODER_ACS_STEP", "DECODER_ACS_COMPLETE"]:
                primary_event = e
                break
        if not primary_event and newly_generated_events: primary_event = newly_generated_events[-1]
        
    return jsonify({
        "status": step_outcome.get("status", "success"), 
        "message": step_outcome.get("details", "ACS step performed."),
        "method_outcome": step_outcome, 
        "event": primary_event,
        "decoder_state": get_current_decoder_state_dict(decoder)
    })

@app.route('/decoder/perform_traceback', methods=['POST'])
def decoder_perform_traceback():
    global viterbi_decoder_instance, all_decoder_events # Corrected global name
    decoder = get_or_create_decoder()
    if not decoder.is_sequence_loaded:
        return jsonify({"status": "error", "message": "No sequence loaded.", "decoder_state": get_current_decoder_state_dict(decoder)}), 400
    if not decoder.is_acs_complete:
        return jsonify({"status": "error", "message": "ACS not yet complete.", "decoder_state": get_current_decoder_state_dict(decoder)}), 400

    data = request.get_json()
    assume_zero_terminated = data.get('assume_zero_terminated', True)

    num_events_before_step = len(all_decoder_events)
    traceback_outcome = decoder.perform_traceback(assume_zero_terminated=assume_zero_terminated)

    newly_generated_events = all_decoder_events[num_events_before_step:]
    primary_event = None
    if newly_generated_events:
        for e in reversed(newly_generated_events): 
            if e.get("type") == "DECODER_TRACEBACK_COMPLETE":
                primary_event = e
                break
        if not primary_event and newly_generated_events: primary_event = newly_generated_events[-1]

    return jsonify({
        "status": traceback_outcome.get("status", "success"), 
        "message": traceback_outcome.get("details", "Traceback performed."),
        "method_outcome": traceback_outcome,
        "event": primary_event,
        "decoder_state": get_current_decoder_state_dict(decoder)
    })

@app.route('/decoder/reset', methods=['POST'])
def decoder_reset():
    global viterbi_decoder_instance, all_decoder_events
    all_decoder_events = [] 
    viterbi_decoder_instance = None 
    decoder = get_or_create_decoder()

    reset_event = None
    if all_decoder_events:
        reset_event = all_decoder_events[0]

    return jsonify({
        "status": "success", "message": "Decoder reset.",
        "event": reset_event, "decoder_state": get_current_decoder_state_dict(decoder)
    })


def get_or_create_decoder(): # Moved here to be defined before use
    global viterbi_decoder_instance
    if viterbi_decoder_instance is None:
        viterbi_decoder_instance = ViterbiDecoderK7G171_133_Stepwise()
        viterbi_decoder_instance.add_listener(flask_decoder_listener)
    return viterbi_decoder_instance
    
# --- Main ---
def main():
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True) 

if __name__ == "__main__":
    main()