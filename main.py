import os
import json

import math

from flask import Flask, send_file, request, jsonify

from src.quantum.quantum_encoder import ConvolutionalEncoder
# Ensure this path is correct for your project structure
from src.quantum.quantum_decoder import ViterbiDecoderK7G171_133_Stepwise 


def sanitize_floats(xs):
    """Replace any NaN or ±Inf with None, so jsonify produces JSON null."""
    return [None if (not isinstance(x, (int, float)) or math.isinf(x) or math.isnan(x)) else x
            for x in xs]

app = Flask(__name__)

# --- Encoder Globals ---
encoder = None
all_encoder_events = []

def flask_encoder_listener(event_data):
    global all_encoder_events
    all_encoder_events.append(event_data)

# --- Decoder Globals ---
viterbi_decoder_instance = None
all_decoder_events = [] # Separate list for decoder events

def flask_decoder_listener(event_data):
    global all_decoder_events
    all_decoder_events.append(event_data)
    # print(f"Flask Decoder Listener Captured Event: {event_data.get('type')}")


def get_or_create_decoder():
    global viterbi_decoder_instance, all_decoder_events
    if viterbi_decoder_instance is None:
        # all_decoder_events = [] # Clear events WHEN a new instance is made.
                                # The ViterbiDecoder's __init__ fires DECODER_RESET,
                                # so this ensures only that reset (and subsequent events) are captured.
        # However, if get_or_create_decoder is called multiple times *without* viterbi_decoder_instance being None,
        # all_decoder_events would persist.
        # The current routes clear all_decoder_events before specific actions like load or reset,
        # which is a more robust approach.
        viterbi_decoder_instance = ViterbiDecoderK7G171_133_Stepwise()
        viterbi_decoder_instance.add_listener(flask_decoder_listener)
        # First event for a new instance will be DECODER_RESET from its __init__
    return viterbi_decoder_instance

def get_current_decoder_state_dict(decoder_inst):
    if not decoder_inst: # Should not happen if get_or_create_decoder is always used
        return {
            "is_sequence_loaded": False, "current_trellis_stage_idx": 0, "T_stages_total": 0,
            "is_acs_complete": False, "is_traceback_complete": False,
            "path_metrics": sanitize_floats([0.0] + [float('inf')] * ( (2**(7-1)) -1)), # Default for 64 states
            "decoded_message_final": [],
            "num_original_message_bits": 0,
            "fixed_params": {"K": 7, "G": ["171", "133"], "num_states": 64}
        }
    return {
        "is_sequence_loaded": decoder_inst.is_sequence_loaded,
        "current_trellis_stage_idx": decoder_inst.get_current_trellis_stage(),
        "T_stages_total": decoder_inst.get_total_trellis_stages(),
        "is_acs_complete": decoder_inst.is_acs_complete,
        "is_traceback_complete": decoder_inst.is_traceback_complete,
        "path_metrics": sanitize_floats(decoder_inst.get_current_path_metrics()), 
        "decoded_message_final": list(decoder_inst.decoded_message_final),
        "num_original_message_bits": decoder_inst.num_original_message_bits,
        "fixed_params": { # Add fixed params for UI display
            "K": decoder_inst.constraint_length,
            "G_octal": decoder_inst.generators_octal,
            "num_states": decoder_inst.num_states
        }
    }

@app.route("/")
def index():
    return send_file('src/index.html') # Make sure this path is correct

# --- Encoder Routes (existing - unchanged) ---
@app.route('/set_generator', methods=['POST'])
def set_generator():
    global encoder, all_encoder_events
    data = request.get_json()

    generators_str = data.get('generators', '')
    message_str = data.get('message', '')
    try:
        constraint_length = int(data.get('constraint_length', 3))
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid Constraint Length (K). Must be an integer."}), 400

    try:
        generators = [g.strip() for g in generators_str.split(',') if g.strip()]
        if not generators:
            raise ValueError("Generators cannot be empty.")

        parsed_message_bits = []
        if message_str.strip():
            raw_bits = message_str.replace(' ', '')
            if not all(c in '01' for c in raw_bits):
                raise ValueError("Message bits must only contain 0s and 1s.")
            parsed_message_bits = [int(bit) for bit in raw_bits]

        message_to_encode_padded = list(parsed_message_bits)
        if constraint_length > 1:
            padding_length = constraint_length - 1
            message_to_encode_padded.extend([0] * padding_length)
        
        all_encoder_events = [] 
        encoder = ConvolutionalEncoder(constraint_length=constraint_length, generators_octal=generators)
        encoder.add_listener(flask_encoder_listener)
        encoder.load_message(message_to_encode_padded)
        initial_event = all_encoder_events[0] if all_encoder_events else None

        return jsonify({
            "status": "success",
            "message": "Encoder initialized and message loaded.",
            "initial_event": initial_event,
            "config": {
                "constraint_length": encoder.constraint_length,
                "generators_octal": encoder.generators_octal,
                "binary_generators": [''.join(map(str,bg)) for bg in encoder.binary_generators],
                "original_message_length": len(parsed_message_bits),
                "padded_message_length": len(message_to_encode_padded)
            }
        })
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/next_step', methods=['POST'])
def next_step():
    global encoder, all_encoder_events
    if encoder is None:
        return jsonify({"status": "error", "message": "Encoder not initialized. Set generator first."}), 400

    num_events_before_step = len(all_encoder_events)
    step_outcome = encoder.go_to_next_step()
    newly_generated_events = all_encoder_events[num_events_before_step:]
    primary_event_for_step = None
    if newly_generated_events:
        for e in reversed(newly_generated_events):
            if e.get("type") in ["ENCODE_STEP", "ENCODING_COMPLETE"]:
                primary_event_for_step = e
                break
        if not primary_event_for_step: # Fallback to last event if specific types not found
            primary_event_for_step = newly_generated_events[-1]
            
    return jsonify({
        "status_from_method": step_outcome["status"], 
        "details_from_method": step_outcome.get("details"),
        "output_bits_from_method": step_outcome.get("output_bits"),
        "primary_event": primary_event_for_step,
        "encoder_state_after_step": {
            "memory": encoder.get_current_memory(),
            "accumulated_output": encoder.get_current_encoded_data(),
            "message_pointer": encoder.get_current_message_pointer(),
            "is_complete": encoder.is_current_message_fully_encoded(),
            "message_length": len(encoder.get_current_message()) if encoder.get_current_message() else 0
        }
    })

# --- Viterbi Decoder Routes (Mostly existing, confirmed for Stepwise class) ---
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
            raw_bits = received_sequence_str.replace(' ', '').replace(',', '') # Allow commas too
            if not all(c in '01' for c in raw_bits):
                raise ValueError("Received sequence must only contain 0s and 1s.")
            parsed_sequence = [int(bit) for bit in raw_bits]
        
        decoder = get_or_create_decoder() 
        
        all_decoder_events = [] # Fresh slate for events for this load operation
                                # DECODER_RESET (from load_received_sequence) and DECODER_RECEIVED_SEQUENCE_LOADED will be captured

        load_outcome = decoder.load_received_sequence(parsed_sequence, num_original_bits)
        
        # The most relevant event is DECODER_RECEIVED_SEQUENCE_LOADED
        # DECODER_RESET would have been fired by _reset_decoding_state inside load_received_sequence
        final_event_for_response = None
        for e in reversed(all_decoder_events): # Get the latest, most significant event
            if e.get("type") == "DECODER_RECEIVED_SEQUENCE_LOADED":
                final_event_for_response = e
                break
        if not final_event_for_response and all_decoder_events: 
            final_event_for_response = all_decoder_events[-1] # Fallback (e.g. only DECODER_RESET)

        return jsonify({
            "status": "success", # Or load_outcome['status'] if it can fail gracefully and return error status
            "message": load_outcome.get("details", "Sequence load processed."),
            "event": final_event_for_response,
            "decoder_state": get_current_decoder_state_dict(decoder)
        })
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e), "decoder_state": get_current_decoder_state_dict(viterbi_decoder_instance)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}", "decoder_state": get_current_decoder_state_dict(viterbi_decoder_instance)}), 500

@app.route('/decoder/next_acs_step', methods=['POST'])
def decoder_next_acs_step():
    global viterbi_decoder_instance, all_decoder_events
    decoder = get_or_create_decoder() # Ensures instance exists
    if not decoder.is_sequence_loaded:
        return jsonify({"status": "error", "message": "No sequence loaded.", "decoder_state": get_current_decoder_state_dict(decoder)}), 400
    if decoder.is_acs_complete:
        return jsonify({"status": "info", "message": "ACS already complete.", "decoder_state": get_current_decoder_state_dict(decoder)}), 200 # Use 200 for info

    num_events_before_step = len(all_decoder_events)
    step_outcome = decoder.go_to_next_decode_step() 

    newly_generated_events = all_decoder_events[num_events_before_step:]
    primary_event = None
    if newly_generated_events:
        for e in reversed(newly_generated_events): 
            if e.get("type") in ["DECODER_ACS_STEP", "DECODER_ACS_COMPLETE"]:
                primary_event = e
                break
        if not primary_event: primary_event = newly_generated_events[-1]
        
    return jsonify({
        "status": step_outcome.get("status", "success"), 
        "message": step_outcome.get("details", "ACS step performed."),
        "method_outcome": step_outcome, 
        "event": primary_event,
        "decoder_state": get_current_decoder_state_dict(decoder)
    })

@app.route('/decoder/perform_traceback', methods=['POST'])
def decoder_perform_traceback():
    global viterbi_decoder_instance, all_decoder_events
    decoder = get_or_create_decoder()
    if not decoder.is_sequence_loaded:
        return jsonify({"status": "error", "message": "No sequence loaded.", "decoder_state": get_current_decoder_state_dict(decoder)}), 400
    if not decoder.is_acs_complete:
        return jsonify({"status": "error", "message": "ACS not yet complete.", "decoder_state": get_current_decoder_state_dict(decoder)}), 400
    # if decoder.is_traceback_complete: # Allow re-running traceback
    #      return jsonify({"status": "info", "message": "Traceback already performed. Re-running.", "decoder_state": get_current_decoder_state_dict(decoder)}), 200

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
        if not primary_event: primary_event = newly_generated_events[-1]

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
    
    all_decoder_events = [] # Clear old events
    viterbi_decoder_instance = None # Force re-creation
    decoder = get_or_create_decoder() # This will create new instance, add listener, and init will fire DECODER_RESET.

    reset_event = None
    # The DECODER_RESET event is fired by ViterbiDecoderK7G171_133_Stepwise.__init__ via _reset_decoding_state
    if all_decoder_events and all_decoder_events[0].get("type") == "DECODER_RESET":
        reset_event = all_decoder_events[0]
    # If there are other events, it's unexpected here, but grab the last one as a fallback.
    elif all_decoder_events:
        reset_event = all_decoder_events[-1]


    return jsonify({
        "status": "success",
        "message": "Decoder reset.",
        "event": reset_event, # Should be the DECODER_RESET event
        "decoder_state": get_current_decoder_state_dict(decoder)
    })

def main():
    # For local development, you might want to use a different port or debug=True
    # For Firebase/Cloud Run, PORT environment variable is standard.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True) 

if __name__ == "__main__":
    main()