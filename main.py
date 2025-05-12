import os
import json

from flask import Flask, send_file, request, jsonify

from src.quantum.quantum import ConvolutionalEncoder

app = Flask(__name__)

encoder = None
message_to_encode = []
encoder_messages = []
message_index = 0 # To track the current bit being processed

def encoder_listener(event_data):
    """Simple listener to collect encoder events."""
    global encoder_messages
    event_type = event_data.get('type', 'UNKNOWN_EVENT')
    encoder_messages.append({'event_type': event_type, 'data': event_data.get('data')})
    print(f"Encoder Event: {event_type}, Data: {event_data.get('data')}") # For server-side debugging



@app.route("/")
def index():
    return send_file('src/index.html')

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

@app.route('/set_generator', methods=['POST'])
def set_generator():
    global encoder, message_to_encode, encoder_messages
    data = request.get_json()
    generators_str = data.get('generators', '')
    message_str = data.get('message', '')

    try:
        generators = [g.strip() for g in generators_str.split(',') if g.strip()]
        message_to_encode = [int(bit) for bit in message_str.replace(' ', '')]
        
        # Assuming a default constraint length for simplicity, or you could add an input for it
        constraint_length = 3 # Example constraint length

        # Pad the message with constraint_length - 1 zeros for flushing
        padding_length = constraint_length - 1
        message_to_encode.extend([0] * padding_length)
        
        encoder_messages = [] # Clear previous messages
        encoder = ConvolutionalEncoder(generators_octal=generators, constraint_length=constraint_length)
        encoder.add_listener(encoder_listener)
        encoder.load_message(message_to_encode)

        return jsonify({"status": "success", "message": "Encoder initialized."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/next_step', methods=['POST'])
def next_step():
    global encoder, encoder_messages
    global message_index # Use the global index

    if encoder is None:
        return jsonify({"status": "error", "message": "Encoder not initialized. Set generator first."})

    if message_index >= len(message_to_encode):
        return jsonify({"status": "complete", "message": "Encoding complete.", "messages": encoder_messages})

    try:
        output_bit = encoder.go_to_next_step()
        message_index += 1 # Move to the next bit for the next step
        return jsonify({"status": "success", "output_bit": output_bit, "messages": encoder_messages})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    main()
