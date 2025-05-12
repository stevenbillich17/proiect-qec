
from quantum import ConvolutionalEncoder
from helpers import explain_generator_taps


if __name__ == '__main__':
    print("===== Encoder Setup Explanation =====")
    target_constraint_length = 3  # K=3
    target_generators_octal = ["7", "5"] # G1=111, G2=101
    
    print("\n===== Step-by-Step Encoding with Listener (Class-based) =====")
    encoder = ConvolutionalEncoder(constraint_length=target_constraint_length, 
                                   generators_octal=target_generators_octal)


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

    print("\n\n===== Batch Encoding Test (uses same underlying logic) =====")
    # Using a new message or resetting then reloading for a clean batch test
    encoder.reset() # Full reset before new batch encoding
    message_batch = [1, 0, 1, 1, 0, 0] # Same message
    print(f"Encoding message via batch 'encode()' method: {message_batch}")
    # Listeners will still fire during batch encode because it uses load_message and go_to_next_step
    encoded_output_batch = encoder.encode(message_batch)
    print(f"\nFinal Encoded Output (from batch 'encode()'): {encoded_output_batch}")
    print(f"Expected Output                               : {expected_output}")
    assert encoded_output_batch == expected_output, "Batch output mismatch!"

    print("\n\n===== Test Empty Message =====")
    encoder.load_message([])
    empty_output = encoder.get_current_encoded_data()
    print(f"Encoded output for empty message: {empty_output}")
    assert empty_output == []

    print("\n\n===== Test Resetting Encoder =====")
    encoder.load_message([1,1]) # Load something
    encoder.go_to_next_step()   # Process one bit
    print(f"Memory before reset: {encoder.get_current_memory()}")
    print(f"Buffer before reset: {encoder.get_current_encoded_data()}")
    encoder.reset()
    print(f"Memory after reset: {encoder.get_current_memory()}")
    print(f"Buffer after reset: {encoder.get_current_encoded_data()}")
    assert encoder.get_current_memory() == [0] * (target_constraint_length -1)
    assert encoder.get_current_encoded_data() == []
    assert not encoder.is_message_loaded

    print("\nAll tests passed successfully!")