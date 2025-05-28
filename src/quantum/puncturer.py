# src/quantum/puncturer.py
import collections # Not strictly needed for Puncturer, but was in your example

class Puncturer:
    def __init__(self, puncturing_matrix, rate_label="Custom"):
        """
        Initializes the puncturer with a given puncturing matrix.

        Args:
            puncturing_matrix (list of lists): The puncturing matrix.
                                               Rows correspond to encoder outputs (G1, G2, ...).
                                               Columns correspond to the time instances within a period.
            rate_label (str): A human-readable label for this puncturing scheme.
        """
        if not puncturing_matrix or not isinstance(puncturing_matrix, list) or \
           not all(isinstance(row, list) for row in puncturing_matrix) or \
           not all(all(bit in [0,1] for bit in row) for row in puncturing_matrix):
            raise ValueError("Puncturing matrix must be a list of lists containing 0s and 1s.")
        
        self.puncturing_matrix = puncturing_matrix
        self.num_output_streams = len(puncturing_matrix)
        if self.num_output_streams == 0:
            raise ValueError("Puncturing matrix cannot have zero rows (output streams).")
            
        self.period = len(puncturing_matrix[0])
        if self.period == 0:
            raise ValueError("Puncturing matrix columns (period) cannot be zero.")

        if not all(len(row) == self.period for row in puncturing_matrix):
            raise ValueError("All rows in the puncturing matrix must have the same length (period).")

        self.rate_label = rate_label
        self.total_ones_in_period = sum(sum(row) for row in puncturing_matrix)
        self.current_period_index = 0 # To track position within the puncturing period for stream processing

    def reset_period_tracker(self):
        """Resets the internal period index, e.g., when starting a new sequence."""
        self.current_period_index = 0

    def puncture_symbol(self, unpunctured_symbol_bits):
        """
        Applies puncturing to a single symbol (e.g., (v1, v2) from a rate 1/2 encoder).

        Args:
            unpunctured_symbol_bits (list or tuple): A list/tuple of bits, one for each output stream
                                                     of the mother encoder (e.g., [v1, v2]).

        Returns:
            list: A list of bits that are kept after puncturing for this symbol.
                  The internal period index is advanced.
        """
        if len(unpunctured_symbol_bits) != self.num_output_streams:
            raise ValueError(f"Input symbol length {len(unpunctured_symbol_bits)} "
                             f"does not match puncturer's expected_streams {self.num_output_streams}.")

        punctured_for_this_symbol = []
        col_index = self.current_period_index

        for stream_idx in range(self.num_output_streams):
            if self.puncturing_matrix[stream_idx][col_index] == 1:
                punctured_for_this_symbol.append(unpunctured_symbol_bits[stream_idx])
        
        self.current_period_index = (self.current_period_index + 1) % self.period
        return punctured_for_this_symbol

    def puncture_stream(self, unpunctured_symbols_stream):
        """
        Applies puncturing to a stream of unpunctured symbols.
        Resets period tracker before starting.

        Args:
            unpunctured_symbols_stream (list of lists/tuples): 
                e.g., [[v1,v2], [v1,v2], ...]

        Returns:
            list: A flat list of punctured bits.
        """
        self.reset_period_tracker()
        punctured_output_flat = []
        for symbol in unpunctured_symbols_stream:
            punctured_output_flat.extend(self.puncture_symbol(symbol))
        return punctured_output_flat

    def get_info(self):
        return {
            "matrix": self.puncturing_matrix,
            "label": self.rate_label,
            "period": self.period,
            "num_streams": self.num_output_streams,
            "ones_in_period": self.total_ones_in_period
        }

# Predefined puncturing matrices for convenience
# Assuming a mother code rate of 1/n_streams (e.g., 1/2 if num_output_streams is 2)
# The effective rate becomes (bits_in_period_of_input_to_mother_encoder * 1) / total_ones_in_puncturing_matrix_period
# Or, more simply for a rate 1/N mother coder, effective rate = (1/N) * (N * period / total_ones_in_period) = period / total_ones_in_period
# For a typical rate 1/2 mother coder (2 output streams):
# Numerator of rate (e.g., 2 in 2/3) is often related to period.
# Denominator (e.g., 3 in 2/3) is often related to total_ones_in_period.
PREDEFINED_PUNCTURING_SCHEMES = {
    "NONE": { # Represents no puncturing, mother code rate 1/2
        "label": "None (Rate 1/2)", # Assuming 2 output streams from mother encoder
        "matrix": [[1], [1]],
        "mother_code_output_streams": 2
    },
    "2/3": {
        "label": "Rate 2/3",
        "matrix": [[1, 0], [1, 1]], # Period 2, 3 ones. (2 input bits -> 3 output bits)
        "mother_code_output_streams": 2
    },
    "3/4": { # Example, not in your original list but common
        "label": "Rate 3/4",
        "matrix": [[1,1,0], [1,0,1]], # Period 3, 4 ones. (3 input bits -> 4 output bits)
        "mother_code_output_streams": 2
    },
    "4/5": {
        "label": "Rate 4/5",
        "matrix": [[1, 1, 1, 0], [1, 0, 0, 1]], # Period 4, 5 ones
        "mother_code_output_streams": 2
    },
    "5/6": {
        "label": "Rate 5/6",
        "matrix": [[1, 0, 1, 0, 1], [1, 1, 0, 1, 0]], # Period 5, 6 ones
        "mother_code_output_streams": 2
    },
    "7/8": {
        "label": "Rate 7/8",
        "matrix": [[1, 0, 0, 0, 1, 0, 1], [1, 1, 1, 1, 0, 1, 0]], # Period 7, 8 ones
        "mother_code_output_streams": 2
    },
}