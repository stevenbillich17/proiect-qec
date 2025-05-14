# Flask Convolutional Encoder and Viterbi Decoder

This application implements a simple convolutional encoder and Viterbi decoder using the Flask web framework. It allows users to input a binary data string, encode it using a pre-defined convolutional code, optionally introduce noise (simulated), and then decode the (potentially noisy) data using the Viterbi algorithm to attempt to recover the original message.

## Screenshot

![App Screenshot](ScreenShotApp.png)

## Features

*   **Web Interface:** User-friendly interface built with Flask.
*   **Convolutional Encoding:** Encodes a user-provided binary string.
    *   Uses a fixed constraint length (K) and generator polynomials (as detailed in the backend code, e.g., K=7, G=["171", "133"] for a rate 1/2 code).
*   **Noise Simulation (Conceptual):** The UI allows for displaying a "received/noisy" version, though actual noise addition logic might be manual or a placeholder for future implementation.
*   **Viterbi Decoding:** Decodes the received (potentially noisy) sequence to recover the original message.
    *   Tailored to the specific encoder parameters.
*   **Clear Output:** Displays the original input, encoded output, received/noisy output, and the final decoded output.
*   **Success/Failure Indication:** Shows whether the decoded message matches the original input.

## Prerequisites

*   Python 3.7+
*   pip (Python package installer)
*   Git (for cloning the repository)

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    (Assuming you have a `requirements.txt` file. If not, you'll need to create one or install Flask manually.)
    ```bash
    pip install Flask
    # If you have a requirements.txt:
    # pip install -r requirements.txt
    ```
    *Note: Based on the provided OCR, the core encoder/decoder logic might be in separate Python files (e.g., `quantum_encoder.py`, `viterbi_decoder.py`). Ensure these are correctly imported by your Flask app file (e.g., `app.py`).*

### Running the Application

1.  **Start the Flask development server:**
    (Assuming your main Flask application file is named `app.py`)
    ```bash
    python app.py
    ```
    Or, if you've set up `FLASK_APP`:
    ```bash
    flask run
    ```

2.  **Open your web browser** and navigate to:
    ```
    http://127.0.0.1:5000/
    ```
    (The port might be different if specified in your Flask app configuration.)

## How to Use

1.  Enter a binary string (e.g., `10110`) into the "Binary Input" field.
2.  Click the "Encode & Decode" button (or similar, based on your UI).
3.  The application will:
    *   Display the "Encoded Output."
    *   Show the "Received (Noisy) Output." (You might manually alter this field if you want to simulate errors before decoding, or the app might have a feature for this).
    *   Present the "Decoded Output" from the Viterbi decoder.
    *   Indicate if the decoded message matches the original input.