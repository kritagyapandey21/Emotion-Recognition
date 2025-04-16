# Emotion Classifier GUI Application

This is a Python-based graphical user interface (GUI) application for classifying emotions. It uses Tkinter for the interface and Fernet encryption (from the `cryptography` library) to securely handle inter-process communication between a client and server.

## Key Features

- User-friendly GUI built with Tkinter
- Secure communication using socket programming
- Encryption and decryption with Fernet
- Emotion classification based on user input
- Modular structure for easy extension

## Technologies Used

- Python 3.x
- Tkinter (for GUI development)
- Socket (for networking)
- Cryptography (Fernet for encryption)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step-by-Step Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/emotion-classifier-gui.git
    cd emotion-classifier-gui
    ```

2. Install required packages:
    ```bash
    pip install cryptography
    ```

3. Run the application:
    ```bash
    python emotion_gui.py
    ```

## Important Notes

- This application is intended to be run on a local machine with a graphical display environment.
- It will not work in headless environments such as Google Colab or remote servers without display support.
- If you are using Linux via SSH, consider using `x11 forwarding` or setting up a virtual display with `xvfb`.

## How It Works

1. The user interacts with the GUI to input data.
2. The data is encrypted and sent to a server using socket communication.
3. The server decrypts the data, processes it, classifies the emotion, and sends an encrypted response.
4. The GUI displays the final output to the user.

## Future Improvements

- Integrate with a machine learning model for more accurate emotion detection
- Add support for multiple clients
- Extend GUI to include audio or image-based emotion recognition
- Convert to a web-based interface using Flask or Streamlit

## Mentor

**Mr. Anand Kumar**  
Assistant Professor  
Lovely Professional University  
Phagwara, Punjab

## Author

**Kritagya Pandey**  
Student  
Lovely Professional University

## License

This project is open source and available under the MIT License.
