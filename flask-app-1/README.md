# Flask Pose Estimation Application

This project is a Flask web application that performs real-time pose estimation using a webcam feed. It classifies the user's posture as either "Good Posture" or "Bad Posture" and allows for the recording of normal and abnormal posture data.

## Project Structure

```
flask-app
├── static
│   └── css
│       └── styles.css       # CSS styles for the application
├── templates
│   └── index.html           # Main HTML template for the application
├── data
│   ├── normal               # Directory to store recorded normal data files
│   └── abnormal             # Directory to store recorded abnormal data files
├── app.py                   # Main application file
└── README.md                # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd flask-app
   ```

2. **Install Dependencies**
   Make sure you have Python and pip installed. Then, install the required packages:
   ```bash
   pip install flask opencv-python mediapipe torch torchvision
   ```

3. **Run the Application**
   Start the Flask application by running:
   ```bash
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000`.

## Usage

- The main page displays a real-time video stream from the webcam.
- The application classifies the user's posture and displays the result on the screen.
- Four buttons are provided for recording:
  - **Start Normal Recording**: Begins recording normal posture data.
  - **Save Normal Data**: Saves the recorded normal data to the `data/normal` directory.
  - **Start Abnormal Recording**: Begins recording abnormal posture data.
  - **Save Abnormal Data**: Saves the recorded abnormal data to the `data/abnormal` directory.
- The application displays the number of files recorded in each directory.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.