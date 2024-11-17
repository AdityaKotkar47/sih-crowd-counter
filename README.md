# Crowd Counter Model 👥

This repository provides a YOLOv8-based model to count the number of people in an image. It is designed for applications such as crowd density monitoring and live CCTV analysis.

## Features
- Counts people in an image using YOLOv8.
- Accepts images in raw byte or base64 format.

## How to Use
1. Clone the repository.
2. Set up a venv
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server
    ```bash
    uvicorn app:app --reload
    ```
5. Open a new terminal and run test
    ```bash
    curl.exe -X GET http://localhost:8000/
    ```
    ```bash
    curl.exe -X POST -F "file=@image.png" http://localhost:8000/predict/
    ```
