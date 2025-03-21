# tolk-proto

This repository contains a prototype application. Follow the steps below to set it up and run it.

## Prerequisites

Ensure you have [Poetry](https://python-poetry.org/) installed on your system.

## Installation

1. Install the necessary dependencies:
    ```bash
    poetry install
    ```

2. Initialize the "database":
    ```bash
    poetry run python app/embed_knowledge.py
    ```

3. Start the development server:
    ```bash
    poetry run fastapi dev app/main.py
    ```

3. Fill up the out-of-scope "database" through the API (don't close the server!):
    ```bash
    poetry run python app/user_log_upload.py
    ```

4. Execute the streamlit app with:
    ```bash
    poetry run streamlit run streamlit_app.py
    ```

## Usage

Once the server is running, you can access the API documentation at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The streamlit app is accessible at:

[http://localhost:8501/](http://localhost:8501/)
