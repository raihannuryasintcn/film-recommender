# Film Recommender

This project is a film recommender system that provides movie recommendations for users based on their past ratings. It uses a collaborative filtering approach with the SVD algorithm. The application is built with Flask and can be accessed through a web interface or a REST API.

## Features

*   **User-based Recommendations:** Get movie recommendations for a specific user.
*   **User Statistics:** View a user's rating statistics and their top-rated movies.
*   **REST API:** Access the recommendation engine programmatically.
*   **Web Interface:** A simple web interface to interact with the recommender.

## Technologies Used

*   **Python:** The core programming language.
*   **Flask:** A micro web framework for the web application and API.
*   **Pandas:** For data manipulation and analysis.
*   **scikit-surprise:** A Python library for building and analyzing recommender systems.
*   **Joblib:** For saving and loading the trained model.
*   **Conda:** For environment management.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/film-recommender.git
    cd film-recommender
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f recommender-env.yml
    conda activate recommender-env
    ```

3.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The application will be running at `http://127.0.0.1:5000`.

## Usage

### Web Interface

Navigate to `http://127.0.0.1:5000` in your web browser. You can select a user ID from the dropdown menu to view their information and get movie recommendations.

### API Endpoints

*   **GET /api/users**
    *   Returns a list of all available user IDs.

*   **GET /api/user/<user_id>/info**
    *   Returns information about a specific user, including their total number of ratings, average rating, and a list of their top 10 rated movies.

*   **GET /api/recommend/<user_id>**
    *   Returns a list of 10 movie recommendations for the specified user.

## File Descriptions

*   **`app.py`**: The main Flask application file. It contains the API endpoints and the logic for serving the web interface.
*   **`recommender.ipynb`**: A Jupyter Notebook that was used to train the SVD model.
*   **`recommender-env.yml`**: The Conda environment file, which lists all the dependencies required to run the project.
*   **`models/svd_model.joblib`**: The pre-trained SVD model.
*   **`data/`**: This directory contains the MovieLens dataset files (`movies.csv` and `ratings.csv`).
*   **`templates/index.html`**: The HTML template for the web interface.
*   **`.gitignore`**: Specifies which files and directories to ignore in Git.
*   **`.gitattributes`**: A Git attributes file.
