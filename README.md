# Melbourne Housing Project

The project includes a Flask API backend, a Streamlit interactive frontend, a CI/CD pipeline, and a data drift monitoring system.

## Docker run

The easiest and fastest way to run the entire project is using Docker.

### Prerequisites
- **Docker** and **Docker Compose** installed on your machine.

### Startup Instructions
1. Open your terminal in the project's root folder.
2. Run the following command:

   docker-compose up --build

3. Wait a few moments.
Note: On the first run, if no saved model is found, the system will automatically perform a "Cold Start" training using melb_data.csv. The Frontend will automatically wait for the API to be healthy before starting.

4. Open your browser at: http://localhost:8501

Access Credentials
The application features two user roles:

- Home Seeker: Free access to search for properties.
- Real Estate Agent: Restricted area to manage the database and monitor the market.
    - Password for Agent Area: admin

**System Architecture**

| Service | Technology | Port | Description |
| :--- | :--- | :--- | :--- |
| **Frontend** | Streamlit | `8501` | User interface for search and inventory management. |
| **Backend (API)** | Flask | `8000` | Exposes endpoints for predictions, data management, and metrics. |
| **Locust** | Locust | `8089` | (Optional) Tool for load testing and performance metrics. |

## Run Manually

Prerequisites: 

- Python 3.12 installed
- Pip (Python package manager)

### Install Dependencies 
Open a terminal in the project root and install the required libraries:

pip install -r requirements.txt

### Start the Backend (API) 

Run the Flask API in the first terminal window:

python api.py

The server will start at http://localhost:8000.

### Start the Frontend (Streamlit) 

Open a new terminal window (keep the API running in the first one) and run:

streamlit run frontend.py

A browser window should automatically open at http://localhost:8501