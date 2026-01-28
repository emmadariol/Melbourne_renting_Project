# Use lightweight Python 3.12 base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (including model artifacts)
COPY . .

# METRICS SETUP: Create the log file and set permissions
# This ensures the Flask app can write logs to the file immediately
RUN touch app_metrics.log && chmod 666 app_metrics.log


# Expose the port used by the API
EXPOSE 8000

# Start the application
CMD ["python", "api.py"]