# Use a Python 3.11 lightweight image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install dependencies
# Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# This step installs all dependencies into the container's isolated Python environment, effectively creating your server-side "venv"
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application files
# The FastAPI deployment script
COPY fastapi_deployment.py .

# NOTE: Ensure you have a 'random_forest_regressor_model.pkl' file next to the Dockerfile.
COPY random_forest_regressor_model.pkl .

# 3. Expose the port that Uvicorn will listen on
EXPOSE 8000

# 4. Define the command to run the application using Uvicorn
CMD ["uvicorn", "fastapi_deployment:app", "--host", "0.0.0.0", "--port", "8000"]