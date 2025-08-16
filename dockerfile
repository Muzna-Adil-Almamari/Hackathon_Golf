# Base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
