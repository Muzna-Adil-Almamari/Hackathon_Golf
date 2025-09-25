# ython base image
FROM python:3.10-slim

# working directory
WORKDIR /app

# Install system dependencies (needed for geopandas, shapely, fiona, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libspatialindex-dev \
    gdal-bin \
    libgdal-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
    pip install --no-cache-dir \
    streamlit \
    folium \
    pandas \
    shapely \
    geopandas \
    streamlit-folium \
    fpdf \
    numpy \
    scikit-learn \
    scipy \
    networkx \
    requests \
    haversine

# Streamlit config: prevent asking for email
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
