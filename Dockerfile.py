# Use a base image with Python 3.9-slim for compatibility with your setup
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for faiss-cpu, hf_xet, and other libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy application files
COPY app.py .
COPY flipkart_laptop_cleaned_new.csv .
# Optionally copy .env file as a fallback (not recommended for production)
COPY .env .env

# Expose the port for Gradio
EXPOSE 7860

# Set TOKENIZERS_PARALLELISM to suppress HuggingFace warning
ENV TOKENIZERS_PARALLELISM=false

# Run the application
CMD ["python3", "app.py"]