# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/backend

# Copy backend files
COPY backend/ /app/backend/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
