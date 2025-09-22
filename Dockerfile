# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files and folders into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: expose Streamlit port
EXPOSE 8501

# Streamlit environment fixes (disable browser launch)
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
