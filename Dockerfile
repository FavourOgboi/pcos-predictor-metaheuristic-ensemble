FROM python:3.12-slim

WORKDIR /app

# Set Streamlit environment variables to suppress prompts and configure for production
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_LOGGER_LEVEL=error \
    STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true \
    STREAMLIT_BROWSER_COLLECT_USAGE_STATS=false

# Copy requirements first for better caching
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Expose port
EXPOSE 8501

# Run streamlit with explicit flags to suppress interactive prompts
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=error"]
