# 1. Use the official lightweight Python 3.13 image
FROM python:3.13-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /code

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the application code
COPY ./app ./app

# (Removed Step 6: Custom User Creation)
# The container will now default to running as 'root'.

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/ml-api"]