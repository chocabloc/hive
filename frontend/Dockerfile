# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your frontend application code
COPY ./app /code/app

# Expose the default Streamlit port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "/code/app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

