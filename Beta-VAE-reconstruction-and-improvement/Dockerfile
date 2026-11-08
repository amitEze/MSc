# Use the exact Python version requested by the authors
FROM python:3.6.4-slim

# Set the working directory inside the container
WORKDIR /workspace

# UPGRADE PIP FIRST: The default pip in this old image is v9.0.3, 
# which is too old to find the correct PyTorch wheels.
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8097 for the Visdom server
EXPOSE 8097

# Default command to keep the container running interactively
CMD ["/bin/bash"]