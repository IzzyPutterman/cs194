# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.6-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY requirements.txt ./requirements.txt
#RUN sed -i 's/cu101/cpu/' requirements.txt
#RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

# Copy only the relevant directories
COPY api_server/ ./api_server

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api_server/app.py
