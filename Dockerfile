FROM python:3.8-slim-buster
WORKDIR /app
RUN apt-get update \
    && adduser --disabled-password --no-create-home userapp \
    && apt-get -y install libpq-dev \
    && apt-get -y install apt-file \
    && apt-get -y install python3-dev build-essential
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python", "main.py", "run" ]