FROM python:3.7.3

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 5000
CMD [ "python" , "app.py"]
