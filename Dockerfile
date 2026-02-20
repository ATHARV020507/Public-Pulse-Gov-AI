FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code
RUN chmod -R 777 /code
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]