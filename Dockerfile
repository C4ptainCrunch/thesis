FROM python:3.9-buster

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ipython.patch .
RUN patch /usr/local/lib/python3.9/site-packages/IPython/core/interactiveshell.py ipython.patch

COPY source .
CMD python simulate.py
