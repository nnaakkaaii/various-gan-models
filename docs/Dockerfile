FROM python:3

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP.ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN mkdir /code
WORKDIR /code

ADD ../requirements.txt ./
ADD ../general_gan ./general_gan
ADD ../scripts ./scripts

ENV PYTHONPATH .

RUN chmod +rwx .
RUN pip3 install -U pip
RUN pip3 install -U setuptools
RUN pip3 install -r requirements.txt