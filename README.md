# Various GAN Models

GANの様々なモデルを一元管理し、コマンドラインにより使い分けを可能にする

## Supported Models

## Supported Data

## Author

Nakai, Yu. The University of Tokyo

## Set Up

* download sample data

    ```bash
    $ make
    ```

* docker build image & create container

    ```bash
    $ docker-compose up -d --build
    ```

## Environment

## Command

### docker usage

* start

    ```bash
    $ docker-compose start
    ```

* exec

    ```bash
    $ docker-compose exec python3 /bin/bash
    ```

* stop

    ```bash
    $ docker-compose stop
    ```

* restart

    ```bash
    $ docker-compose restart
    ```

* remove container

    ```bash
    $ docker-compose down
    ```

* remove image

    ```bash
    $ docker-compose down --rmi all
    ```
