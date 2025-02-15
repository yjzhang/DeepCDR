FROM tensorflow/tensorflow:1.13.2-gpu
MAINTAINER Yue Zhang <yue.zhang@isbscience.org>

WORKDIR /DeepCDR

ADD requirements.txt /DeepCDR

RUN pip install -r requirements.txt

ADD . /DeepCDR

#CMD ["python2", "run_prod.sh"]
