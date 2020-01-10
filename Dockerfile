FROM python:3.7
LABEL Name=deep Version=0.0.1

VOLUME /tmp/.X11-unix
ENV DISPLAY :0
ENV PYTHONIOENCODING=utf8
ENV QT_X11_NO_MITSHM=1

ENV HTTP_PROXY http://proxy.global.logica.com
ENV HTTPS_PROXY http://proxy.global.logica.com
# ARG HTTP_PROXY 
# ARG HTTPS_PROXY


WORKDIR /app
ADD . /app

# Using pip:
RUN python3 -m pip install -r requirements.txt
CMD ["tail", "-f", "/dev/null"]

