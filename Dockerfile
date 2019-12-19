FROM python:3.7
ENV  PIP_INDEX_URL=http://nexus.dtni.macif.fr/repository/pypi/simple/
ENV  PIP_TRUSTED_HOST=nexus.dtni.macif.fr

LABEL Name=deep Version=0.0.1

VOLUME /tmp/.X11-unix
ENV DISPLAY :0
ENV PYTHONIOENCODING=utf8
ENV QT_X11_NO_MITSHM=1

WORKDIR /app
ADD . /app

# Using pip:
RUN python3 -m pip install -r requirements.txt
CMD ["tail", "-f", "/dev/null"]

