FROM python:3.9-slim-buster

# Install scikit-learn and pandas
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg libsndfile-dev
RUN pip3 install pandas==1.5.1 librosa==0.9.2 boto3==1.20.24 pillow==9.3.0 sagemaker==2.116.0 bs4 noisereduce matplotlib lxml

# Add a Python script and configure Docker to run it
ADD baseball-audio-cheating-detection /baseball-audio-cheating-detection
ENTRYPOINT ["python3", "/baseball-audio-cheating-detection/code/1-process_data.py"]