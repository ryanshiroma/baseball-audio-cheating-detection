import pandas as pd
import numpy as np
import os.path
from PIL import Image
import requests
from bs4 import BeautifulSoup
import lxml
import os
import subprocess
import urllib.request
import pandas as pd
from urllib.parse import urlencode
import yaml
import datetime
import librosa
from librosa.display import specshow
import noisereduce as nr
import matplotlib.pyplot as plt


# path settings
BASEBALL_SAVANT_BASE_URL = 'https://baseballsavant.mlb.com'
VIDEO_CLIP_BASE_URL = 'https://sporty-clips.mlb.com/'
RAW_DATA_PATH = '../data/raw'
PITCH_TABLE_PATH = '../data'
PROCESSED_IMAGE_PATH = '../data/processed'





def download_data(config: dict) -> pd.DataFrame:

    # if not os.path.exists(os.path.join(PITCH_TABLE_PATH,'pitch_table.csv')):
    df=pd.DataFrame(columns=['pitch','mph','exit_velocty','pitcher','batter','dist','spin_rate',
                            'launch_angle','zone','date','count','inning','pitch_result','pitch_id'])

    # loop through each batter
    for batter in config['batter_ids']:

        param_dict={
                'home_road':config['home_road'],
                'player_type':'batter',
                'type': 'details',
                'game_date_gt': config['date_from'],
                'game_date_lt': config['date_to'],
                'player_id':batter,
                'team':','.join(config['teams'])}

        params=urlencode(param_dict)
        url =  BASEBALL_SAVANT_BASE_URL+'/statcast_search?' + params
        page = requests.get(url)
        soup = BeautifulSoup(page.text,"lxml" )

        i=0
        for table in soup.find_all('table'):
            for pitch in table.find_all('tr'):
                elements=pitch.find_all('td')
                if len(elements) != 0:
                    try:
                        play = BeautifulSoup(requests.get(BASEBALL_SAVANT_BASE_URL+elements[14].a['href']).text,"lxml")
                        pitch_id=play.find_all('video')[0].source['src'].split('/')[-1].split('.')[0]
                        # print(pitch_id)
                        if len(pitch_id)>1:
                            if not os.path.exists(os.path.join(RAW_DATA_PATH,'video',pitch_id+'.mp4')):
                                urllib.request.urlretrieve(VIDEO_CLIP_BASE_URL+pitch_id+'.mp4', os.path.join(RAW_DATA_PATH,'video',pitch_id+'.mp4'))
                            
                            if os.path.exists(os.path.join(RAW_DATA_PATH,'video',pitch_id+'.mp4')):
                                df = df.append({
                                    'pitch':elements[0].text,
                                    'mph':elements[1].text,
                                    'exit_velocty':elements[2].text,
                                    'pitcher':elements[3].text,
                                    'batter':elements[4].text,
                                    'dist':elements[5].text,
                                    'spin_rate':elements[6].text,
                                    'launch_angle':elements[7].text,
                                    'zone':elements[8].text,
                                    'date':elements[9].text,
                                    'count':elements[10].text,
                                    'inning':elements[11].text,
                                    'pitch_result':elements[12].text,
                                    'pitch_id':pitch_id},ignore_index=True)

                            df.to_csv(os.path.join(PITCH_TABLE_PATH,'pitch_table_temp.csv'),index=False)
                    except:
                        pass
                    i=i+1
    df.to_csv(os.path.join(PITCH_TABLE_PATH,'pitch_table.csv'),index=False)
    return df


def process_data(pitch_ids: list,
                 keep_wavs: bool=False):

    # mel spectrogram settings
    duration = 5
    sr = 44100
    fmax = 2000
    nr_threshold = 0.5
    n_mels=128
    n_fft=8192
    hop_length=2048

    # create audio folder path if it doesn't exist
    if not os.path.exists(os.path.join(RAW_DATA_PATH,'audio')):
        os.makedirs(os.path.join(RAW_DATA_PATH,'audio'))


    for pitch_id in pitch_ids:
        print(pitch_id)
        video_file = os.path.join(RAW_DATA_PATH,'video', pitch_id + '.mp4')
        audio_file = os.path.join(RAW_DATA_PATH,'audio', pitch_id + '.wav')
        image_file = os.path.join(PROCESSED_IMAGE_PATH, pitch_id + '.png')

        #### first check if the image already exists and skip if so
        if os.path.exists(image_file):
            print('image file: {image_file} already exists'.format(image_file=image_file))
            continue

        #### check that there is a video file to process
        if not os.path.exists(video_file):
            print('video file: {video_file} does not exist'.format(video_file=video_file))
            continue

        #### extract the audio from the video file
        if not os.path.exists(audio_file):
            command = "ffmpeg -i " + video_file + " -vn -acodec pcm_s16le -ar 44100 -ac 1 -loglevel quiet -stats " + audio_file
            print(subprocess.call(command, shell=True))


        #### convert the wav to the mel-spectrogram
        y, sr = librosa.load(audio_file,sr=sr,offset=0,duration = duration)[:sr*duration]

        # remove noise from audio
        reduced_noise_y = nr.reduce_noise(y = y, sr=sr, n_std_thresh_stationary=nr_threshold,stationary=True)

        # extend short clips to the duration in seconds
        full_length = sr*duration
        reduced_noise_y = np.hstack((reduced_noise_y,np.zeros(full_length-len(reduced_noise_y))))
        s = librosa.feature.melspectrogram(y=reduced_noise_y, sr=sr, n_mels=n_mels,n_fft=n_fft,hop_length=hop_length,fmax=fmax)
        fig,ax = plt.subplots(1)
        ys,xs=s.shape
        fig.set_size_inches(xs/400, ys/400)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.axis('off')
        specshow(librosa.amplitude_to_db(s,ref=np.max),y_axis='mel', fmax=fmax,x_axis='time',ax=ax,cmap='gray_r')
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.savefig(image_file, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

        if not keep_wavs:
            os.remove(audio_file)



def get_pitch_table() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PITCH_TABLE_PATH,'pitch_table.csv'))
    df['label']=df['pitch'].isin(['CU','SL','CH','KC','FC','EP'])*1
    return df


def get_images(pitch_ids: list):
    images=[]
    for pitch_id in pitch_ids:
        # load the image
        path = os.path.join(PROCESSED_IMAGE_PATH, pitch_id + '.png')
        image = Image.open(path).convert('L')
        images.append(np.asarray(image).reshape(128, -1, 1).astype(np.float16))
    return np.array(images)
