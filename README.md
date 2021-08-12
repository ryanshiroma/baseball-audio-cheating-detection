# Finding evidence of cheating in MLB broadcast audio

 ## Work In Progress!!!! Analysis, better documentation, and metadata to be commited soon.

## Preface
**background** https://en.wikipedia.org/wiki/Houston_Astros_sign_stealing_scandal


Baseball is famously the sport with the most statistics obsessed fans. Every move of the ball and the players are tracked, tabulated, and compared which makes it a popular sport for data-lovers. As you might expect, huge communities exist online on websites like fangraphs.com and sabr.org. I've read many analysis articles online showing how the cheating may have benefited their numbers and eventual World Series win, post-hoc. (This also was extensively looked at during the foreign substance ban in 2021 and its affect on pitch spin rates :) ) What I was interested in was how, if possible, could we have identified the cheating itself, and not just evidence from its result?

Given how clear the banging sound was in the broadcast audio, I first aim to see if a model can accurately identify the specific trash can banging sounds in broadcast audio. If that is successful, can we then apply this technique to catch other teams cheating with audio cues?


What struck me the most while watching the Astros sign stealing cheating scandal unfold *nearly a year after the cheating occurred* was how blatantly obvious the sign stealing came through in the TV broadcasts. The banging sounds were so clear it almost seemed like the can was mic'd up itself. I suppose in a world of instant tabulation of newer and newer sabermetrics, the sounds and buzz of the stadium are still sacred.



## TL;DR.
I created a model to see if I could find any link between a pitch's audio and the type of pitch being thrown. Under the assumption of no cheating, a perfectly fitted model would have nothing to learn. However, since a trash can bang was used to relay to the batter that an off-speed pitch was coming, the fitted model was  able to identify the pitches with bangs to a very high accuracy and thus improve its overall pitch type prediction accuracy over random guessing.

## Analysis Outline
- Data Preparation
  - Video Data
    - Scrape MLB.com for at-bat videos
    - Strip out audio from video clips
    - denoise the audio
    - Generate mel-spectrograms images
  - Meta Data
    - Scrape MLB.com for pitch metadata
    - one-hot encode batters and month
    - create the target by binarizing off-speed and fastball pitches to 0-1
- Model Training
  - Split dataset into training and test datasets
  - Fit C-NN and optimize hyperparameters
- Model Inference and Analysis
  - Inspect the prediction distribution to determine if cheating signals are present
  - use integrated gradients (or similar) to identify the cheating sound itself
- Future Work
  - Rerun the analysis for past seasons and other teams

## The Data

### Meta Data
The data was scraped from the MLB baseball savant website https://baseballsavant.mlb.com/statcast_search using Beautiful Soup. I pulled all pitches thrown to Astros batters in home games during the 2017 seasons.


- 12,642 total pitches
- 19 players
- 45.4% of pitches were off-speed 

| batter          | percent off-speed | total |
|-----------------|-----------|-------|
| George Springer | 43.5%    | 1382  |
| Jose Altuve     | 49.1%    | 1313  |
| Alex Bregman    | 43.4%    | 1248  |
| Marwin Gonzalez | 52.1%    | 1223  |
| Josh Reddick    | 42.0%    | 1094  |
| Yuli Gurriel    | 46.3%    | 1062  |
| Carlos Beltrán  | 43.7%    | 996   |
| Carlos Correa   | 45.6%    | 961   |
| Brian McCann    | 46.5%    | 871   |
| Evan Gattis     | 53.6%    | 683   |
| Jake Marisnick  | 39.9%    | 486   |
| Norichika Aoki  | 40.2%    | 378   |
| Derek Fisher    | 40.3%    | 347   |
| Tyler White     | 43.9%    | 132   |
| Juan Centeno    | 33.3%    | 123   |
| J.D. Davis      | 42.0%    | 112   |
| Cameron Maybin  | 41.2%    | 97    |
| Tony Kemp       | 25.5%    | 55    |
| Max Stassi      | 36.5%    | 52    |
| AJ Reed         | 44.4%    | 27    |

Three example rows in the table:

| pitch | pitcher          | batter           | count | date     | dist | exit_velocty | inning | launch_angle | mph  | pitch_id                             | pitch_result | spin_rate | zone |
|-------|------------------|------------------|-------|----------|------|--------------|--------|--------------|------|--------------------------------------|--------------|-----------|------|
| SL    | Kershaw, Clayton | Springer, George | 2-Mar | 10/29/17 | NaN  | NaN          | Bot 5  | NaN          | 88   | 27df5d03-0f01-495b-8528-b8a277c30c2b | ball         | 2613      | 13   |
| FF    | Kershaw, Clayton | Springer, George | 2-Mar | 10/29/17 | 222  | 100.7        | Bot 5  | 14           | 93.2 | 1dd2e806-8a45-4433-a7fa-64b637537161 | foul         | 2345      | 4    |
| SL    | Kershaw, Clayton | Springer, George | 2-Mar | 10/29/17 | 259  | 85.4         | Bot 5  | 40           | 89.1 | ef5299e9-c8af-4c49-bf89-0611c311e6f6 | foul         | 2578      | 9    |

### Video Data
The pitch_id from the metadata then used to download the actual video file for that pitch using `wget`. For example, the first pitch in the table can be downloaded here:
https://sporty-clips.mlb.com/27df5d03-0f01-495b-8528-b8a277c30c2b.mp4

From this video file, I then extracted the audio using `ffmpeg`, denoised the audio with `noise_reduce` and then converted to a mel-spectrogram using `librosa`.
Below are afew examples of the outputed spectrograms.

![spec1](/docs/0d9d5cfb-8f69-4cf5-88ec-60d921e5ad3e.png)
![spec1](/docs/0cf1f959-3566-456e-9f4d-26a4a33b3790.png)
![spec1](/docs/0d3fde58-b2e7-43a9-bfc0-6f591ec41f46.png)

## Model Training

I started with a commonly used Convolutional neural network model structure as a baseline and then tweaked the number of filters in the convolutional layers and nodes in the dense layers and chose the model with the best validation AUC.
```
# input for the images
input_1 = keras.Input(shape=input_shape)

# convolutional layer 1
conv1 = layers.Conv2D(64, (3, 3), activation='relu')(input_1)
mp1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
drop1 = layers.Dropout(0.1)(mp1)

# convolutional layer 2
conv2 = layers.Conv2D(64, (3, 3), activation='relu')(drop1)
mp2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
drop2 = layers.Dropout(0.25)(mp2)

# convolutional layer 3
conv3 = layers.Conv2D(128, (3, 3), activation='relu')(drop2)
mp3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
drop3 = layers.Dropout(0.1)(mp3)

# dense layer for the images
flat = layers.Flatten()(drop2)
dense1 = layers.Dense(124, activation='relu')(flat)

# input for the metadata
input_2 = keras.Input(shape = (19,))

# dense layer for the metadata
dense2 = layers.Dense(4, activation='relu')(input_2)

# dense layer for the concatenated image and metadata dense layers
combined = tf.keras.layers.Concatenate()([dense1,dense2])
dense3 = layers.Dense(128, activation='relu')(combined)
drop3 = layers.Dropout(0.1)(dense3)

# single class output
output = layers.Dense(1, activation='sigmoid')(drop3)
```


![Model Diagram](/trained_models/model_diagram.png)

The model stabilized after around 40 epochs and showed no signs of overfitting.
![Loss](/docs/loss.png)

![Accuracy](/docs/accuracy.png)

### Results

Since the the audio and the metadata are expected to contain **no** valuable information about the pitch about to be thrown, we should not expect a good generalized model to learn anything. A model that *shows* learning, however, may indicate cheating. In the case of the 2017 Astros, audio cues(i.e. trash can banging) would indeed let the model learn.

**Two caveats that may not indicate cheating:**
- the announcers may say the pitch type out load if they are aware of whats coming next
- the model may learn from the presence of a bat sound if a player is more succesfull with a certain type of pitch.


When plotting a histogram of the predictions we see a clear clustering of a small percentage of pitches with high value predictions.

![Histogram of predictions](/docs/histogram.png)

The "bang" is seen in all of these audio spectrograms as the triangular shape at the bottom of each image. Some have more than 1 bang.
![Highest Predictions](/docs/top_15_preds.png)


[MLB Video Playlist of Top 5 Predictions](https://www.mlb.com/video/00u7yc7ivV9ndZQst356/reels/highest-offspeed-predictions)

