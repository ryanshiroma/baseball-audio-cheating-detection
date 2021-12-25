# Finding evidence of cheating in MLB broadcast audio

 ## Work In Progress!!!! Analysis, better documentation, and metadata to be commited soon.


## TL;DR.
I created a model to see if I could find any link between the type of pitch thrown and the sounds that precede the pitch. Under the assumption of no cheating, a well-generalized fitted model would find no link. However, in the case of the Astros 2017 season, the banging of a trash can was used to relay to the batter that an off-speed pitch was coming. This begged the question, if a model can identify that an off-speed pitch correlates with a trash can bang in the audio, can we identify random audio-based cheating with no prior knowledge? The results from this analysis shows that in the case of the blatant Astros cheating scheme, yes we can.



## Preface
**background** https://en.wikipedia.org/wiki/Houston_Astros_sign_stealing_scandal


Data and statistics in baseball has always been a central part of the sport. Every move of the ball and action from the players are tracked, tabulated, and compared against. With so much data available, and especially now with the introduction of MLB Statcast a few years ago, fan-based baseball analysis is everywhere.
After the Astros cheating scandal came out, tons of articles were published analyzing the effect of the cheating on their World Series win. These were all super interesting reads but what I was interested in was how, if possible, could we have identified the cheating itself and not just evidence from its result?

What struck me the most while watching the Astros sign stealing cheating scandal unfold *nearly a year after the cheating occurred* was how blatantly obvious the sign stealing came through in the TV broadcasts. The banging sounds were so clear it almost seemed like the can was mic'd up itself. I suppose in a world of instant tabulation of newer and newer sabermetrics, the sounds and buzz of the stadium are still sacred. 

Given how clear the banging sound was in the broadcast audio, it seems reasonable that a model can accurately identify the trash can banging sounds. The more interesting question to me is could this be used to flag cheating without prior knowledge that cheating is going on?






# Results


The conclusion of this project is that we successfully identified the trash can banging without prior knowledge of the banging itself.

The model identified  --XX-- cheating instances from the following players.

These results can be validated against manually tagged data from Tony Adams at http://signstealingscandal.com/.

 -- comparisons here --

A lot of things had to go right in order for the model to learn in this case:

- the trash can was either near a broadcast microphone or was extremely loud.
- The Astros did not switch up their bang signal game-to-game(bang always meant off-speed).
- They did this for full season of data giving us thousands of data points.
- They used the same trash can which gives us an almost identical noises each time.

**Two caveats that may not indicate cheating:**
- the announcers may say the pitch type out load if they are aware of whats coming next
- the model may learn from the presence of a bat sound if a player is more succesful with a certain type of pitch. This should be accounted for when validating the results.


When plotting a histogram of the predictions we see a clear clustering of a small percentage of pitches with high value predictions.

![Histogram of predictions](/docs/histogram.png)

The "bang" is seen in all of these audio spectrograms as the triangular shape at the bottom of each image. Some have more than 1 bang.
![Highest Predictions](/docs/top_15_preds.png)


[MLB Video Playlist of Top 5 Predictions](https://www.mlb.com/video/00u7yc7ivV9ndZQst356/reels/highest-offspeed-predictions)




# Project Outline
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

**add more charts here with ground truth data**

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
The traning loss/accuracy is *not* non-monotonically decreasing/increasing because I apply dropout after each epoch and subsequent epochs might have a worse loss/accuracy.


