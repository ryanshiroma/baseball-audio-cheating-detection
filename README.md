#  How can machine learning be used on sports broadcast audio to detect cheating?


## Preface
**background** https://en.wikipedia.org/wiki/Houston_Astros_sign_stealing_scandal

What struck me the most while watching the Astros sign stealing cheating scandal unfold was, in restrospect, how blatantly obvious the sign stealing came through in the TV broadcasts. Its just hard to imagine that not one baseball fan spotted the trash can banging for well over a year. It seems that in a world of sabermetrics and online sports betting, the sounds and buzz of the stadium are still sacred.

The bigger question I want to explore is how machine learning can be leveraged to catch instances of cheating in sports. Given that this cheating scandal happened right in front of our eyes, I wouldn't be surprised if we can find similar past cheating instances that blew right past our ears. This project serves as a 

 ## Work In Progress!!!! Analysis, more documentation, and metadata to be commited soon.

### TL;DR.

I used a Convolutional Neural Net to search for evidence of cheating in the MLB broadcast audio data. The model was able to identify (~90) of the instances of cheating without any specific training on trash can bangs sounds.

#### Steps
- Data Preparation
   - Image Data
      - Scrape MLB.com for at-bat videos and pitch information
      - Strip out audio from video clips
      - denoise the wav audio
      - Generate mel-spectrograms
   - Meta Data
      - one-hot encode batters
      - create labels using 
- Model Training
  - Split dataset into training and test datasets
  - Fit C-NN and optimize hyperparameters
  - validate model outputs
- Model Inference
  - Use SHAP to explain which part of the spectrogram contributed most towards the cheating predictions
- Analysis
  - placeholder
- Future Work
  - Repeat this analysis on other teams' broadcasts using some combination of swings/hits/pitch type as a proxy for potential cheating opportunities.
  - If the above is successful, repeat using broadcast image data rather than audio data.


## The Data

28,965 pitches to Astros batters at home games in the 2016-2018 seasons.


|batter | pitch count|
|---------|------------|
|Springer, George   |   3977|
|Altuve, Jose        |  3512|
|Correa, Carlos      |  3111|
|Gonzalez, Marwin    |  3090|
|Bregman, Alex       |  2930|
|Gattis, Evan        |  2331|
|Gurriel, Yuli       |  2155|
|Reddick, Josh       |  1918|
|Marisnick, Jake     |  1388|
|White, Tyler        |  1144|
|Kemp, Tony          |   781|
|Fisher, Derek       |   548|
|Stassi, Max         |   505|
|Tucker, Preston     | 347|
|Davis, J.D.         |   252|
|Reed, AJ            |   240|
|Hernández, Teoscar   |  180|
|Tucker, Kyle         |  167|
|Centeno, Juan        |  121|
|Federowicz, Tim      | 104|
|Worth, Danny         |   78|
|Moran, Colin         |   39|
|Straw, Myles         |    4|

*TODO: oops! I missed a couple of players! I'll be adding them shortly.*

After scraping MLB.com I compiled a dataframe of pitch data and video links
| pitch_num |pitch_type| zone | speed | batter | pitcher       | date       | count      | inning | pitch_result | pa_result     | video_id                                          |                   
|-------|------|-------|--------|---------------|---------------|------------|--------|--------------|---------------|---------------------------------------------------|------------------------------------------|
| 0     | FT   | 3.0   | 91.5   | Gurriel, Yuli | Skaggs, Tyler | 2018-09-23 | 3-2    | Bot 3        | hit_into_play | Yuli Gurriel pops out to first baseman Jefry M... | 6f710820-894a-4f5c-be7e-8a0f1380addd.mp4 |
| 1     | FF   | 6.0   | 90.9   | Gurriel, Yuli | Skaggs, Tyler | 2018-09-23 | 3-1    | Bot 3        | called_strike | Yuli Gurriel pops out to first baseman Jefry M... | 4697a2e1-0bc0-4c8c-b137-986772983193.mp4 |
| 2     | CH   | 13.0  | 84.2   | Gurriel, Yuli | Skaggs, Tyler | 2018-09-23 | 2-1    | Bot 3        | ball          | Yuli Gurriel pops out to first baseman Jefry M... | 5c216bf7-ec4b-40e8-9bfe-f5d0a5aa9086.mp4 |

**example spectrograms with trash can banging**

<p float="left">
  <img src="docs/images/bang_example1.png" width="200" />
  <img src="docs/images/bang_example2.png" width="200" />
  <img src="docs/images/bang_example3.png" width="200" />
</p>

*wear headphones!*

https://youtu.be/pUP2LHBA5Oo


**example spectrograms with no banging**

<p float="left">
  <img src="docs/images/not_bang_example1.png" width="200" />
  <img src="docs/images/not_bang_example2.png" width="200" />
  <img src="docs/images/not_bang_example3.png" width="200" />
</p>
