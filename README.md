# Finding evidence of cheating in MLB broadcast audio


## TL;DR.
The original intent of this project was to see if I could train a model to identify the trash can banging *but* without doing any trash can banging labeling. This would mimic a real life scenario of a front office analyst trying to detect audio based cheating in general. To do this, I would try to train a model on pitch type only(no trash can bang labels) and see if it could do better than random. In the end, the model was successfully able to perform better than random and also point out exactly where in the audio the cheating was being being detected. See video below for 20 minutes of suspected bangs! 

[![youtube link](https://img.youtube.com/vi/wWOyXkG35tk/0.jpg)](https://www.youtube.com/watch?v=wWOyXkG35tk)
![shap3](baseball-audio-cheating-detection/docs/shap_5.png)
## Preface

What struck me the most while watching the Astros sign stealing cheating scandal unfold *nearly a year after the cheating occurred* was how in retrospect, how blatantly obvious the sign stealing came through in the TV broadcasts. The banging sounds were so clear it almost seemed like the can was mic'd up itself. Even in a world of instant tabulation of newer and newer sabermetrics, in the spirit of baseball, quantifying the sounds and buzz of the stadium still feels wrong.



## Sagemaker code
 I also took this project as an opportunity to learn AWS Sagemaker so the codebase for each of the steps below are contained in a separate repo.

https://github.com/ryanshiroma/sagemaker-baseball

# Results


I've also validated my results against Tony Adams manually tagged data from his website, http://signstealingscandal.com/.
In summary, the model was a bit conservative in its predictions using a threshold of 0.5 and captured only about half of all of the bangs but meanwhile only missclassified predicted bangs for a small fraction indicating that a smaller threshold that balances out the metrics would be better. However, the point of this project wasn't necesarily to find all bangs, but to identify the existence of cheating and relay that to an expert baseball analyst for review.

- Recall: 0.482
- Precision: 0.889
- Accuracy: 0.917
- ROCAUC: 0.757


|   | Positive (Actual)  |Negative(Actual)    | Unknown (Actual)
|---|---|---|---|
| Positive(Pred)  | 550 (TP)  |69 (FP)  | 173
| Negative (Pred)  |  590 (FN)| 6757 (TN)  | 2912






# Project Outline
- Data Preparation
  - Video Data
    - Scrape MLB.com for at-bat videos
    - Strip out audio from video clips
    - denoise the audio
    - Generate mel-spectrograms images
  - Meta Data
    - Scrape MLB.com for pitch metadata
    - one-hot encode batters
    - create the target by binarizing off-speed and fastball pitches to 0-1
- Model Training
  - Split dataset into training and test datasets
  - Fit C-NN and optimize hyperparameters
- Model Inference and Analysis
  - Inspect the prediction distribution to determine if cheating signals are present
  - Use SHAPley values to pinpoint the sound itself
- Future Work
  - Rerun the analysis for past seasons and other teams

# The Data

## Meta Data
The data was scraped from the MLB baseball savant website https://baseballsavant.mlb.com/statcast_search using Beautiful Soup. I pulled all pitches thrown to Astros batters in home games during the 2017 seasons.

**add more charts here with ground truth data**

- 12,642 total pitches
- 19 players
- 45.4% of pitches were off-speed 

Three example rows in the table:

| pitch | pitcher          | batter           | count | date     | dist | exit_velocty | inning | launch_angle | mph  | pitch_id                             | pitch_result | spin_rate | zone |
|-------|------------------|------------------|-------|----------|------|--------------|--------|--------------|------|--------------------------------------|--------------|-----------|------|
| SL    | Kershaw, Clayton | Springer, George | 2-Mar | 10/29/17 | NaN  | NaN          | Bot 5  | NaN          | 88   | 27df5d03-0f01-495b-8528-b8a277c30c2b | ball         | 2613      | 13   |
| FF    | Kershaw, Clayton | Springer, George | 2-Mar | 10/29/17 | 222  | 100.7        | Bot 5  | 14           | 93.2 | 1dd2e806-8a45-4433-a7fa-64b637537161 | foul         | 2345      | 4    |
| SL    | Kershaw, Clayton | Springer, George | 2-Mar | 10/29/17 | 259  | 85.4         | Bot 5  | 40           | 89.1 | ef5299e9-c8af-4c49-bf89-0611c311e6f6 | foul         | 2578      | 9    |

The meta data used for the model was a one-hot encoded vector of the batters(length of 19).

## Video Data Download
The pitch_id from the metadata then used to download the actual video file for that pitch using `wget`. For example, the first pitch in the table can be downloaded using a link like:
https://sporty-clips.mlb.com/27df5d03-0f01-495b-8528-b8a277c30c2b.mp4


## Video to Spectrogram Image  conversion

From this video file, I then extracted the audio using `ffmpeg`, denoised the audio with `noise_reduce` and then converted to a mel-spectrogram using `librosa`.
Below are a few examples of the outputed spectrograms. *NOTE: all examples include bangs. Look for the small triangle shaped blobs at the bottoms of the images.*
![Highest Predictions](baseball-audio-cheating-detection/docs/top_15_preds.png)

These images were then exported as a numpy matrix of size 108x128 using the image's pixel intensity values.


# Model Architecture

The model consists of two combined components:
1. image data goes through through 3 convolutional layers and a dense layer on its output
2. meta data goes through a single 4 node dense layer

Both of these layers are then concatenated and outputed to a second 32 node dense layer before finally outputing to a sigmoid function.

```
class BaseballCNN(nn.Module):
    def __init__(self,filters=32,dropout=0.1,nodes=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(filters, filters, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(p=dropout)
        
        self.conv3 = nn.Conv2d(filters, filters, 3)
        self.pool3 = nn.MaxPool2d(2)
        
        self.linear_image = nn.Linear(filters* 11* 14,nodes-4)
        self.linear_meta = nn.Linear(20,4)
        self.drop3 = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(nodes,nodes) 
        self.drop4 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(nodes,1)
        
    def forward(self, x_image,x_meta):
        
        # 3 convolutional layers on the image with max pooling
        x_image = F.relu(self.conv1(x_image))
        x_image = self.pool1(x_image)
        x_image = self.drop1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = self.drop2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)
        x_image = torch.flatten(x_image,1)

        # a dense layer with 4 nodes for the meta data
        x_image = self.linear_image(x_image)

        # a dense layer on the flattened image layers
        x_meta = self.linear_meta(x_meta)

        # concatenating the two dense components 
        x = torch.cat((x_image,x_meta),1)

        # sigmoid output
        x= torch.sigmoid(self.linear3(self.drop4(self.linear2(x))))
        return x
```


![Model Diagram](baseball-audio-cheating-detection/trained_models/model_diagram.png)

Interestingly, the model doesn't really start to fit to the data until after ~16 epochs and continues to decrease loss on the validation set until epoch ~30. Early stopping was applied to ultimately select the weights after 31 epochs.

![Loss](baseball-audio-cheating-detection/docs/cnn_loss.png)

![Accuracy](baseball-audio-cheating-detection/docs/cnn_acc.png)


## SHAP Interpretation of the Predictions

I then ran SHAP DeepExplainer on the predictions and plotted out where within the image the prediction was derived from.
- the blue line shows the magnitude of the contribution towards the model score was within that point in time in the clip
- the red spots are the exact spots that the shapley values were higher within the image
- the black background is the original image spectrogram
- the grey box shows the area that was selected for the final youtube video output clip(centered around the maximum point on the blue line)
![shap1](baseball-audio-cheating-detection/docs/shap_1.png)
![shap2](baseball-audio-cheating-detection/docs/shap_2.png)
![shap3](baseball-audio-cheating-detection/docs/shap_3.png)
![shap2](baseball-audio-cheating-detection/docs/shap_4.png)



# Future Work

A lot of things had to go right in order for the model to learn in this case:

- the trash can was either near a broadcast microphone or was extremely loud.
- The Astros did not switch up their bang signal game-to-game(bang always meant off-speed).
- They did this for full season of data giving us thousands of data points.
- They used the same trash can which gives us an almost identical noises each time.

Given that the model is agnostic to the sound itself, I would like to apply this technique to other seasons and teams where, if cheating happened elsewhere, the noise of the cheating would not be known a priori.

I would also like to apply more modern ML techniques beyond CNNs with spectrograms as well as leverage video data.
