## Thanks

This is a fork of https://github.com/jitinkrishnan/Quick-Deployables.

# Quick-Deployables
Presenting quick deployable models (that are trained using tweets collected from 20 different crisis events labeled for priority/urgency) to filter critical messages during a crisis response.

## Priority Classifier
A bert-based model that predicts binary labels ```0``` or ```1``` (```1``` = high priority/urgent; ```0``` = rest) for a given set of input sentences. 

```Input:``` A text file

```Output:``` Predictions as a list of ```0```s and ```1```s

To be specific, we use ```DistilBert``` for English and ```bert-base-multilingual-uncased``` for multilingual.

### Requirements
- Python3.6+ and ```pip install -r requirements.txt``` to install necessary packages.
- Download the pytorch model from our Google Drive [urgency_en.pt](https://drive.google.com/file/d/1a2xFP8RVF0QE4qk7sW5rOww5EWM9FkL-/view?usp=sharing) to the current folder.
- Input sentences or tweets should be in text format as shown in ```sample.txt```

### How to run
```python predictor.py 'sample.txt' 768 'en' 'urgency_en.pt'```

Sample Output: ```[0,1]```

corresponding to the following two sentences in ```sample.txt```:
```
Hello there.
We need rescue at the train station.
```

#### Calling from another code
If you need to call from another code to return predictions as a numpy array
```
from predictor import predict
x = predict('sample.txt' 768 'en' 'urgency_en.pt')
```

#### Results on Crisis Data
We train using tweets from a set of crisis events and test using a **fully unseen** crisis. For example, when the target crisis is ```Maria```, we train using tweets from rest of all crises and test on tweets from ```Maria```. Our dataset is collected from various sources such as [CitizenHelper](https://ist.gmu.edu/~hpurohit/informatics-lab/icwsm17-citizenhelper.html) and [TREC](http://dcs.gla.ac.uk/~richardm/TREC_IS/).

| Target Crisis  | Accuracy  | Recall |
 :-: |  :-: |  :-:
| Maria                   | 0.89 | 0.88 |
| Harvey                  | 0.82 | 0.83 |
| Florence                | 0.97 | 0.97 |
| Irma                    | 0.83 | 0.82 |
| Australia Bushfire      | 0.89 | 0.87 |
| Philipinnes Floods      | 0.75 | 0.74 |
| Alberta Floods          | 0.86 | 0.81 |
| Nepal Earthquake        | 0.76 | 0.81 |
| Typhoon Hagupit         | 0.81 | 0.83 |
| Chile Earthquake        | 0.86 | 0.81 |
| Joplin Tornado          | 0.77 | 0.77 |
| Typhoon Yolanda         | 0.86 | 0.88 |
| Queensland Floods       | 0.78 | 0.79 |
| Manila Floods           | 0.80 | 0.76 |
| Paris Attacks           | 0.88 | 0.86 |
| Italy Earthquakes       | 0.77 | 0.81 |
| Guatemala Earthquake    | 0.76 | 0.74 |
| Boston Bombings         | 0.82 | 0.86 |
| Florida School Shooting | 0.83 | 0.78 |
| Covid                   | 0.81 | 0.82 |
| **Average**		          | **0.83** | **0.82** |

#### Multilingual
Use [urgency_ml.pt](https://drive.google.com/file/d/1Ljd5mnU2jVeHtatrC2V3MLdLRQrqt242/view?usp=sharing) instead.
```python predictor.py 'sample.txt' 105879 'ml' 'urgency_ml.pt'```

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
