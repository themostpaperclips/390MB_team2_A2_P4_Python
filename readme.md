# CS 390MB Fall 2016 Group 2 A2 Part 4

## This is kind of a mess so please read this!

## Usage

### To get labeled data from the server saved to a csv:

```
python collect-labelled-activity-data.py
```

This stores data in a file called **my-activity-data.csv** in the current directory. This is done in a section we can't change.

### To train the classifier:

Right now the classifier gets trained from **data/sample-data.csv** this was done to create **classifier.pickle** which needs to exist for **activity-recognition.py** to run without error. To train on the new data you must go into **activity-classification-train.py** and change line 52:

```
data_file = os.path.join('data', 'sample-data.csv')
```

To this:

```
data_file = 'my-activity-data.csv'
```

I have not done this since I have no data from the server so activity-classification-train would not run if I changed it. Once you've made the change and made sure you have data in **my-activity-data.csv** run:

```
python activity-classification-train.py
```

This will train the classifier and serialize it into a pickle file called **classifier.pickle**. It will overwrite the file since it is there already.

### To do server-side prediction

Make sure you have a classifier in **classifier.pickle** and run:

```
python activity-recognition.py
```

## Files

### data/sample-data.csv

Starter data, used to make the files so things can all run without error.

### activity-classification-train.py

This trains a decision tree classifier on data and serializes the classifier to **classifier.pickle**. This has a bunch of code from part 3 that I commented out because it's not needed for part 4. Right now it gets data from **data/sample-data.csv**. I set it this way so it can run without error. For part 4 it must get data from the data we send to the server.

### activity-recognition.py

This handles server-side activity recognition.

### classifier.pickle

A serialization of the classifier.

### collect-labelled-activity-data.py

Server side data collection. This is needed to make the training data for part 4.

### features.py

This extracts features, no need to worry about this for part 4.

### plot.py

I used this to make plots for an earlier part but this might be the only thing we submit so I threw it in. Don't worry about this.

### util.py

They gave us this to handle windows. Don't worry about this.
