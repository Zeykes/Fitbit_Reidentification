#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from SiameseModel import Recognizer

RAW_DATA_DIR = r'C:\Users\Marti\Documents\Master Thesis\PMData'
PARTICIPANTS = os.listdir(RAW_DATA_DIR)[:-1] # Not including the .xlsx file
MAX_WEEKS = 14
CHUNK_SIZE = {
    'heart_rate': 17280, #86400, #day=86400, week=604800
    'step': 10080
}

# Convert to taking inputs for these parameters.
DATA_TYPE = "heart_rate" # step, heart_rate

NUM_PARTICIPANTS = 16
assert (1 <= NUM_PARTICIPANTS <= 16), "Number of participants must be between 1-16"

NUM_WEEKS = 28 # NOTE: this is actually days now, not weeks
assert_msg = f"Number of weeks must be between 1-{MAX_WEEKS}"
#assert (1 <= NUM_WEEKS <= MAX_WEEKS), assert_msg

START_WEEK = 0 # input - 1
assert_msg = f"Must be <= {MAX_WEEKS - NUM_WEEKS}"
#assert (START_WEEK <= (MAX_WEEKS - NUM_WEEKS)), assert_msg

TEST_NUM_WEEKS = 4
assert_msg = f"Must be <= {MAX_WEEKS - (START_WEEK + NUM_WEEKS)}"
#assert (TEST_NUM_WEEKS <= (MAX_WEEKS - (START_WEEK + NUM_WEEKS))), assert_msg

TEST_START_WEEK = START_WEEK + NUM_WEEKS


def read_file(maindir, p, file):
    path = f"{maindir}\\{p}\\fitbit\\{file}"
    if ".json" in file:
        return pd.read_json(path)
    elif ".csv" in file:
        return pd.read_csv(path) # TODO: verify that pandas reads the PMData csv files properly
    else:
        raise TypeError("Unsupported file type!")


def clean_data_hr(df, start_date="2019-11-18 00:00:00", end_date="2020-02-23 23:59:59"):
    # Preserve datetime as index
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    index = df['dateTime']
    
    # Convert dict-values to columns and set datetimes as index
    df = pd.DataFrame.from_dict(df.value.to_dict(), orient='index')
    df = df.set_index(index)

    # Check if start_date and end_date rows exist (create if not)
    if not start_date in df.index:
        stamp = pd.Timestamp(start_date)
        df.loc[stamp] = None
    if not end_date in df.index:
        stamp = pd.Timestamp(end_date)
        df.loc[stamp] = None
    
    # Sort chronologically using datetime-index
    df = df.sort_index()

    # Shave off incomplete weeks at start/end
    df = df.loc[start_date:end_date] # Adjust dates as necessary

    # Fill in missing rows to make data set consistent (secondly)
    df = df.resample('1s').first()

    # Interpolate, round off and cast to int
    df['bpm'] = df.bpm.interpolate(method='linear', limit_direction='both').round(0).astype(np.int64)
    
    # Resample back to 5-second frequency
    df = df.resample('5s').first()
    
    # Replace datetime index with normal index and remove confidence
    df = df.reset_index(drop=True)
    del df['confidence']
    
    return df


# This is the converted for-loop from DataProcessor.py
def prepare_chunks(datasets, start_week=0, weeks=5, no_ID=False):
    chunks = []
    start_ind = start_week * CHUNK_SIZE[DATA_TYPE] # Finds index corresponding to monday 00:00:00 of start_week
    for p in datasets:
        for i in range(weeks): # weeks mon-sun
            from_ind = start_ind + (CHUNK_SIZE[DATA_TYPE] * i)
            to_ind = from_ind + (CHUNK_SIZE[DATA_TYPE] - 1)
            chunk = p[1].loc[from_ind:to_ind]
            chunk = np.array(chunk)
            if no_ID:
                chunks.append(chunk)
            else:
                chunks.append((p[0], chunk))
    return chunks


def remove_ID(dataset):
    # Remove ID
    dataset[i] = dataset[i][1]



# DATA PROCESSING

datasets = []
for p in PARTICIPANTS[:NUM_PARTICIPANTS]:
    pdata = read_file(RAW_DATA_DIR, p, "heart_rate.json")
    d = clean_data_hr(pdata)
    tup = (p, d)
    datasets.append(tup)

# List with participant name and corresponding week-chunks
# [('p01', chunks_array), ('p02', chunks_array), ...]
chunks = prepare_chunks(datasets, start_week=START_WEEK, weeks=NUM_WEEKS)

# BALANCED VERSION
input_1 = []
input_2 = []
labels = []

# Match all chunks (balanced)
for i in range(len(chunks)):
    same = []
    not_same = []
    for j in range(len(chunks)):
        if chunks[j][0] == chunks[i][0]:
            same.append(chunks[j])
        else:
            not_same.append(chunks[j])
    random.shuffle(same)
    random.shuffle(not_same)
    not_same = not_same[:len(same)] # match lengths for balance
    to_add = same + not_same
    random.shuffle(to_add)
    input_2.extend(to_add)
    for x in range(len(to_add)):
        input_1.append(chunks[i])

# GENERATE LABELS AND REMOVE ID
for i in range(len(input_1)):
    # Check if they have the same ID
    p1 = input_1[i][0]
    p2 = input_2[i][0]
    if p1 == p2:
        labels.append(1)
    else:
        labels.append(0)
    remove_ID(input_1)
    remove_ID(input_2)
    
X1 = np.array(input_1)
X2 = np.array(input_2)
Y = np.array(labels)



# MODEL TRAINING

recognizer = Recognizer()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

parameters = {
    'batch_size' : 16 ,
    'epochs' : 5000 , # 5000 if callback is implemented
    'callbacks' : early_stop ,
    'val_data' : None
}

recognizer.fit([X1, X2], Y, hyperparameters=parameters)

shuffled_chunks = chunks.copy()
random.shuffle(shuffled_chunks)
test_chunks = prepare_chunks(datasets, start_week=TEST_START_WEEK, weeks=TEST_NUM_WEEKS)



# TESTING

# VERY SLOW. Checks every test_chunk against every data_chunk.
scores = []
labels = []
truths = []
for i in range(len(test_chunks)):
    score = []
    label = []
    truths.append(test_chunks[i][0]) # Real label from test_chunks
    for data_chunk in shuffled_chunks:
        label.append(data_chunk[0])
        test_chunk = test_chunks[i][1].reshape((1,-1))
        data_chunk = data_chunk[1].reshape((1,-1))
        try:
            score.append(recognizer.predict([test_chunk, data_chunk])[0])
        except:
            print(i)
            print(test_chunk)
            print(len(test_chunks[i][1]))
    scores.append(score)
    labels.append(label)

scores = np.array( scores )
labels = np.array( labels )
truths = np.array( truths )

# Checks every test_chunk against a random data_chunk.
scores = []
labels = []
correct = []
for i in range(len(test_chunks)):
    input_1 = test_chunks[i][1].reshape((1,-1))
    input_2 = shuffled_chunks[i][1].reshape((1,-1))
    label = shuffled_chunks[i][0]
    truth = test_chunks[i][0]
    scores.append(recognizer.predict([input_1, input_2])[0])
    labels.append(label)
    if label == truth:
        correct.append(1)
    else:
        correct.append(0)

scores = np.array( scores )
labels = np.array( labels )
correct = np.array( correct )
