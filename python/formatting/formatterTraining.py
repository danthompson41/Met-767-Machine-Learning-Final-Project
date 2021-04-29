
import os
import shutil
from sys import argv
import math

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import json
tf.get_logger().setLevel('ERROR')

# Load all of the data in the training folder
training_list = []
for roots, dirs, files in os.walk("../../data/3_formatting/2_training"):
    print(roots)
    print(files)
    for i in files:
        name = "../../data/3_formatting/2_training/"+i
        print(name)
        contents = pd.read_json(name)
        print(contents.keys())
        training_list.append(contents)

x = []
y = []


for bill in training_list:
    print(bill.keys())
    for i in bill.values.tolist():
        print(i)
        x.append(i[0])
        y.append(i[1])

# Count the representation of document labels
countset = {}
print(len(y))
for i in range(0,len(y)):
    print(i)
    if y[i] not in countset:
        countset[y[i]] = []
        countset[y[i]].append(x[i])
    else:
        countset[y[i]].append(x[i])

# Find which label has the largest count
max_count = 0
for i in countset:
    print(len(countset[i]))
    if max_count < len(countset[i]):
        max_count = len(countset[i])

# For sets that are underrepresented, add some oversampling
for i in countset:
    undersampled_factor = math.floor(max_count / len(countset[i]))
    print(undersampled_factor)
    countset[i] = countset[i] * undersampled_factor

# Print the new, oversampled counts
for i in countset:
    print("Oversampled")
    print(len(countset[i]))

x_new = []
y_new = []
for i in countset:
    x_new = x_new + countset[i]
    y_new = y_new + [i] * len(countset[i])

print(x_new)
print(y_new)

x = x_new
y = y_new

# Generate one hot encodings for the output
encoder = OneHotEncoder(sparse=False)
y = np.array(y).reshape(-1,1)
y = encoder.fit_transform(y)
print(y)
encoder.get_feature_names()


# Load relevant, pretrained BERT layers for transfer learning
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

# Build the classifier model, based on bert transfer learning
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(4, activation='softmax')(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# Create the optimizer
num_train_steps = 3000
num_warmup_steps = int(0.1*num_train_steps)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# If you have a weight to start from, load that weight
if len(argv) == 2:
    classifier_model.load_weights(argv[1])
label = "cli"
early_stopping = EarlyStopping(monitor='loss', patience=5)
filepath = "../../data/3_formatting/0_model_checkpoints/" + label + "-weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='accuracy',
                             verbose=1,
                             save_best_only=False,
                             mode='max')


# Split for training and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.80, random_state=42, stratify=y)
dataset = tf.data.Dataset.from_tensor_slices(([X_train],[y_train]))
print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=dataset,
                               epochs=100,
                               callbacks=[early_stopping, checkpoint]
                              )
