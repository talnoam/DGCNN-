import os
import pandas as pd
import numpy as np
import awkward
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt
import math

from tensorflow import keras
from tf_keras_model import get_particle_net
from sklearn.model_selection import train_test_split
from convert_dataset import load_data

#load and convert data
data_dir = './data/'
dark_path = data_dir + "DarkTrack.csv"
qcd_path = data_dir + "QCDTrack.csv"
load_data(dark_path, qcd_path, data_dir)

#train
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

train_path = './data/trackTrain_0.awkd'
val_path = './data/trackValidation_0.awkd'

def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)

def pad_array(a, maxlen, value=0, dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x

class Dataset(object):

    def __init__(self, filepath, feature_dict = {}, label='label', pad_len=50, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_etarel', 'part_phirel', 'part_deltaR', 'part_logptrel']
            feature_dict['mask'] = ['part_pt_log']
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            self._label = a[self.label]
            for k in self.feature_dict:
                cols = self.feature_dict[k]
                if not isinstance(cols, (list, tuple)):
                    cols = [cols]
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(pad_array(a[col], self.pad_len))
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

train_dataset = Dataset(train_path, data_format='channel_last')
val_dataset = Dataset(val_path, data_format='channel_last')


model_type = 'particle_net' # choose between 'particle_net' and 'particle_net_lite'
num_classes = train_dataset.y.shape[1]
input_shapes = {k:train_dataset[k].shape[1:] for k in train_dataset.X}
if 'lite' in model_type:
    model = get_particle_net_lite(num_classes, input_shapes)
else:
    model = get_particle_net(num_classes, input_shapes)

# Training parameters
batch_size = 1024 if 'lite' in model_type else 384
epochs = 20

def lr_schedule(epoch):
  lr = 3e-4
  if 0 < epoch <= 8:
    lr = lr + epoch*3.375e-4

  elif 8 < epoch <= 16:
    lr = 3e-3 - (epoch-8)*3.375e-4
  
  elif epoch > 16:
      lr = 5e-7
  logging.info('Learning rate: %f'%lr)
  return lr

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()

# Prepare model model saving directory.
save_dir = 'model_checkpoints'
model_name = '%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger(count_mode='steps')
callbacks = [checkpoint, lr_scheduler, progress_bar]

train_dataset.shuffle()
history = model.fit(train_dataset.X, train_dataset.y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(val_dataset.X, val_dataset.y),
          shuffle=True,
          callbacks=callbacks)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = list(range(epochs))


plt.plot(epoch, train_loss, label='train loss')
plt.plot(epoch, val_loss, label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss Vs Epoch')
plt.legend()
perform_dir = 'performance'
if not os.path.isdir(perform_dir):
    os.makedirs(perform_dir)
plt.savefig(perform_dir + '/Loss Vs Epoch')
print('Train completed')
