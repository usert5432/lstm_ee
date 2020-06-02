"""Custom `keras` callbacks"""

import time
from keras.callbacks import Callback

class TrainTime(Callback):
    """Callback that saves cumulative training time for each epoch in log."""

    def __init__(self):
        super(TrainTime, self).__init__()
        self.start_time = None

    def on_train_begin(self, logs = None):
        self.start_time = time.perf_counter()

        if logs is not None:
            logs['train_time'] = 0

    def on_epoch_end(self, epoch, logs = None):
        if logs is not None:
            timestamp = time.perf_counter()
            logs['train_time'] = timestamp - self.start_time

