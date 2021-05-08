from dataloader import trainGenerator, valGenerator
from model import build_model
from learning_support import EarlyStoppingAtMinLoss
import tensorflow as tf
import os
import gc

train_dataset = tf.data.Dataset.from_generator(trainGenerator,(tf.float32, tf.int32))
val_dataset = tf.data.Dataset.from_generator(valGenerator, (tf.float32, tf.int32))

train_dataset = train_dataset.batch(1).prefetch(1)
#val_dataset = val_dataset.batch(1).prefetch(1)

tf.debugging.set_log_device_placement(True)
strategy  = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = build_model()

    print(model.summary())

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr = 1e-3),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    )

    es = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)
    mc = tf.keras.callbacks.ModelCheckpoint("sleep_stage.h5", monitor = "val_loss", mode = "min", save_best_only = True)

    for image_batch, label_batch in train_dataset:
        image_batch = tf.reshape(image_batch, [-1,280,480,1])
        label_batch = tf.reshape(label_batch, [-1,1])
        hist = model.fit(image_batch, label_batch, batch_size = 1, epochs = 1, callbacks = [es, mc], verbose = 1)
        print("\n")
        print(image_batch.shape)
        print(label_batch.shape)

        loss, acc = model.evaluate(image_batch, label_batch)

        print("train_loss:", loss, "train_acc:", acc)

