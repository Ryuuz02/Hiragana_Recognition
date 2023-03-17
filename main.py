import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

dataset = tfds.load("kmnist", split=['train', 'test'], shuffle_files=True)

train_ds = dataset[0]
train_images = np.ndarray(shape=(len(train_ds), 28, 28, 1))
train_labels = np.ndarray(shape=(len(train_ds)))
inc = 0
for example in train_ds:
    train_images[inc] = example['image']
    train_labels[inc] = example['label']
    inc += 1
inc = 0

test_ds = dataset[1]
test_images = np.ndarray(shape=(len(test_ds), 28, 28, 1))
test_labels = np.ndarray(shape=(len(test_ds)))
for example in test_ds:
    test_images[inc] = example['image']
    test_labels[inc] = example['label']
    inc += 1

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)