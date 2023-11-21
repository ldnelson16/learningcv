import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def display_examples(examples,labels):
  plt.figure(figsize=(12,12))
  for i in range(36):
    idx = random.randint(0,examples.shape[0]-1)
    img = examples[idx]
    label = labels[idx]
    plt.subplot(6,6,i+1)
    plt.title(str(label))
    plt.tight_layout()
    plt.imshow(img,cmap="gray")
  plt.show()

def final_display(examples,labels,totalpredictions):
  plt.figure(figsize=(12,12))
  for i in range(examples.shape[0]):
    idx = i
    img = examples[idx]
    label = labels[idx]
    plt.subplot(6,6,i+1)
    plt.title(totalpredictions[i])
    plt.tight_layout()
    plt.imshow(img,cmap="gray")
    if (totalpredictions[i][0,0] > 0.5 and i < examples.shape[0]/2) or (totalpredictions[i][0,1] > 0.5 and i>=examples.shape[0]/2):
      subtitle = "Luke" if i < examples.shape[0]/2 else "Parker"
      color='green'
    else:
      subtitle = "Luke" if i >= examples.shape[0]/2 else "Parker"
      color='red'
    plt.xlabel(subtitle, size=16, color=color)
  plt.show()

class LukevsParkerNN(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.conv0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
    self.maxpool0 = tf.keras.layers.MaxPooling2D((2, 2))
    self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
    self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
    self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
    self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))
    self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
    self.maxpool3 = tf.keras.layers.MaxPooling2D((2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(128, activation='relu')
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.dense2 = tf.keras.layers.Dense(2, activation='softmax')

  def call(self,input):
    x = self.conv0(input)
    x = self.maxpool0(x)
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)
    x = self.conv3(x)
    x = self.maxpool3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dropout(x)
    x = self.dense2(x)
    
    return x



numphotos = 100
numtestphotos = 13
image_shape = (250,500)
image_shape_channels = (250,500,1)
x_train = np.empty((numphotos * 2, *image_shape), dtype=np.float32)
y_train = np.empty((numphotos * 2,), dtype=np.int32)
# read image data into an array
j = 0
for name in ["luke","parker"]:
  for i in range(numphotos):
    img = cv2.imread("images/"+name+str(i)+".jpg")
    img = cv2.resize(img, (500, 250))
    if img is None:
      print("Couldn't find correct image with filename","/images/"+name+str(i)+".jpg")
      break
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32) / 255.0
    x_train[j] = img
    y_train[j] = 0 if name=="luke" else 1
    j+=1

x_test = np.empty((numtestphotos * 2, *image_shape), dtype=np.float32)
y_test = np.empty((numtestphotos * 2,), dtype=np.int32)
# read image data into an array
j = 0
for name in ["luke","parker"]:
  for i in range(numtestphotos):
    img = cv2.imread("images/test"+name+str(i)+".jpg")
    if img is None:
      print("Couldn't find correct image with filename","/images/test"+name+str(i)+".jpg")
      break
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32) / 255.0
    # img = img.flatten()
    x_test[j] = img
    y_test[j] = 0 if name=="luke" else 1
    j+=1

# x_test = x_test[:j]

num_classes = 2

print(x_train.shape, y_train.shape)
print(x_test.shape,y_test.shape)

display_examples(x_train,y_train)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test,axis=-1)

y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_test = tf.keras.utils.to_categorical(y_test,num_classes)



model = LukevsParkerNN()
# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model's architecture
# model.summary()

model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

total_predictions = []

for i in range(x_train.shape[0]):
  predictions = model.predict(x_train[i].reshape((1, 250, 500, 1)))
  print("Predicting for","Luke" if i < x_train.shape[0]/2 else "Parker")
  print("Correct!" if ((predictions[0,0]>0.5 and i < x_train.shape[0]/2) or (predictions[0,1]>0.5 and i >= x_train.shape[0]/2)) else "Wrong!")
  print(predictions)

for i in range(x_test.shape[0]):
  predictions = model.predict(x_test[i].reshape((1, 250, 500, 1)))
  print("Predicting for","Luke" if i < x_test.shape[0]/2 else "Parker")
  print("Correct!" if ((predictions[0,0]>0.5 and i < x_test.shape[0]/2) or (predictions[0,1]>0.5 and i >= x_test.shape[0]/2)) else "Wrong!")
  print(predictions)
  total_predictions+=[predictions]

final_display(x_test,y_test,total_predictions)