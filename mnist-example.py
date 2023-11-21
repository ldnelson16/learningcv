import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# tensorflow.keras.Sequential strategy
sequentialmodel = tf.keras.Sequential(
  [
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.BatchNormalization(), 

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(10,activation='softmax')
  ]
)

# functional approach
def functional_model():
  input = tf.keras.layers.Input(shape=(28,28,1))
  x = tf.keras.layers.Conv2D(32, (3,3), activation="relu")(input)
  x = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.BatchNormalization()(x) 

  x = tf.keras.layers.Conv2D(128, (3,3), activation="relu")(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.GlobalAvgPool2D()(x)
  x = tf.keras.layers.Dense(64,activation="relu")(x)
  x = tf.keras.layers.Dense(10,activation='softmax')(x)

  model = tf.keras.Model(inputs = input, outputs = x)

  return model

# tensorflow.keras.Model class
class MyCustomModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation="relu")
    self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")
    self.maxpool1 = tf.keras.layers.MaxPool2D()
    self.batchnorm1 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(128, (3,3), activation="relu")
    self.maxpool2 = tf.keras.layers.MaxPool2D()
    self.batchnorm2 = tf.keras.layers.BatchNormalization()

    self.globalpool1 = tf.keras.layers.GlobalAvgPool2D()
    self.dense1 = tf.keras.layers.Dense(64,activation="relu")
    self.dense2 = tf.keras.layers.Dense(10,activation='softmax')


  def call(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    x = self.maxpool1(x)
    x = self.batchnorm1(x) 

    x = self.conv3(x)
    x = self.maxpool2(x)
    x = self.batchnorm2(x)

    x = self.globalpool1(x)
    x = self.dense1(x)
    x = self.dense2(x)

    return x


def display_some_examples(examples,labels):
  plt.figure(figsize=(10,10))
  for i in range(25):
    idx = np.random.randint(0,examples.shape[0]-1)
    img = examples[idx]
    label = labels[idx]
    plt.subplot(5,5,i+1)
    plt.title(str(label))
    plt.tight_layout()
    plt.imshow(img,cmap="gray")
  plt.show()

if __name__=="__main__":
  (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

  if False: display_some_examples(x_train,y_train)

  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255

  x_train = np.expand_dims(x_train, axis=-1)
  x_test = np.expand_dims(x_test,axis=-1)

  print(x_train.shape,y_train.shape)
  print(x_test.shape,y_test.shape)

  y_train = tf.keras.utils.to_categorical(y_train,10)
  y_test = tf.keras.utils.to_categorical(y_test,10)

  # model = functional_model()

  model = MyCustomModel()
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
  model.fit(x_train,y_train,batch_size=32,epochs=5, validation_split=0.2) # train model
  model.evaluate(x_test,y_test,batch_size=32) # test model on unseen data