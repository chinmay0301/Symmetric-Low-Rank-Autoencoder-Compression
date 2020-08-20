## Note this is a dump taken from Google Colab notebook ! 

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()

from tensorflow import keras
import numpy as np
import pathlib
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model


# get the MNIST data set
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# plot and see the digits
plt.figure(figsize=(20, 2))
for i in range(1,11):
    ax = plt.subplot(1, 10, i)
    plt.imshow(x_test[i].reshape(28, 28), cmap="binary")
    plt.gray()
plt.show()


# build an autoencoder model , compile and fit
autoencoder = keras.Sequential([
keras.layers.InputLayer(input_shape=(784,)),
keras.layers.Dense(128,activation='relu'),
keras.layers.Dense(64,activation='relu'),
keras.layers.Dense(32,activation='relu'),
keras.layers.Dense(64,activation='relu'),
keras.layers.Dense(128,activation='relu'),
keras.layers.Dense(784,activation='sigmoid')
])

autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)


# look at the reconstructed images
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# evaluate model on test set
autoencoder.evaluate(x_test,x_test)

# get a histogram
plt.hist((autoencoder.layers[5].get_weights()[1]),bins=100)
plt.show()

# use TFlite to quantize
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("/content/gdrive/My Drive/MNIST_models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_recons_fp32_model.tflite"
tflite_model_file.write_bytes(tflite_model)

# FP16
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"mnist_model_recons_f16.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
interpreter_fp16.allocate_tensors()

# function to evaluate 
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  loss = 0
  for image in x_test:
    image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    for i in range(783):
      loss+= (output()[0][i]- image[0][i])* (output()[0][i]- image[0][i]) 
    #print(output()[0].shape)
    #print(image[0].shape)
        
  return (loss/10000)

# loss for orig model 
evaluate_model(interpreter)

# loss for FP16 model 
evaluate_model(interpreter_fp16)

# INT8 quantization of weights and activations 
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
  for i in range(1000):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [np.array(x_train[i],dtype=np.float32,ndmin=2)]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant8_model = converter.convert()

tflite_model_int8_file = tflite_models_dir/"mnist_model_recons_int8.tflite"
tflite_model_int8_file.write_bytes(tflite_quant8_model)

interpreter_int8 = tf.lite.Interpreter(model_path=str(tflite_model_int8_file))
interpreter_int8.allocate_tensors()

# get loss for the int8 model
evaluate_model(interpreter_int8)

# plot a test image reconstructed from a quantized model
test_image = np.expand_dims(x_test[0], axis=0).astype(np.float32)
input_index = interpreter_int8.get_input_details()[0]["index"]
output_index = interpreter_int8.get_output_details()[0]["index"]
interpreter_int8.set_tensor(input_index, test_image)
interpreter_int8.invoke()
out_img = interpreter_int8.get_tensor(output_index)

plt.figure(figsize=(20, 4))
plt.imshow(out_img.reshape(28, 28))
plt.gray()
plt.show()