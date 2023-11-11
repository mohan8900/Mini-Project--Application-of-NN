# MINI-PROJECT--APPLICATION-OF-NN..

### PROJECT TITLE :

## CONVOLUTIONAL AUTOENCODER FOR IMAGE DENOISING...

### PROJECT DESCRIPTION :

### AIM :

To develop a convolutional autoencoder for image denoising application.

### PROBLEM STATEMENT :

*  Image Denoising is the process of removing noise from the Images. The noise present in the images may be caused by various intrinsic or extrinsic conditions which are practically hard to deal with. The problem of Image Denoising is a very fundamental challenge in the domain of Image processing and Computer vision. Therefore, it plays an important role in a wide variety of domains where getting the original image is really important for robust performance.

*  Modeling image data requires a special approach in the neural network world. The best-known neural network for modeling image data is the Convolutional Neural Network (CNN).

*  It can better retain the connected information between the pixels of an image. The particular design of the layers in a CNN makes it a better choice to process image data.

*  The CNN design can be used for image recognition/classification, or be used for image noise reduction or coloring. We can train the CNN model by taking many image samples as the inputs and labels as the outputs. We then use this trained CNN model to a new image to recognize if it is a “dog”, or “cat”, etc. CNN also can be used as an autoencoder for image noise reduction or coloring.

*  This program demonstrates how to implement a deep convolutional autoencoder for image denoising, mapping noisy digits images from the MNIST dataset to clean digits images.

* The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The training dataset in Keras has 60,000 records and the test dataset has 10,000 records. Each record has 28 x 28 pixels.

### DATASET :

The link for the dataset for developing the convolutional autoencoder for image denoising application is given.

The MNIST dataset is downloaded from kaggle and splitted into training set and testing set.

https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/versions/2?resource=download

#### TRAINING SET (mnist_train) :

![train](https://user-images.githubusercontent.com/93427534/205541438-95340217-ffd4-4199-810e-32c3ae45282a.png)

#### TESTING SET (mnist_test) :

![test](https://user-images.githubusercontent.com/93427534/205541451-294af908-d2c8-4662-a697-1a3fb901ba9a.png)

#### CONVOLUTIONAL AUTOENCODER NEURAL NETWORK MODEL EXAMPLE :

![out1](https://user-images.githubusercontent.com/93427534/205541771-aa27e4a5-b2b5-4845-a4fe-5a502767f8ef.png)

#### CONVOLUTIONAL AUTOENCODER NEURAL NETWORK MODEL :

![out2](https://user-images.githubusercontent.com/93427534/205541779-ffc79fc7-78da-465b-8af0-75f9c4624ade.png)

### ALGORITHM :

#### STEP 1:

Import the necessary libraries and download the mnist dataset.

#### STEP 2:

Load the dataset and scale the values for easier computation.

#### STEP 3:

Add noise to the images randomly for the process of denoising it with the convolutional denoising autoencoders for both the training and testing sets.

#### STEP 4:

Build the Neural Model for convolutional denoising autoencoders using Convolutional, Pooling and Up Sampling layers. Make sure the input shape and output shape of the model are identical.

#### STEP 5:

Pass test data for validating manually. Compile and fit the created model.

#### STEP 6:

Plot the Original, Noisy and Reconstructed Image predictions for visualization.

#### STEP 7:

End the program.

### PROGRAM :

### Program to develop a convolutional autoencoder for image denoising application.

#### Importing the necessary libraries :
```python

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

```

#### Loading the MNIST dataset :
```
(x_train, _), (x_test, _) = mnist.load_data()
```

```
x_train.shape
```

#### Scaling and reshaping the values as x_train_scaled, x_test_scaled :
```
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
```

#### Adding noise to check how the autoencoders denoise it :
```
noise_factor = 0.8
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```

#### Taking 10 values and displaying them with the added noise :
```
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

#### Creating the model :
```
input_img = keras.Input(shape=(28, 28, 1))
```

#### Encoding layer of the model :
```
x = layers.Conv2D(32, (3, 3),activation = 'relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
```

#### Decoding layer of the model :
```
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
```

####  Autoencoder.summary() :
```
autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()
```

#### Compiling the optimizer and loss for the model :
```
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

#### Fitting of the model by adding epochs and batch_size for the batch-wise training of the model :
```
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```

```
decoded_imgs = autoencoder.predict(x_test_noisy)
```

#### Displaying the normal values :
```
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original:
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
```

#### Displaying the values after the adding of noise :
```
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
```

#### Displaying the values after the reconstruction of the images :
```
    # Display reconstruction:
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
```

```
plt.show()

```

### OUTPUT :

#### ADDING NOISE TO THE MNIST DATASET :

![out3](https://user-images.githubusercontent.com/93427534/205545261-e9b8423e-1d58-440e-a244-2b1eb499c51e.png)

#### AUTOENCODER.SUMMARY() :

![out4](https://user-images.githubusercontent.com/93427534/205545298-8d3f8bf9-cebb-4e83-a8e5-470ac1def9f9.png)

#### ORIGINAL V/S NOISY V/S RECONSTRUCTED IMAGE :

![out5](https://user-images.githubusercontent.com/93427534/205545326-ecee38a9-7b14-4d5d-839f-eb1fd0be2b35.png)

### GOOGLE COLAB PYTHON FILE :

The colab file is also uploaded (check the uploaded files) as well as the link for accessing the colab file is given.

https://colab.research.google.com/drive/1sR1MqHsbXLFnASkMGSCy0NYTREgqpbs0

### ADVANTAGES :

*  Modeling image data requires a special approach in the neural network world. The best-known neural network for modeling image data is the Convolutional Neural Network (CNN, or ConvNet). It can better retain the connected information between the pixels of an image. The particular design of the layers in a CNN makes it a better choice to process image data.

*  In the case of image data, the autoencoder will first encode the image into a lower-dimensional representation, then decodes that representation back to the image.

*  Autoencoders provide a useful way to greatly reduce the noise of input data, making the creation of deep learning models much more efficient.

*  Autoencoders provide a useful way to greatly reduce the noise of input data, making the creation of deep learning models much more efficient. They can be used to detect anomalies, tackle unsupervised learning problems, and eliminate complexity within datasets.

*  A denoising autoencoder, in addition to learning to compress data (like an autoencoder), it learns to remove noise in images, which allows to perform well even when the inputs are noisy. So denoising autoencoders are more robust than autoencoders + they learn more features from the data than a standard autoencoder.

### RESULT:

Thus, the program to develop a convolutional autoencoder for image denoising application is developed and executted successfully.
