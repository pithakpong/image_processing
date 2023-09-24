#Array, image processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Model Operation
from keras import Model, Input
import keras.utils as image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.wrappers.scikit_learn import KerasRegressor
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
# io
import glob
import warnings;
warnings.filterwarnings('ignore')

def create_autoencoder(optimizer='adam', learning_rate=0.001, batch_size=16, epochs=2):
    
    model = tf.keras.Sequential()
    # Layer 1
    model.add(Input(shape=(100,100,3)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    # Layer 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    # Layer 3
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Layer 4
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))#

    model.add(UpSampling2D((2, 2)))

    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    # Compile the model

    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_squared_error'])
    
    return model
    """
    Input_img = Input(shape=(100,100,3))
    x1 = Conv2D(256,(3, 3), activation='relu', padding='same')(Input_img)
    x2 = Conv2D(128,(3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPool2D((2,2))(x2)
    encoded = Conv2D(64,(3, 3), activation='relu', padding='same')(x2)
    x3 = Conv2D(64,(3, 3), activation='relu', padding='same')(encoded)
    x3 = UpSampling2D((2,2))(x3)
    x2 = Conv2D(128,(3, 3), activation='relu', padding='same')(x3)
    x1 = Conv2D(256,(3, 3), activation='relu', padding='same')(x2)
    decoded = Conv2D(3, (3,3), padding='same')(x1)

    autoencoder = Model(Input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    autoencoder.summary()

    return autoencoder
    """
def main():
    print("hi")
    jpg_files = glob.glob('/lustre/ai/dataset/dip/Face_mini/*/*.jpg')
    imgs = []
    for jpg_file in jpg_files:
      print(jpg_file)
      image = load_img(jpg_file, target_size=(100,100), interpolation="nearest")
      img = img_to_array(image)
      img = img/255
      imgs.append(img)

    print("load imgs successfull")

    train_x, test_x = train_test_split(imgs, random_state=42, test_size=0.3)
    train_x, val_x = train_test_split(train_x, random_state=42, test_size=0.2)
    noise_mean = 0
    noise_std = 1
    noise_factor = 0.2

    #train_x = np.array(train_x)
    #test_x = np.array(test_x)
    #val_x = np.array(val_x)

    train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=(100,100,3)) )
    val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=(100,100,3)) )
    test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=(100,100,3)) )

    print("prepare data successfull")

    model = KerasRegressor(build_fn=create_autoencoder, epochs=2, batch_size=16, verbose=0)
    opts = ['Adam', 'RMSProp']  # Add more optimizers as needed
    lnR = [0.001]  # Learning rates to experiment with
    bs = [2, 32]  # Batch sizes to experiment with
    eps = [200, 400]  # Number of epochs to experiment with

    param_grid = dict(batch_size=bs, epochs=eps, optimizer=opts, learning_rate=lnR)

    grid = GridSearchCV(estimator=model, n_jobs=1 ,verbose=10,cv=2, param_grid=param_grid)

    # Run the grid search
    grid_result = grid.fit(train_x_noise, np.array(train_x))

    # Display the best parameters and best score
    best_params = grid_result.best_params_
    best_score = grid_result.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    # Display mean and standard deviation of scores for each set of hyperparameters
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print(f"Mean: {mean}, Std: {std}, Params: {param}")

    
    e = grid_result.best_params_['epochs']
    b = grid_result.best_params_['batch_size']
    o = grid_result.best_params_['optimizer']
    l = grid_result.best_params_['learning_rate']
    autoencoder = create_autoencoder()

    callback = EarlyStopping(monitor='loss', patience=3)
    history = autoencoder.fit(train_x_noise, train_x, epochs=e, batch_size=b, shuffle=True, validation_data=(val_x_noise, val_x), callbacks=[callback])

    test_predictions = autoencoder.predict(test_x_noise)
    plt.imsave('test_prediction_plt.png',test_predictions)
    cv2.imwrite('test_predictation_cv2.png',test_predictions)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_loss_plot.png') 
    plt.show()
    
if __name__ == "__main__":
    main()