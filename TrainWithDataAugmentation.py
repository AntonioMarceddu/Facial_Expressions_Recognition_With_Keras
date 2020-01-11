from CommonVariables import *
import matplotlib.pyplot as plt
from decimal import *
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D, multiply, Reshape
from keras.losses import categorical_crossentropy
from keras.models import Model, load_model, model_from_json, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import KFold

# Seed for KFold, dropout and other functions.
seed = 56

# Variables.
min_validation_losses = []
min_validation_losses_epochs = []
max_validation_accuracies = []
max_validation_accuracies_epochs = []
final_validation_losses = []
final_validation_accuracies = []
no_of_epochs = []
test_accuracies = []

train_images = np.load(output_folder + '/Data/Train_images.npy')
train_ground_truth = np.load(output_folder + '/Data/Train_ground_truth.npy')
test_images = np.load(output_folder + '/Data/Test_images.npy')
test_ground_truth = np.load(output_folder + '/Data/Test_ground_truth.npy')

# Uncomment for a preview of the images.
#for img in range(10):
#    plt.figure(img)
#    plt.imshow(train_images[img].reshape((image_size, image_size)), interpolation='none', cmap='gray')
#plt.show()


# Function for creating directories.
def create_directory(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")


# Creating directories.
create_directory(output_folder + '/Data')
create_directory(output_folder + '/Models')

for i in range(0,folds):
    create_directory(output_folder + '/Models/Model'+str(i))


# Function for defining the model.
def get_model():
    # Model taken from: S. Miao, H. Xu, Z. Han, and Y. Zhu, "Recognizing facial expressions using a shallow convolutional neural network", IEEE Access, 7:78000-78011, 2019.
    model = Sequential()
    model.add(Conv2D(44, kernel_size=(5, 5), padding='same', data_format='channels_last', input_shape=(image_size, image_size, 1)))
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(Conv2D(44, kernel_size=(3, 3), padding='same', data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(Conv2D(88, kernel_size=(5, 5), padding='same', data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.02))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.4, seed=seed))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.4, seed=seed))
    model.add(Dense(class_number, activation="softmax"))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-6), metrics=['accuracy'])
    model.summary()
    return model


# Function for plotting the loss and accuracy for the training and validation set.
# Taken from https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
def plot_and_save_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists.
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss.
    plt.figure()
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_folder + '/Models/Model' + str(count)+'/Loss.png')
    #Clear axis and figure.
    plt.cla()
    plt.clf()

    ## Accuracy.
    plt.figure()
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig(output_folder + '/Models/Model' + str(count) + '/Accuracy.png')
    #Clear axis and figure.
    plt.cla()
    plt.clf()


# Generator for images - data augmentation.
TrainAndValidationDatagen = ImageDataGenerator(
    samplewise_center = True, # subtract the mean from the image.
    samplewise_std_normalization = True, # divides by the standard deviation.
    #rescale = 1.0 / 255.0,  # a value by which we will multiply the data after any other processing.
    brightness_range = [0.5, 1.0], # a range withing each augmented image will be darkened or brightened.
    rotation_range = 2.5, # a range within which to randomly rotate pictures.
    width_shift_range = 0.025, # range within which to randomly translate pictures horizontally.
    height_shift_range = 0.025, # range within which to randomly translate pictures vertically.
    shear_range = 0.025, # for randomly applying shearing transformations.
    zoom_range = 0.025, # for randomly zooming inside pictures
    #channel_shift_range = 5, # for randomly shifting the channel values.
    #vertical_flip = False, # for randomly flipping half of the images vertically.
    horizontal_flip = True, # for randomly flipping half of the images horizontally.
    fill_mode = 'nearest') # the strategy used for filling in newly created pixels, which can appear after applying the transformation.

testDatagen = ImageDataGenerator(
    samplewise_center = True, # subtract the mean from the image.
    samplewise_std_normalization = True, # divides by the standard deviation.
    #rescale = 1.0 / 255.0,  # a value by which we will multiply the data after any other processing.
    )

# Initialize the function for performing k-Fold on data.
kf = KFold(n_splits = folds, shuffle = True, random_state = seed)

for train_index, validation_index in kf.split(train_images):

    print('\nFOLD '+str(count))

    # Generate batches from indices.
    X_train, X_validation = train_images[train_index], train_images[validation_index]
    y_train, y_validation = train_ground_truth[train_index], train_ground_truth[validation_index]

    # Get the model and compile it.
    model = None
    model = get_model()

    # Train variables.
    # We stop the training of the model if the value of the loss function of the validation set does not improve after a certain number of epochs (patience).
    early_stopper = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 20, verbose = 1, mode = 'auto')

    # We'll save our model during training as long as it gets a better result than the previous epoch. Thus, we will have the best possible model at the end of the training.
    checkpointer = ModelCheckpoint(output_folder + '/Models/Model' + str(count) + '/Checkpoint_Model.h5', monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)

    # We'll help the loss function to get rid of the “plateaus” by reducing the learning rate parameter of the optimization function of a certain value (factor) if  the value of the loss function of the validation set does not improve after a certain number of epochs (patience).
    lr_reducer = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.99, patience = 1, mode = 'auto', verbose = 1)

    # TensorBoard allows to visualize dynamic graphs of the training and test metrics, as well as activation histograms for the different layers in the model.
    tensorboard = TensorBoard(log_dir = './Logs')

    # Generate the batches for train and validation.
    train_iterator = TrainAndValidationDatagen.flow(X_train, y_train, batch_size = batch_size)
    validation_iterator = TrainAndValidationDatagen.flow(X_validation, y_validation, batch_size = batch_size)

    history = None
    # Fits the model on batches with real-time data augmentation:
    history = model.fit_generator(train_iterator,
                                  epochs = epochs,
                                  steps_per_epoch = len(train_iterator),
                                  validation_data = validation_iterator,
                                  validation_steps = len(validation_iterator),
                                  verbose = 2,
                                  shuffle = True,
                                  callbacks = [lr_reducer, tensorboard, early_stopper, checkpointer])

    # Plot and save the history.
    plot_and_save_history(history)

    # Save the min losses, the related epochs, the max accuracies, the related epochs, the final losses, the final accuracies, and the total number of epochs of the networks.
    min_validation_losses.append(str(Decimal(min(history.history['val_loss'])).quantize(Decimal('1.00000'))))
    min_validation_losses_epochs.append(str(history.history['val_loss'].index(min(history.history['val_loss'])) + 1))

    max_validation_accuracies.append(str(Decimal(max(history.history['val_acc'])).quantize(Decimal('1.00000'))))
    max_validation_accuracies_epochs.append(str(history.history['val_acc'].index(max(history.history['val_acc'])) + 1))

    final_validation_losses.append(str(Decimal(history.history['val_loss'][len(history.history['val_loss']) - 1]).quantize(Decimal('1.00000'))))
    final_validation_accuracies.append(str(Decimal(history.history['val_acc'][len(history.history['val_acc'])-1]).quantize(Decimal('1.00000'))))

    no_of_epochs.append(str(len(history.history['loss'])))

    # Save the  model to be used later.
    fer_json = model.to_json()
    with open(output_folder + '/Models/Model' + str(count) + '/Model.json', 'w') as json_file:
        json_file.write(fer_json)
    model.save_weights(output_folder + '/Models/Model' + str(count) + '/Model_weights.h5')
    print("Model saved.")

    # Evaluate network with the test data.
    test_iterator = testDatagen.flow(test_images, test_ground_truth, batch_size = batch_size)
    score = model.evaluate_generator(test_iterator, steps = len(test_iterator), verbose=0)
    test_accuracies.append(str(Decimal(score[1]).quantize(Decimal('1.00000'))))

    # Next fold.
    count+=1


# Write losses and accuracies on file and on screen.
file = open(output_folder + '/Data/results.txt', 'w')

count=0
print ('\nTest accuracies:\n')
file.write('Test accuracies:\n')
for i in test_accuracies:
    print('Fold '+ str(count) + ': ' + i)
    file.write('Fold '+ str(count) +': ' + i + '\n')
    count+=1

count=0
print ('\nMinimum validation losses:\n')
file.write('\nMinimum validation losses:\n')
for i, j in zip(min_validation_losses, min_validation_losses_epochs):
    print('Fold '+ str(count) + ': ' + i + ' at epoch ' + j)
    file.write('Fold '+ str(count) +': ' + i + ' at epoch ' + j + '\n')
    count+=1

count=0
print ('\nMaximum validation accuracies:\n')
file.write('\nMaximum validation accuracies:\n')
for i, j in zip(max_validation_accuracies, max_validation_accuracies_epochs):
    print('Fold '+ str(count) + ': ' + i + ' at epoch ' + j)
    file.write('Fold '+ str(count) +': ' + i + ' at epoch ' + j + '\n')
    count+=1

count=0
print ('\nFinal validation losses:\n')
file.write('\nFinal validation losses:\n')
for i in final_validation_losses:
    print('Fold '+ str(count) + ': ' + i)
    file.write('Fold '+ str(count) +': ' + i + '\n')
    count+=1

count=0
print ('\nFinal validation accuracies:\n')
file.write('\nFinal validation accuracies:\n')
for i in final_validation_accuracies:
    print('Fold '+ str(count) + ': ' + i)
    file.write('Fold '+ str(count) +': ' + i + '\n')
    count+=1

count=0
print ('\nEpochs of training:\n')
file.write('\nEpochs of training:\n')
for i in no_of_epochs:
    print('Fold '+ str(count) + ': ' + i)
    file.write('Fold '+ str(count) +': ' + i + '\n')
    count+=1

file.close()