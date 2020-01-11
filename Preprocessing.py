from CommonVariables import *
import cv2
import keras.backend as K
from keras.preprocessing.image import load_img
from keras.utils import to_categorical

# Creates Data directory.
if not os.path.exists(output_folder+'/Data'):
    os.mkdir(output_folder+'/Data')
    print("Directory " + output_folder + '/Data',  "created.")
else:
    print("Directory " + output_folder + '/Data',  "already exists.")

# Creates a CLAHE object (arguments are optional).
clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8,8))

# Function for loading all images and labels in NumPY format (uncomment to perform some useful operations with OpenCV or Numpy).
def load_images(folder, img_data_list, ground_truth):
    class_label = -1
    folder_dir = os.listdir(folder)
    for dataset in folder_dir:
        img_list = os.listdir(folder+'/'+ dataset)
        print ('Loading the images from ' + folder + '/{}'.format(dataset))
        class_label = class_label + 1
        for img in img_list:
            input_img = cv2.imread(folder + '/' + dataset + '/' + img , cv2.IMREAD_GRAYSCALE)
            #input_img = cv2.resize(input_img, (int(image_size),int(image_size)))
            # Histogram equalization.
            # input_img=cv2.equalizeHist(input_img)
            # CLAHE
            # input_img = clahe.apply(input_img)
            
            # Operations with OpenCV.
            # Normalization.
            #input_img=cv2.normalize(input_img, input_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # Z-score standardization.
            #mean, std = cv2.meanStdDev(input_img)
            #input_img=(input_img-mean)/std

            # The same operations as before but performed with NumPY.
            #input_img=input_img.astype('float32')
            # Normalization.
            #input_img = input_img/255.0
            # Z-score standardization.
            #input_img -= np.mean(input_img)
            #input_img /= np.std(input_img)

            img_data_list.append(input_img.astype('float32'))
            ground_truth.append(class_label)
    img_data = np.array(img_data_list)
    print(img_data.shape)
    if num_channel == 1:
        if K.image_dim_ordering() == 'th':
            img_data = np.expand_dims(img_data, axis=1)
            print (img_data.shape)
        else:
            img_data= np.expand_dims(img_data, axis=4)
            print (img_data.shape)
    else:
        if K.image_dim_ordering() == 'th':
            img_data=np.rollaxis(img_data, 3, 1)
            print (img_data.shape)

print("Preprocessing started...")

train_images = []
train_ground_truth = []
test_images = []
test_ground_truth = []

load_images(train_path, train_images, train_ground_truth)
load_images(test_path, test_images, test_ground_truth)

def process_images(images):
    images = np.asarray(images)
    images = np.expand_dims(images, -1)
    return images

# Process the train and test images and save them.
train_images=process_images(train_images)
test_images=process_images(test_images)
print('Train images: min: ' + str(train_images.min()) + ', max: ' + str(train_images.max()))
print('Test images: min: ' + str(test_images.min()) + ', max: ' + str(test_images.max()))
np.save(output_folder + '/Data/Train_images', train_images)
np.save(output_folder + '/Data/Test_images', test_images)

# One-hot encode of the ground truths and saving.
train_ground_truth = np.array(to_categorical(train_ground_truth, class_number))
test_ground_truth = np.array(to_categorical(test_ground_truth, class_number))
np.save(output_folder + '/Data/Train_ground_truth', train_ground_truth)
np.save(output_folder + '/Data/Test_ground_truth', test_ground_truth)

print("Preprocessing Done.")
print("Training:")
print("Image size: " + str(len(train_images[0])))
print("Number of images in train dataset: " + str(len(train_images)))
print("Features and labels stored in Train_Images.npy and Train_ground_truth.npy respectively.")
print("Test:")
print("Image size: " + str(len(test_images[0])))
print("Number of images in test dataset: " + str(len(test_images)))
print("Features and labels stored in Test_Images.npy and Test_ground_truth.npy respectively.")
print("NB all the data are in the Data folder.")