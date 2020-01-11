from __future__ import division
from CommonVariables import *
from keras.models import Sequential, model_from_json, load_model
from keras.preprocessing.image import ImageDataGenerator

# Put here the fold that performed best.
best_model=str(...)

# Choose the mode you prefer to load the model.
# Load the entire model in .json format.
json_file = open(output_folder + '/Models/Model'+best_model+'/Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights(output_folder + '/Models/Model'+best_model+'/Model_weights.h5')

# Load the entire model in .h5 format.
#loaded_model = load_model('./Data/Checkpoint_Model.h5')
print("Model loaded.")

test_images = np.load(output_folder + '/Data/Test_images.npy')
test_ground_truth = np.load(output_folder + '/Data/Test_ground_truth.npy')

ground_truth=[]
predictions=[]

# NO DATA AUGMENTATION.
# Operations on the test images.
# Normalization.
#test_images = test_images/255.0
# Z-score standardization.
#test_images -= np.mean(test_images)
#test_images /= np.std(test_images)
#pred = loaded_model.predict(test_images)

# WITH DATA AUGMENTATION.
TestDatagen = ImageDataGenerator(
    samplewise_center = True, # subtract the mean from the image.
    samplewise_std_normalization = True, # divides by standard deviation.
    #rescale = 1.0 / 255.0, # a value by which we will multiply the data after any other processing.
)
test_iterator = TestDatagen.flow(test_images, test_ground_truth, shuffle=False, batch_size=len(test_images))
pred = loaded_model.predict_generator(test_iterator, steps=1)


yh = pred.tolist()
yt = test_ground_truth.tolist()

for i in range(len(test_ground_truth)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predictions.append(yh[i].index(yy))
    ground_truth.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1

accuracy = (count/len(test_ground_truth))*100

# Saving values for confusion matrix and analysis.
np.save(output_folder+'/Data/Ground_truth', ground_truth)
np.save(output_folder+'/Data/Predictions', predictions)
print("Predicted and true label values saved")
print("Accuracy on test dataset :"+str(accuracy)+"%")