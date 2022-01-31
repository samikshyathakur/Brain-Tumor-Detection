import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
import seaborn as sns


image_directory = "datasets/"
no_tumor_images=os.listdir(image_directory+'no/')
yes_tumor_images=os.listdir(image_directory+'yes/')

dataset=[]
label=[]
input_size=64
#print(no_tumor_images)
path='no0.jpg'
#print(path.split('.')[1])
for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((input_size,input_size))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((input_size,input_size))
        dataset.append(np.array(image))
        label.append(1)
        
#print(dataset)
#print (label)
dataset=np.array(dataset)
label=np.array(label)
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=0)

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)



#CROPPING THE IMAGE
from PIL import Image
im = Image.open('C:/sam_project/brain_tumordetection/datasets/no/no1.jpg')
left = 1
top = 2
right = 200
bottom = 200
im1 = im.crop((left, top, right, bottom))
im1.show()



#DATA AUGMENTATION
# Importing necessary functions
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
   
# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))
    
# Loading a sample image 
img5 = load_img('C:/sam_project/brain_tumordetection/pred/pred3.jpg') 
# Converting the input sample image to an array
x = img_to_array(img5)
# Reshaping the input image
x1 = x.reshape((1, ) + x.shape) 
   
# Generating and saving 5 augmented samples 
# using the above defined parameters. 
i = 0
for batch in datagen.flow(x1, batch_size = 1,
                          save_to_dir ='preview', 
                          save_prefix ='image', save_format ='jpg'):
    i += 1
    if i > 5:
        break



#MODEL BUILDING
model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(input_size,input_size,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(128,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
'''model.add(Dense(1))
model.add(Activation('sigmoid'))'''

model.add(Dense(2))
model.add(Activation('softmax'))
#it is for categorical
'''
Binary CrossEntropy=1,simodel
Categorical Cross entropy=2,softmax'''

'''
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

'''
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#it is for categorical'''
print(model.summary())


history = model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=35,validation_data=(x_test,y_test),shuffle=False)
model.save('brainTumor10Epochs.h5')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()


def names(number):
    if number==1:
        return 'Its A Tumor'
    else:
        return 'Its Not A Tumor'
    
    
    
img = Image.open(r"C:/sam_project/brain_tumordetection/datasets/yes/y20.jpg")
x = np.array(img.resize((64,64)))
x = x.reshape(1,64,64,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
plt.imshow(img)
print(str(res[0][classification]*100) + '% Confidence ' + names(classification))    
plt.title(names(classification))
plt.show()


img = Image.open(r"C:/sam_project/brain_tumordetection/datasets/no/no105.jpg")
y = np.array(img.resize((64,64)))
y = y.reshape(1,64,64,3)
res = model.predict_on_batch(y)
classification = np.where(res == np.amax(res))[1][0]
plt.imshow(img)
print(str(res[0][classification]*100) + '% Confidence ' + names(classification))    
plt.title(names(classification))
plt.show()



"""
ae = Autoencoder()
ae.x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)



def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds
   """ 
y_pred=model.predict(x_test).round(2) 
y_pred


model.evaluate(x_test,y_test)


model.predict(np.expand_dims(x_test[0],axis=0)).round(2)

np.argmax(model.predict(np.expand_dims(x_test[0],axis=0)).round(2))
y_test[0]




# CLASSIFICATION REPORT AND CONFUSION MATRIX
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(x_test, batch_size= 16)
predictions = np.argmax(predictions, axis= 1)
actuals = np.argmax(y_test, axis= 1)

print('Accuracy score is :', metrics.accuracy_score(actuals, predictions))
print('Precision score is :', metrics.precision_score(actuals, predictions, average='weighted'))
print('Recall score is :', metrics.recall_score(actuals, predictions, average='weighted'))
print('F1 Score is :', metrics.f1_score(actuals, predictions, average='weighted'))
print('ROC AUC Score is :', metrics.roc_auc_score(y_test, y_pred,multi_class='ovo', average='weighted'))
print('Cohen Kappa Score:', metrics.cohen_kappa_score(actuals, predictions))

print('\t\tClassification Report:\n', metrics.classification_report(actuals, predictions))


cm = confusion_matrix(actuals, predictions)
print(cm)
    

sns.heatmap(cm, annot = True, fmt ='d', xticklabels = 0, yticklabels = 1)

   
