import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("/Users/chaitalishah/Desktop/Krsna_WHJ/Projects/Project-110/converted_keras/keras_model.h5")
camera = cv2.VideoCapture(0)

while True:

	status , frame = camera.read()
	
	frame = cv2.flip(frame , 1)
	img = cv2.resize(frame,(224,224))
		
    #test_img = np.array(img,dtype = np.float32)
    #test_img = np.expand_dims(test_img, excess = 0)
    	
	#normalised_img = test_img/255.0

    #prediction = model.predict(normalised_img)
    #print("Prediction: ",prediction)

	cv2.imshow('Feed' , frame)

	code = cv2.waitKey(1)
	if code == 32:
		break

camera.release()
cv2.destroyAllWindows()
