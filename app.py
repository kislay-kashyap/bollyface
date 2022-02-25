import streamlit as st
from PIL import Image
import os
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from cv2 import cv2

model = VGGFace(model='resnet50', include_top = False, input_shape=(224,224,3), pooling = 'avg')

detector = MTCNN()

feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x,y,width,height = results[0]['box']
    face = img[y:y+height, x:x+width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    return result

def recommend(feature_list, features):
    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

    index_list = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])
    return index_list

st.title("Which Bollywood celebrity are You? (One that really works)")

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    if(save_uploaded_image(uploaded_image)):
        display_image = Image.open(uploaded_image)

        try:
            features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
            index_list = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_list[0][0]].split('/')[1].split('_'))
            
            st.header('Your uploaded image')
            st.image(display_image)

            st.title("The Full List")
            for i in range(0,5):
                st.write(" ".join(filenames[index_list[i][0]].split('/')[1].split('_')) + " ==> "+str(int(index_list[i][1]*100)) + "%")
                st.image(filenames[index_list[i][0]], width=200)

        except:
            st.image(display_image)
            st.title('Please upload an image with a face or a more clearer one.')
