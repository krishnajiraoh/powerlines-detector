import streamlit as st
import Unet
import cv2
import numpy as np

def load_model():
    model = Unet.build_unet_model()
    model.load_weights("model/100epochs/best_model.h5")
    return model


class PowerlineDetector():

    def __init__(self):
        self.model = load_model()

    def preprocess_image(self, img):
        img = cv2.resize(img, (128,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(img.shape[0], img.shape[1], 1)
        return img

    def detect(self, img):
        img = np.array([img])
        return self.model.predict(img)

st.set_page_config(
        page_title="Powerlines Detector",
        layout = "wide"
)

st.title("Powerlines Detector")

uploaded_file = st.file_uploader("Choose a PNG/JPG file", type=['png','jpg'], accept_multiple_files=False)

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    pd = PowerlineDetector()
    pp_img = pd.preprocess_image(img)
    pred_img = pd.detect(pp_img)

    c1,c2,c3 = st.columns(3)
    c1.subheader("Source Image")
    c1.image(img)

    c2.subheader("Preprocessed Image")
    c2.image(pp_img)

    c3.subheader("Detected Powerlines")
    c3.image(pred_img[0])
