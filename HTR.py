import streamlit as st
from PIL import Image
import pytesseract as pt
from textblob import TextBlob
import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt
#pt.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
pt.pytesseract.tesseract_cmd = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"'

st.sidebar.markdown(
    """<style>body {background-color: #2C3454;color:white;}</style><body></body>""", unsafe_allow_html=True)
st.markdown("""<h2 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>Our HTR Project</h2><h2 style='text-align: center; color: white;font-size:30px;margin-top:-30px;'></h2>""", unsafe_allow_html=True)


def intro():
    st.markdown("""<h2 style='text-align: left; color: white;'>TASK</h2><p style='color: white;'>Extract text from an handwritting image and save as .txt file</p>""", unsafe_allow_html=True)
    st.markdown("""<h2 style='text-align: left; color: white;'>Our Approach</h2><p style='color: white;'>
    <ul style='text-align: left; color: white;'><li>Extract Text from Images using Pytesseract API</li>
    <li>Train model using IAM datasaet</li>
    <li>Improve accuracy</li>
    <li>Analyze model</li>
    <li>Improve accuracy</li></ul></p>""", unsafe_allow_html=True)
    st.markdown("""<a style='text-align: center; color: white;font-size:30px;' href="" target="_blank">Github Link</a>""", unsafe_allow_html=True)


st.sidebar.markdown(
    "<h1 style='text-align: center;color: #2C3454;margin-top:30px;margin-bottom:-20px;'>Select Image</h1>", unsafe_allow_html=True)

image_file = st.sidebar.file_uploader("", type=["jpg", "png", "jpeg"])


def sentiments(p):
    if(p > 0):
        return("Positive")
    elif(p < 0):
        return("Negative")
    else:
        return("Random")


def extract(img):
    slide = st.sidebar.slider("Select Page Segmentation Mode", 1, 14)
    conf = f"-l eng --oem 3 --psm {slide}"
    text = pt.image_to_string(img, config=conf)
    if(text != ""):
        st.markdown("<h1 style='color:yellow;'>Extracted Text</h1>",
                    unsafe_allow_html=True)
        slot1 = st.empty()
        slot2 = st.empty()
        if(st.sidebar.checkbox("Apply Spelling Correction")):
            corrected = TextBlob(text).correct()
            slot1.markdown(f"{corrected}", unsafe_allow_html=True)
            polar = round(corrected.sentiment.polarity, 2)
            slot2.markdown(
                f"""<h1 style='color:yellow;'>Polarity: <span style='color:white;'>{polar}</span> Sentiment: <span style='color:white;'>{sentiments(polar)}</span></h1>""", unsafe_allow_html=True)
        if(st.sidebar.checkbox("Remove Numbers & Special Characters")):
            filtered = re.sub(r'[^A-Za-z ]+', '', text)
            slot1.markdown(f"{filtered}", unsafe_allow_html=True)
            polar = round(TextBlob(filtered).sentiment.polarity, 2)
            slot2.markdown(
                f"""<h1 style='color:yellow;'>Polarity: <span style='color:white;'>{polar}</span> Sentiment: <span style='color:white;'>{sentiments(polar)}</span></h1>""", unsafe_allow_html=True)
        else:
            slot1.markdown(f"{text}", unsafe_allow_html=True)
            polar = round(TextBlob(text).sentiment.polarity, 2)
            slot2.markdown(
                f"""<h1 style='color:yellow;'>Polarity: <span style='color:white;'>{polar}</span> Sentiment: <span style='color:white;'>{sentiments(polar)}</span></h1>""", unsafe_allow_html=True)
    else:
        st.markdown(
            """<h1 style='text-align: center;'>TEXTLESS IMAGE</h1>""", unsafe_allow_html=True)


def plot(name, value):
    st.markdown(
        f"""<h1 style='text-align: center;'>{name} OUTPUT</h1>""", unsafe_allow_html=True)
    plt.imshow(value, 'gray')
    thplot = "Thresholded plots\plot.png"
    plt.savefig(thplot, transparent=True)
    st.image('Thresholded plots/plot.png', width=600)
    extract(value)


if image_file is not None:
    st.markdown("<h1 style='color:yellow;'>Uploaded Image</h1>",
                unsafe_allow_html=True)
    st.image(image_file, width=400)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    radio = st.sidebar.radio(
        "Select Action", ('Text Extraction', 'Thresholding'))
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if(radio == "Text Extraction"):
        extract(img)
    else:
        types = ['Binary', 'Binary Inverse', 'Truncate', 'Tozero', 'Tozero Inverse',
                 'Global Thresholding', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        thresh = st.sidebar.selectbox("Select Threshold Type", types)
        if(thresh == types[0]):
            if(st.sidebar.checkbox("Grayscale")):
                img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            slide = st.sidebar.slider("Select threshold", 0, 255)
            ret, th = cv.threshold(img, slide, 255, cv.THRESH_BINARY)
            plot(thresh, th)
        elif(thresh == types[1]):
            if(st.sidebar.checkbox("Grayscale")):
                img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            slide = st.sidebar.slider("Select threshold", 0, 255)
            ret, th = cv.threshold(img, slide, 255, cv.THRESH_BINARY_INV)
            plot(thresh, th)
        elif(thresh == types[2]):
            if(st.sidebar.checkbox("Grayscale")):
                img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            slide = st.sidebar.slider("Select threshold", 0, 255)
            ret, th = cv.threshold(img, slide, 255, cv.THRESH_TRUNC)
            plot(thresh, th)
        elif(thresh == types[3]):
            if(st.sidebar.checkbox("Grayscale")):
                img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            slide = st.sidebar.slider("Select threshold", 0, 255)
            ret, th = cv.threshold(img, slide, 255, cv.THRESH_TOZERO)
            plot(thresh, th)
        elif(thresh == types[4]):
            if(st.sidebar.checkbox("Grayscale")):
                img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            slide = st.sidebar.slider("Select threshold", 0, 255)
            ret, th = cv.threshold(img, slide, 255, cv.THRESH_TOZERO_INV)
            plot(thresh, th)
        elif(thresh == types[5]):
            if(st.sidebar.checkbox("Grayscale")):
                img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            img = cv.medianBlur(img, 5)
            slide = st.sidebar.slider("Select threshold", 0, 255)
            ret, th = cv.threshold(img, slide, 255, cv.THRESH_BINARY)
            plot(thresh, th)
        elif(thresh == types[6]):
            img = cv.medianBlur(img, 5)
            img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                      cv.THRESH_BINARY, 11, 2)
            plot(thresh, th)
        elif(thresh == types[-1]):
            img = cv.medianBlur(img, 5)
            img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
            th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, 11, 2)
            plot(thresh, th)
else:
    intro()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
