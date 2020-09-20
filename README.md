<h1> Text Extraction</h1>
Feel free to fork/clone!<br>

In this project, we are trying to do these; <br>
1. Take a picture of a handwritten text, and upload it on a streamlit app https://docs.streamlit.io
2. Train the model using this tutorial: https://github.com/githubharald/SimpleHTR
3. Process the image using the model already trained. We may need to apply Threasholding if necessary
4. Extract text from Image, and save the text as .txt format
5. Smile at the Outcome😄


 For Heroku Deployment:
    1. Create Heroku account and Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli#download-and-install
    2. Create APP from Heroku Dashboard.
    3. Go to your APP dashboard settings on heroku website and in the buildpacks URL,<br>enter `https://github.com/heroku/heroku-buildpack-apt`<br>
    Now, Reveal Config vars and add <br>KEY: `TESSDATA_PREFIX`<br>VALUE:`./.apt/usr/share/tesseract-ocr/4.00/tessdata`
    3. Run cmd in current folder and enter `heroku login`( Logging into your account)
    4. After successful login, follow these steps:
    ```
    git add .
    git commit -am "First commit"
    heroku git:remote -a app-name
    git push heroku master
    ```
    5. App will be deployed at app-name.herokuapp.com

<br>

References:
1. Streamlit: https://docs.streamlit.io/en/stable/api.html
2. OpenCV Thresholding: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
3. Pytesseract: https://pypi.org/project/pytesseract/ 
4. TextBlob: https://pypi.org/project/textblob/
5. Tesseract deployment on Heroku using Flask: https://towardsdatascience.com/deploy-python-tesseract-ocr-on-heroku-bbcc39391a8d
