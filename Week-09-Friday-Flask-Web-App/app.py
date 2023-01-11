import flask #obv import
import os
import pickle
import pandas as pd
from skimage import io
from skimage import transform
#rest is stuff to help us out with our application

#step 1, find our app and the template folder. Much match names.
app = flask.Flask(__name__, template_folder='templates')

#loading in model files. will skip explanation for now
#vectorizor code inside create_model.py
path_to_vectorizer = 'models/vectorizer.pkl'
path_to_text_classifier = 'models/text-classifier.pkl'
path_to_image_classifier = 'models/image-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f: #rb means read binary. Pickle loads this stuff
    vectorizer = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)

with open(path_to_image_classifier, 'rb') as f:
    image_classifier = pickle.load(f)

#this empty slash is the home page
#HTML language, HTML recieves info from user, GET
#user input for instance, POST
@app.route('/', methods=['GET', 'POST'])  
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input!
        return(flask.render_template('index.html'))

#on the website after we click the submit button the user has POSTED info to us


    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text'] #looking for key `user_input_text`
        #we are storying the user input as a variable. When form is looking, we can find user_input_text in index.html

        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([user_input_text])
        
        # Make the prediction 
        predictions = model.predict(X)
        
        # Get the first and only value of the prediction.
        prediction = predictions[0]

        # Get the predicted probabs
        predicted_probas = model.predict_proba(X)

        # Get the value of the first, and only, predicted proba.
        predicted_proba = predicted_probas[0]

        # The first element in the predicted probabs is % democrat
        precent_democrat = predicted_proba[0]

        # The second elemnt in predicted probas is % republican
        precent_republican = predicted_proba[1]

#pass the HTML file to be rendered
        return flask.render_template('index.html', 
            input_text=user_input_text,
            result=prediction,
            precent_democrat=precent_democrat,
            precent_republican=precent_republican)
#variable names will be the variables we will be accessing.
# Remember to use curly braces for python code in HTML.



@app.route('/input_values/', methods=['GET', 'POST'])
def input_values():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('input_values.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        var_one = flask.request.form['input_variable_one']
        var_two = flask.request.form['another-input-variable']
        var_three = flask.request.form['third-input-variable']

        list_of_inputs = [var_one, var_two, var_three]

        return(flask.render_template('input_values.html', 
            returned_var_one=var_one,
            returned_var_two=var_two,
            returned_var_three=var_three,
            returned_list=list_of_inputs))

    return(flask.render_template('input_values.html'))

#for every page on our website we need a new route. This is called a decorator
#/images/ must corespond with an html file in our templates
#fcn name must also be the same. whenevr our app goes to images, run the script/fcn
@app.route('/images/')
def images(): #fcn name same as route name
    return flask.render_template('images.html') #will return this HTML file.


@app.route('/bootstrap/') #must include app route always
def bootstrap():
    return flask.render_template('bootstrap.html')


@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('classify_image.html'))

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file:
            # Read the image using skimage
            img = io.imread(file)

            # Resize the image to match the input the model will accept
            img = transform.resize(img, (28, 28))

            # Flatten the pixels from 28x28 to 784x0
            img = img.flatten()

            # Get prediction of image from classifier
            predictions = image_classifier.predict([img])

            # Get the value of the prediction
            prediction = predictions[0]

            return flask.render_template('classify_image.html', prediction=str(prediction))

    return(flask.render_template('classify_image.html'))

#last command of our app. Debug is true. if there is a bug, error message will display
#turn off in production code
if __name__ == '__main__':
    app.run(debug=True)