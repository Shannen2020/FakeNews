"""
Offload as much pre-computations, loading and such before deploying the endpoint.
Endpoint should be light-weighted and should perform minimal information loading of any kinds.

Directory at container runtime:

WD:assignment8
    |---src
        |---static
            |---<TODO: adding CSS for HTML templates>
        |---templates
            |---index.html
        |---upload
            |---<KIV: placeholder if intending to save file in container; inadvisable>
        |---___init___.py
        |---app.py
        |---inference.py
    |---tests
        |---__init__.py
        |---<TODO: unit tests for ETL and model>
    |---log
        |---app_YYYY_MM_DD_HH_MM_SS_MS.log
    |---conda.yml
    |---model.h5
    
Application presumably runs in WD:assignment8 with `python -m src.app.py`

References:
    - https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    - https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-ii-templates
"""

import sys
import os
import errno
import time
import logging
from datetime import datetime
from flask import Flask, Response, jsonify, request, render_template, \
    redirect, url_for, abort, send_from_directory, redirect, flash
from werkzeug.utils import secure_filename
from waitress import serve
import tensorflow as tf
from src.modelling.dataloader import Dataloadpipeline
from src.datapipeline.pipeline import Pipeline
import sys
import tensorflow_hub as hub
from readable_content.parser import ContentParser
import html
from urllib.error import HTTPError
from lxml.etree import XMLSyntaxError
from langdetect import detect
import iso639

####################################
### App configuration parameters ###
####################################
# Resource and file paths
## Model related
MODEL_FPATHS = ['./model/fakenews_nnlm.h5', './model/model.h5']

## Flask-app directories and resources 
UPLOAD_FPATH = './src/uploads'
STATIC_FPATH = 'static'
TEMPLATES_FPATH = 'templates' # relative to where `app.py` is
ACCEPTABLE_FEXT = ['.txt']
MAX_FILE_SIZE_BYTES = 1024 * 1024 # 1MB

## General run-time configs.
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

try:
    os.makedirs('./log')
except OSError as e:
    if e.errno != errno.EEXIST:
        print('Logging directory already exists')
        
runtime_log_fname = './log/' \
    + os.path.basename(sys.argv[0]).replace('.py', '') \
    + datetime.utcfromtimestamp(time.time()).strftime('_%Y_%m_%d_%H_%M_%S_%f') \
    + '.log'
logging.basicConfig(format='', filename=runtime_log_fname, filemode='w')


#######################################
### Initialised inference resources ###
#######################################
# Load model into memory space
# The NNLM needs a bit of love to handle custom hub.KerasLayer object
model_nnlm = tf.keras.models.load_model(MODEL_FPATHS[0], custom_objects={'KerasLayer':hub.KerasLayer})
model_gru = tf.keras.models.load_model(MODEL_FPATHS[1])

#################################
### Web application functions ###
#################################
app = Flask(
    import_name=__name__,
    static_folder=STATIC_FPATH,
    template_folder=TEMPLATES_FPATH)
app.config['UPLOAD_PATH'] = UPLOAD_FPATH
app.config['UPLOAD_EXTENSIONS'] = ACCEPTABLE_FEXT
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_BYTES
app.config['SECRET_KEY'] = 'Fake news is all you need!'
app.config['SESSION_TYPE'] = 'filesystem' # Avoid cookie size issue


##############################
### Error handling ###
@app.errorhandler(404)        
def error_404(error):
    return render_template('404.html'), 404

@app.errorhandler(500)        
def error_500(error):
    return render_template('500.html'), 500

@app.errorhandler(HTTPError)        
def error_http(error):
    return render_template('httperr.html'), 500

@app.errorhandler(XMLSyntaxError)        
def error_xml(error):
    return render_template('httperr.html'), 500

@app.errorhandler(ValueError)
def error_value(error):
    return render_template('urlerr.html'), 500

##############################
### Base information pages ###
@app.route('/', methods=['GET'])
def index():
    current_pred = ''
    return render_template('form.html', css_pred=current_pred)

#####################################
### Testing download functionality
@app.route('/download', methods=['GET', 'POST'])
def download():
    if request.method=='POST':
        req = request.form
        url = req['url']
        model = req['model']

        parser = ContentParser(url)
        content = parser.get_content()
        content = html.unescape(content) 
        det_lang = detect(content)[0:2]

        if det_lang != 'en':
            flash(content[0:1000])
            det_langname = iso639.to_name(det_lang)
            return render_template('form.html', css_pred="That's " + det_langname + ". English only!")

        if model=='NNLM':
            # Pre-processing of the pipeline
            pipeline = Pipeline()
            content_cleaned = pipeline.process_raw_string(content)
            flash(content_cleaned['text_clean'].values[0][0:1000])

            pred = model_nnlm.predict([content_cleaned['text_clean'].values[0]]) 
            
            score = pred[0][0]
            
            if score < -0.5:
                message = "Absolutely real."

            if (score >= -0.5) and (score < 0):
                message = "Might be real news."

            if (score >= 0) and (score <= 0.5): # since message is a requirement and if prediction = 0, it doesn't get set
                message = "This might be fake news."

            if score > 0.5:
                message = "Totally fake news!"

            # score = tf.nn.softmax(pred)[0][0].numpy()

            # if score < 0.25:
            #     message = "Absolutely real."

            # elif (score >= 0.25) and (score < 0.5):
            #     message = "Might be real news."

            # elif (score >= 0.5) and (score < 0.75):
            #     message = "This might be fake news."

            # else score > 0.75:
            #     message = "Totally fake news!"

            return(render_template("form.html", css_pred=message))

        elif model=='GRU':
            
            pipeline = Dataloadpipeline()
            content_tokenized = pipeline.process_raw_string([content])
            score = model_gru.predict(content_tokenized)
            flash(content[0:1000])
            
            if score < 0.25:
                message = "Absolutely real."

            if (score >= 0.25) and (score < 0.5):
                message = "Might be real news."

            if (score >= 0.5) and (score < 0.75):
                message = "This might be fake news."

            if score > 0.75:
                message = "Totally fake news!"

            return(render_template("form.html", css_pred=message))
            
    return render_template("form.html")


#####################################
### Model serving and predictions ###
@app.route('/predict', methods=['POST'])
def predict():
    current_pred = 'No prediction'
        
    # Sanitise user uploaded file before reading into memory for inference
    uploaded_file = request.files['css_txt_file']
    filename = secure_filename(uploaded_file.filename)

    if filename != '':
        file_ext = os.path.splitext(filename)[1]

        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            print('File with invalid extension received, aborting operation')
            abort(Response(f'You have attempted to upload a file with an invalid extension! \
            Acceptable format(s) are {ACCEPTABLE_FEXT}'))

    # Read payload stream as text, convert to UTF-8 strings to predict
    uploaded_text = uploaded_file.read().decode('utf-8')
    
    ## Amend code below when model / transform scripts are ready
    processed_text = uploaded_text
    # processed_text = dpl.transform_data(uploaded_text)
    # pred_proba = model.predict(processed_text)

    # pred = pred_proba.argmax(1)
    # pred_label = 'Fake News' if pred == 1 else 'Not Fake News'
    # current_pred = {'Prediction':pred_label, 'Probability':pred_proba}

    current_pred = str(processed_text) # placeholder casting as string

    return render_template('form.html', css_pred=current_pred)


if __name__ == "__main__":
    
    # app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    serve(app, host="0.0.0.0", port=8000)

    
    

        
        
        
        
