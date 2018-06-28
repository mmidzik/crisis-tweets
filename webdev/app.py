import flask
import numpy as np
import pandas as pd
import pickle
from flask import Flask, send_file,render_template, request, url_for, make_response
from flask_cors import CORS, cross_origin

#---------- MODEL IN MEMORY ----------------#

# Read in tweets with sentiment
# tweets = pickle.load(open("df_sent_with_ids_flask.pkl", "rb"))
tweets = pickle.load(open("df_sent_ner_flask.pkl", "rb"))


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

# allow CORS
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})



# Get an example and return it's score from the predictor model
@app.route("/tweet", methods=["POST"])
@cross_origin()
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, get the highest x tweet, and return id
    """
    # Get decision score for our example that came with the request
    data = request.form
    print(data)
    value = data['value']
    topic = data['topic']
    print(topic)
    value = int(value)
    tweet_id = 0
    if value == 1:
        topic_selected = tweets[tweets['entities'].apply(lambda x: topic in x)]
        selected = topic_selected.sort_values('polarity', ascending = False).reset_index()[:5]
        tweet_id = list(selected.sample(1).id)[0]
    if value == 0:
        topic_selected = tweets[tweets['entities'].apply(lambda x: topic in x)]
        selected = topic_selected.sort_values('polarity', ascending = True).reset_index()[:5]
        tweet_id = list(selected.sample(1).id)[0]
    results = {"tweet": tweet_id}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)
