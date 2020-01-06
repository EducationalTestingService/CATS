'''
This is an API to call the logic of segmenting an input text.

Author: Binod Gyawali
Date: November, 2019
'''

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os.path import join
from smtplib import SMTP

from flask import Flask, jsonify, request

import segment
import utils
import requests
import json
import html

# load the embeddings and vocabulary
embeddings, vocabulary = utils.load_models()
app = Flask(__name__)
executor = ThreadPoolExecutor(1)


def send_email(body, receipient):
    """
    A function that uses the SMTP library to send an
    email with the given body and receipients.

    Parameters
    ----------
    body : str
        Body of the email to be sent.

    Raises
    ------
    Exception
        If the email could not be sent.
    """

    # set up the email S/MIME and other fields
    # depending on whether we are sending the email
    # as plain text or not
    msg = MIMEMultipart('alternative')
    text_body = body
    content = MIMEText(text_body)
    msg.attach(content)

    msg['Subject'] = "segments"
    msg['From'] = "bgyawali@ets.org"
    msg['To'] = receipient

    # try sending the email; if it fails, raise an exception
    try:
        # send the message via our own SMTP server running on localhost
        s = SMTP('127.0.0.1')
        s.send_message(msg)
        print('Email sent successfully')
        s.quit()
    except Exception as e:
        raise Exception(str(e))

def submit_segments(segments, request_id, return_url):
    segments_str = '====='.join(segments)
    segments_str_encoded = html.escape(segments_str)
    print(segments_str)
    print(request_id)
    print(return_url)
    #post_url = 'http://c3dev.research.ets.org/TextSegmentation/handlers/SegmentationResponse.ashx'
    try:
        to_submit = json.dumps({'requestId': request_id, 'segmentedText': segments_str_encoded, 'statusCode': 1, 'statusMessage': 'completed'})
    except Exception as e:
        print(e)
    r = requests.post(return_url, data={'message': to_submit})
    print(r.status_code, r.reason)

def segment_text(input_text, request_id, return_url):
    """
    A function to segment the input text and send an email with the segments.
    Paramters
    ---------
    input_text: str
        The text to segment
    send_to: str
        The email address to send back the segments
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        input_dir = join(temp_dir, 'input')
        output_dir = join(temp_dir, 'output')
        for dirname in [input_dir, output_dir]:
            os.makedirs(dirname)
        with open(join(input_dir, 'input.txt'), 'w') as wf:
            wf.write(input_text)

        segment.run_segmentation(input_dir, output_dir, embeddings, vocabulary)

        segmented_text = open(join(output_dir, 'input.txt.seg')).read()
        segments = segmented_text.split('=====')
        #print('==='.join(segments))
        submit_segments(segments, request_id, return_url)
        # send email
        #send_email(segmented_text, send_to)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/post_segments', methods=["POST"])
def post_segments():
    request_json = request.get_json()
    input_text = html.unescape(request_json.get('passageText'))
    request_id = request_json.get('requestId')
    return_url = request_json.get('returnURL')
    #send_to = request_json.get('send_to_email')
    executor.submit(segment_text, input_text, request_id, return_url)

    print(request_json)
    return (jsonify({"status" : "ok"}), 201)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
