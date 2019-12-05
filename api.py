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
    text_body = '\n'.join(body)
    content = MIMEText(text_body, 'html')
    msg.attach(content)

    msg['Subject'] = "segments"
    msg['From'] = "bgyawali@ets.org"
    msg['To'] = receipient

    # try sending the email; if it fails, raise an exception
    try:
        # send the message via our own SMTP server running on localhost
        s = SMTP('127.0.0.1')
        s.send_message(msg)
        s.quit()
    except Exception as e:
        raise Exception(str(e))


def segment_text(input_text, send_to):
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

        segmented_text = open(join(output_dir, 'input.txt.seg')).readlines()
        print(''.join(segmented_text))
        # send email
        send_email(segmented_text, send_to)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/post_segments', methods=["POST"])
def post_segments():
    request_json = request.get_json()
    input_text = request_json.get('text')
    send_to = request_json.get('send_to_email')
    executor.submit(segment_text, input_text, send_to)

    print(request_json)
    return (jsonify({"status" : "ok"}), 201)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
