This is the repository for text segmentation. The paper about the tool is in `CATS_Paper.pdf`. 

### Setup

* `conda create -n cats --file requirements.txt`
* `conda activate cats`


### Run

`segment.sh` and `segment.py` are basically doing the same thing. Only different is bash vs python

If you want to process so many files in an input directory, then use the following

* `sh segment.sh {input_dir} {output_dir}`
* `python segment.py {input_dir} {output_dir}`

If you want to give an input as an argument, then try this

* `python segment_input_text.py {input_text}`


### Unit tests
* `nosetests -v tests/test_segmentation.py`


### API

Login to any server (let's say bragi) and run the following:

`python api.py`

If you want it to be available for someone to use it, I would recommend to run it in a screen session. It may take a while (like a minute) to load the models. After the model is loaded and is ready to use, you will see like this in the terminal

```
python api.py
Loaded
 * Serving Flask app "api" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
 * Restarting with stat
Loaded
 * Debugger is active!
 * Debugger PIN: 102-564-776
```

Now, you are ready to submit text to get segmented output. To submit a sample text to segment, run as follows:

* `curl -i -H "Content-Type: application/json" -X POST -d '{"text":"{your_input_text}", "send_to_email": "{your_email_id}"}' http://bragi.research.ets.org:5000/post_segments`

You should be receiving an email to the provided email address with the segments of the input text.

##### Authors

* Binod Gyawali
* Ananya Ganesh

