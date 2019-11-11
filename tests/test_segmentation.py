import subprocess
import pathlib
import shutil
import os

def tearDown():
  test_output_path = "/tmp/aganesh_cats"
  shutil.rmtree(test_output_path) 
  

def test_output():
  test_output_path = "/tmp/aganesh_cats"
  if not os.path.isdir(test_output_path):
    try:    
      os.makedirs(test_output_path, exist_ok=True)
    except Exception as e:
      print(e)

  subprocess.call(["./segment.sh", "-p", "1", "/home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/data/datasets/en/sample-text", test_output_path])

  orig_output = "/home/nlp-text/dynamic/aganesh002/text-segmentation/cats_reinstall/data/datasets/en/sample-segmented/wiki_sample.txt.seg"
  
  test_output = os.path.join(test_output_path, "wiki_sample.txt.seg")

  with open(orig_output) as orig_file, open(test_output) as test_file:
    for orig_line, test_line in zip(orig_file, test_file):
      orig_score = orig_line.split()[-1]
      test_score = orig_line.split()[-1]
      assert orig_score[:5] == test_score[:5]
      
