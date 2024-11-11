import os
def load_dotenv(filename):
  with open(filename) as f:
    os.environ.update(dict([tuple(i.split('=')) for i in f.readlines()]))

load_dotenv("resourcecenter.env")
