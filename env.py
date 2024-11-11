import os
def load_dotenv(filename):
    with open(filename) as f:
        new = [tuple(i.split('=')) for i in f.readlines() if i]
        env = []
        for elem in new:
            env.append((elem[0].strip().lower(), '='.join(elem[1:]).strip('"\' \n')))
        os.environ.update(dict(env))

load_dotenv("resourcecenter.env")
