import datetime as dt
import os

def results_timestamp_dir():
    timestamp = str(dt.datetime.now())
    timestamp = timestamp.replace(' ', '_').replace(':', '-').replace('.', '-')
    project_path = os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0]
    _run_folder = os.path.join(project_path, 'results', timestamp)
    return _run_folder
