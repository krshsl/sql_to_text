from .create_models import create
import os

curr_dir = os.path.dirname(os.path.realpath(__file__)).split('/')
if curr_dir[-1] != 'project':
    exit("Please run this script from the project directory")

create()
