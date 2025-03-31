import os
from .clean_data import create_datasets

curr_dir = os.getcwd().split('/')
if curr_dir[-1].lower() != 'sql_to_text':
    print(curr_dir)
    exit("Please run this script from the project directory")

create_datasets()
