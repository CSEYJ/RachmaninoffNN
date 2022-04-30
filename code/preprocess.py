import pandas as pd
import json
import shutil,os
from constants import *

#load in metadata
js = pd.read_json('../data/maestro-v3.0.0/maestro-v3.0.0.json')

#list all composers, comp-genre pairs
composers = []
comp_genre = dict()
for i in styles:
    for j in i:
        genre = j.split('/')[-2]
        comp = j.split('/')[-1]
        comp_genre[comp] = genre
        composers += [comp]

#find all paths corresponding to a composer  
def find_composer_path(df, name):
    
    return '../data/maestro-v3.0.0/' + df[df.canonical_composer.str.lower().str.contains(name)].midi_filename

#composer with paths in a dictionary
name_path = dict()
for i in composers:
    name_path[i] = find_composer_path(js, i)

for i in composers:
    paths = name_path[i]
    for j in paths:
        directory = os.path.join('../data', comp_genre[i], i)
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copy(j, directory)