#!/bin/bash

curl https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip --output ../data/maestro-v3.0.0.zip 
unzip ../data/maestro-v3.0.0.zip -d ../data/
rm ../data/maestro-v3.0.0.zip
