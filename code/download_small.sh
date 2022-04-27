#!/bin/bash

curl https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip --output ../data/maestro-v3.0.0-midi.zip
unzip ../data/maestro-v3.0.0-midi.zip -d ../data/
rm ../data/maestro-v3.0.0-midi.zip