sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt -y install unzip
sudo apt -y install python
sudo apt -y install python3-pip
sudo apt -y install libasound2-dev swig
sudo apt -y install nvidia-cuda-toolkit
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install git+https://github.com/vishnubob/python-midi@feature/python3
