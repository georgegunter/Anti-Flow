#!/bin/bash

echo "Installing system dependencies for SUMO"
sudo apt-get update
sudo apt-get install -y cmake swig libgtest-dev python-pygame python-scipy
sudo apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
sudo apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
sudo apt-get install -y build-essential curl unzip flex bison python3.8-dev python3.8
sudo pip3 install cmake cython

echo "Installing sumo binaries"
mkdir -p $HOME/sumo_binaries/bin
pushd $HOME/sumo_binaries/bin
wget https://github.com/CIRCLES-consortium/flow/releases/download/u20.04/sumo-binaries-ubuntu2004.tar.xz
tar -xf sumo-binaries-ubuntu2004.tar.xz
rm sumo-binaries-ubuntu2004.tar.xz
chmod +x *
popd
echo 'export PATH="$HOME/sumo_binaries/bin:$PATH"' >> ~/.bashrc
echo 'export SUMO_HOME="$HOME/sumo_binaries/bin"' >> ~/.bashrc
pip3 install https://github.com/CIRCLES-consortium/flow/releases/download/u20.04/sumotools-ubuntu2004.0-py3-none-any.whl
