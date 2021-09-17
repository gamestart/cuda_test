#! /bin/bash
set -e

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P)"
ROOT_DIR=$SCRIPT_DIR/../

download_unzip () {
    URL=$1
    DIR=$2
    echo "downloading from $URL and putting in $DIR"
    rm  -rf $DIR
    mkdir -p $DIR
    wget $URL -O $DIR/package.zip
    unzip $DIR/package.zip -d $DIR
    rm $DIR/package.zip
}

OPENCV_URL=http://192.168.100.12/artifactory/opencv/opencv-4.2.0-linux-x64.zip
OPENCV_DIR=$ROOT_DIR/third_party/opencv

download_unzip $OPENCV_URL $OPENCV_DIR