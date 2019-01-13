wget -O ./model/model_all_r100134e7963.h5 'https://www.dropbox.com/s/374l81dqun0asqz/model_all_r100134e7963.h5?dl=1'
wget -O ./model/model_weights_all_r100134e7963.h5 'https://www.dropbox.com/s/fgqwohrbkr12sv6/model_weights_all_r100134e7963.h5?dl=1'
cd ./src/Guide/
python3 Guide_0.py
cd ../../
cd ./src/SufferMing
python3 Testing.py
cd ../../
cd ./src/Guide
python3 Guide_4.py
cd ../../ 
