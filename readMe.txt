Make sure you have following files in your directory, in order to run the various examples:

1. vrep.py
2. vrepConst.py
3. the appropriate remote API library: "remoteApi.dll" (Windows), "remoteApi.dylib" (Mac) or "remoteApi.so" (Linux)

In order to create a dataset, load the file: bubleRob03.ttt into v-rep and play the simulation then back to python run create_dataset.py

 dataset will be generated in two folders:
    - data/capture for 10K (depending on the constants set in create_dataset.py) images generated in v-rep
    - data/perturbed for 1K (depending on the constants set in create_dataset.py) images that are randomly selected from the captured images and then perturbed using global lighting and occlusions.