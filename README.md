# IKR
IKR 2018 projekt, audio classification part

Potrebne nainstalovat:
```
pip3 install keras
pip3 install tensorflow
pip3 install tqdm
pip3 install librosa
```

Sources:
- https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
- https://github.com/manashmndl/DeadSimpleSpeechRecognizer
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

      
directory structure:
```
  .
  ├── eval
  │   ├── [evaluation data for clasification(could be mixed .png with wav)]
  ├── my_data
  │   ├── nontarget
  │       ├── [coppied all nontarget wav files only]
  │   └── target
  │       ├── [coppied all target wav files only]
  ├── nontarget.npy
  └── target.npy
```