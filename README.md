## Sign Language Classification using Convolutional Neural Networks
### CS 6001 - Masters Project by Frank Ibem and Susan Mengel (Advisor)

The goal of this project was to apply convolutional neural networks to the problem of
sign language classification in hopes of advancing the field. The models we experimented
with were evaluated on the [LSA64 Argentinian Sign Language dataset](http://facundoq.github.io/unlp/lsa64/).
Whilst the results don't outperform the current state of the art, the results are promising
and we are certain that with more data, they would be even better.

We experimented with 3 models:
1. 3D CNN
2. CNN-LSTM
3. Encoder-Decoder with Attention

Of the 3, the Encoder-Decoder with Attention model obtained the best results - an average
of 75% accuracy across both signer-dependent and signer-independent settings.

#### Requirements
- Python 3.5
- CNTK v2.2
- OpenCV 3
- Tensorboard

### Instructions
* Note that each brach defines a specific model. Switch to an appropriate brach when running

##### Step 1
Download the dataset from [here](https://mega.nz/#!5dJjQaLQ!POWOo88zyrAwNbBNP99F-YZNcWK4g2VIz0N3_gSF4gw).
Extract it and store alongside the code in a folder called data (if you use something else,
modify the following steps accordingly).

##### Step 2
```python labels.py```

This will generate a json file with id assignments for each label

##### Step 3
```python extract.py -np 6, -if ./data -of ./segments -s 20```

* This step requires considerable amount of time (I/O intensive)

This will extract 20 frames from each video file using 6 processes in parallel and store
them in './segments'. If a video file has less than the indicated number of frames, we
pad with the last frame. 

##### Step 4
```python create_dataset.py -np 6 -if ./segments -of ./dataset/stack``` 

* This step requires considerable amount of time and disk space (I/O intensive)

This will create train and test splits for the signer-dependent and signer-independent
experimental settings using multiple processes and store them in the CBF format. Multiple
files are created during the process which are then combined into a single file for a total
of 11 training/test files (10 signer-independent, 1 signer-dependent).
- 2560/640 train/test split for the signer-dependent setting
- 2880/320 train/test split for the signer-independent setting

###### Sequential models
```python stack2seq.py -np 6 -if ./dataset/stack -of ./dataset/sequential```

For models 2 and 3, the dataset needs to be in sequential form. Run the command above to
convert it.

##### Step 5
```python train.py -if ./dataset/sequential -of models/run1 -lf logs/run1 -e 2880 -c 320 -p 1```

* This applies for models 2 and 3 for the signer-independent settings. Modify accordingly

This will train a model using the default hyperparameters specified in train.py. In this
case, signer-independent setting 1 (all samples from user 1 are for testing). It is
important to specify the train/test sizes for the signer-independent settings.

##### Step 6
* Applies only to model 3
1. Run ```visualize.ipynb``` to evaluate the trained models on all the datasets and store the
results. A sample visualization of intermediate convolution and pooling layer outputs is also
generated and stored in ```./images```.

2. Run ```confusion.ipynb``` to generate confusion matrices using the results generated
in 1

3. (Optional) Run ```graphs.ipynb``` to plot and save the training progress (from Tensorboard)  