EECS545 Final Project - Novel Speech Emotion Recognition Using Audio and Text

A. Data Preprocess Files
text_data_extraction.ipynb: pair the sentences and labels based on the recording time
text_data_preprocess.ipynb: set the words to lower case, expand the abbreviation, and remove the punctuation
sentence_label_comb.csv: store pre-processing result of text data
FC_label.txt: store label information of audio data
FC_MFCC12EDA.npy: store MFCC features of audio data in array format. It is too huge to upload to Canvas, you can download it from our group Google Drive: https://drive.google.com/a/umich.edu/file/d/1kw5wWWB3Y2VZMmUV8-kcyifT5H1HNjmW/view?usp=sharing

B. Network Training Files
(1)Audio Part
AER_train.py: train a neural network which categorizes a speech audio to one of four categories
model_trained_audio_only.pkl: store parameters of AER network
AER_test.py: test performance on 10% of data in IEMOCAP and achieve an accuracy of 53%

(2)Text Part
TER_train+test.py: train a neural network which analyses sentiment of text and categorizes a speech script to one of four categories, then test its performance on 10% data of IEMOCAP and achieves an accuracy of 61.5%

(3)Audio+Text Part
ATM_train.py: train a neural network which categories emotion type on both audio signal and speech text
model_train.pkl: store parameters of ATM network
ATM_test.py: test performance on 10% of data in IEMOCAP and achieve an accuracy of 68%




