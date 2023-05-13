# A-Hybrid-Spatial-Temporal-Deep-Learning-Architecture-for-Lane-Detection

This project performs lane detection on continuous driving scenes and approaches it as a segmentation task. The hybrid network implements a novel hybrid spatial-temporal sequence-to-one deep learning architecture that integrates the following aspects: (a) the single image feature extraction module equipped with the spatial convolutional neural network (SCNN); (b) the spatial-temporal feature integration module constructed by spatial-temporal recurrent neural network (ST-RNN); (c) the encoder-decoder structure, which makes this image segmentation problem work in an end-to-end supervised learning format. Several experiments were performed to measure the accuracy, precision, recall and F1 measure of various networks formed by a combination of different variants of ST-RNN and Encoder-Decoder modules along with SCNN module.

## Video Demonstration

This video was recorded on the campus of Worcester Polytechnic Institute. The results (i.e Lane Detection) are not accurate because the model was tested only after training it for 3 epochs. 

<img src="images/gitup.gif" width="640"> 
