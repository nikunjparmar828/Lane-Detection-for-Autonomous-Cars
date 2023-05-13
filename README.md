# A-Hybrid-Spatial-Temporal-Deep-Learning-Architecture-for-Lane-Detection

This project performs lane detection on continuous driving scenes and approaches it as a segmentation task. The hybrid network implements a novel hybrid spatial-temporal sequence-to-one deep learning architecture that integrates the following aspects: (a) the single image feature extraction module equipped with the spatial convolutional neural network (SCNN); (b) the spatial-temporal feature integration module constructed by spatial-temporal recurrent neural network (ST-RNN); (c) the encoder-decoder structure, which makes this image segmentation problem work in an end-to-end supervised learning format. Several experiments were performed to measure the accuracy, precision, recall and F1 measure of various networks formed by a combination of different variants of ST-RNN and Encoder-Decoder modules along with SCNN module.

## Network Architecture 

<img src="images/Network.png" width="640">

## tvtLANE Dataset
This project uses tvtLANE Dataset for the training and evaluation of the network. This dataset contains 19383 image sequences for lane detection, and 39460 frames of them are labeled. These images were divided into two parts, a training dataset contains 9548 labeled images and augmented by four times, and a test dataset has 1268 labeled images. The size of images in this dataset is 128*256. 

Detailed imformation about the dataset and it's augmentation process can be found https://github.com/qinnzou/Robust-Lane-Detection 

The dataset can be downloaded from https://drive.google.com/drive/folders/1MI5gMDspzuV44lfwzpK6PX0vKuOHUbb_?usp=sharing

#### The dataset folder should have following structure:
 
 * dataset
   * trainset
     *  trainset
        * image
        * truth
     *  train_index.txt
     *  val_index.txt     
   * testset
     * image
     * truth
     * test_index_0530.txt
     * test_index_0531.txt 
     * test_index_0601.txt
   
*Note* - If you have different folder structure, you will need to adjust the paths accordingly.


## Results

The figure below shows the detection of Lane in different conditions:

Lane Detection in Normal Condition              |  Lane Detection under Occlusion
:----------------------------------------------:|:------------------------------------------------:
<img src="images/result4.jpg" width="480">      |  <img src="images/result3.jpg" width="480">

Lane Detection inside a tunnel                  |  Failed Lane Detection in low lightning condition
:----------------------------------------------:|:------------------------------------------------:
<img src="images/result1.jpg" width="480">      |  <img src="images/result5.jpg" width="480">


## Video Demonstration

This video was recorded on the campus of Worcester Polytechnic Institute. The results (i.e Lane Detection) are not accurate because the model was tested only after training it for 3 epochs. 

<img src="images/WPI_campus.gif" width="640"> 
