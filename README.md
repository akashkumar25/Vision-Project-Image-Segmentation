# Vision-Project-Image-Segmentation

#### In this project we implemented the image segmentation using three different models for three different task.

#### Task1:

For this task we have implemented Vgg-16 pretrained model on PascalVoc 2012 Dataset. We got the F1- score of 0.8560 and Dice-coeffiecient as 0.7482.

#### Task2:

For this task we have to implement exactly same R2-UNet model given in this paper https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf on CityScape dataset. We got the training accuracy as 0.9904, sensitivity as 0.8713, specificity as 0.9950, F1-score as 0.8713, and Jaccard score as 0.7720 and testing accurracy as 0.9843, sensitivity as 0.8042, specificity as 0.9918, F1-score as 0.8042, and Jaccard score as 0.6725.

#### Task 3:

For this task, we have to improve the existing results of task 2. So in this work we combined the R2-UNet model with an attention gate known as Focusnet. We got the training accuracy as 0.9982, sensitivity as 0.8854, specificity as 0.9932, F1-score as 0.88632, and Jaccard score as 0.8056 and testing accurracy as 0.9862, sensitivity as 0.8323, specificity as 0.9543, F1-score as 0.8413, and Jaccard score as 0.7523.







### Reference:

https://github.com/LeeJunHyun/Image_Segmentation



