# Face-Recognition-Emotion-Detection-in-Occlusion

Facial expression-based emotion detection is a challenging task, especially in the presence of occlusion. Face masks have become increasingly common in recent years, due to the COVID-19 pandemic. 
This has made it more difficult to accurately detect emotions from facial expressions. This project proposes a novel approach to facial expression-based emotion detection that can be used to accurately detect emotions from images with occlusion, including face masks.
The proposed approach is based on a deep learning model that is trained on a large dataset of images with and without occlusion. The model is able to learn to identify the facial features that are most important for emotion detection, even when these features are partially or fully occluded. 
The proposed approach was evaluated on a dataset of images with and without occlusion.The results showed that the proposed approach was able to achieve state-of-the-art accuracy for emotion detection in the presence of occlusion.

We have used convolutional neural network (CNN) in the model architecture to classify emotions. It has several layers, including convolutional and fully linked layers.
The network is fed a three-channel image (the dataset includes a 48X48 image), with each channel representing the RGB (Red, Green, Blue) values of the image pixels.
The first layer is a 3x3 kernel convolutional layer with 32 filters. From the input image, this layer retrieves low-level characteristics. Layers 2-6 proceed in a similar fashion, gradually increasing the number of filters to capture more complicated and abstract aspects. At each layer, the number of filters is doubled, beginning with 32 and progressing to
1024 in 6 layers
