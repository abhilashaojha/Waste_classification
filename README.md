
# Waste Classification Model

This repository contains code for training a waste classification model using deep learning. The model is designed to classify images of different types of waste into six categories: electronic waste, medical waste, plastic waste, metal waste, paper waste, and glass waste.

```
project
│   README.md
|   waste_train.py 
|   google_image_scraper.py
|    └───imgs
│       │   e_waste
│       │   medical_waste
│       │   plastic_waste
|       |   metal_waste
│       |   paper_waste
|       |   glass_waste
└───────|   sample_image       

```

## Dataset

The dataset images were scraped through the automation python script `google_image_scraper.py` that uses selenium to scrape images. 

The training data is then organized into directories for each waste category. 

The image data is augmented using techniques like shearing, zooming, and horizontal flipping to increase model robustness.

### Dataset Structure

The dataset is organised as mentioned above with `imgs` as the parent directory and other folders as the sub-directory.


## Dependencies

- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Python](https://www.python.org/)

Install dependencies using:

```bash
pip install keras tensorflow
```

## Model Architecture

The waste classification model is a convolutional neural network (CNN) with the following architecture:

* Convolutional layers with ReLU activation
* MaxPooling layers
* Dense layers with ReLU activation
* Softmax activation for multi-class classification

## Training
The model is trained using the Adam optimizer and categorical crossentropy loss. During training, the best weights are saved using the ModelCheckpoint callback.

To train the model, run the following command:
```
python waste_train.py
```
Adjust the hyperparameters (e.g., epochs, batch size) in the script based on your requirements.

## Saving Best Weights
The best weights of the model are saved in a file named waste_model.h5. These weights can be used for making predictions on new data without retraining the entire model.

## Output
The model training script will output information about the training process, including accuracy and loss metrics. Additionally, the prediction script will display the predicted waste category for a given image.

## Improvements
This repository is still under development. Further enhancements include: 
* Continuous model improvement through regular updates and fine-tuning based on feedback and new data.
* Integration with existing waste management systems, development of user-friendly interfaces, and providing training to personnel for enhancing the model's usability and adoption.
* Collaboration with stakeholders, such as regulatory bodies and recycling facilities

## Conclusion

The project aims to provide a practical solution for automating waste classification, thereby improving efficiency, reducing contamination, and promoting environmentally sustainable waste management practices.

## Author
* Abhilasha Ojha


