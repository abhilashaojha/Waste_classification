import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 6  # Number of classes

# Define dataset paths
DATASET_PATH = 'D:/Desktop/waste_classification/imgs/'
TRAIN_PATH = os.path.join(DATASET_PATH)

# Image data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['e_waste', 'medical_waste', 'plastic_waste', 'metal_waste', 'paper_waste', 'glass_waste']  # Add your actual class names
)

# Build the waste classification model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))  # Adjusted for 6 classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the best weights during training
checkpoint = ModelCheckpoint('waste_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint]
)
