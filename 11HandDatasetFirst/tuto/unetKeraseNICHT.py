import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from .readmatanddraw.main_mitrichtigemkoordinatensystem import gatefeatures
import readimages
import pickle   #HIER REIN LOAD

def makefeature():


def dumbserial():
    pickle.load(()

features = gatefeatures()

def create_mask(image_shape, coords, class_label):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(coords, dtype=np.int32)
    cv2.fillPoly(mask, [points], class_label)
    return mask


def prepare_data(image_files, coord_files, image_shape, num_classes):
    images = []
    masks = []
    for img_file, coord_file in zip(image_files, coord_files):
        image = cv2.imread(img_file)
        coords = np.load(coord_file)  # Assumes coords are saved as numpy arrays
        for class_label in range(num_classes):
            class_coords = coords[class_label]
            mask = create_mask(image_shape, class_coords, class_label)
            images.append(image)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    return train_test_split(images, masks, test_size=0.2, random_state=42)


# Beispiel-Daten
image_files = ['image1.png', 'image2.png']
image_files = readimages
coord_files = ['coords1.npy', 'coords2.npy']
image_shape = (1280, 720, 3)
num_classes = 4

X_train, X_test, y_train, y_test = prepare_data(image_files, coord_files, image_shape, num_classes)


def unet_model(input_size=(256, 256, 3), num_classes=2):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = unet_model(input_size=image_shape, num_classes=num_classes)

mein_bild = np.zeros(256, 256, 3)



model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
