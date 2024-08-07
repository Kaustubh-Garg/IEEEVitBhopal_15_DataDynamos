import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image

DATASET_DIR = 'train2'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 20

# Preprocessing images
def preprocess_image(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

# Load and preprocess data
def load_data(dataset_dir):
    images = []
    labels = []
    label_map = {label: i for i, label in enumerate(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, label))}
    
    print("Label map:", label_map)
    
    for label_name in label_map:
        label_path = os.path.join(dataset_dir, label_name)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = preprocess_image(img_array)
            images.append(img_array)
            label = label_map[label_name]
            
            # Ensure label is within valid range
            if label >= NUM_CLASSES:
                raise ValueError(f"Label {label} is out of range. Expected [0, {NUM_CLASSES-1}]")
                
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)

# Model definition
def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    base_model.trainable = False

    model = Sequential([
        base_model,
        Dense(num_classes, activation='softmax')  # Output layer for classification
    ])
    return model

# Main function to run training
def main():
    train_dataset = load_data(DATASET_DIR)
    
    # Create and compile the model
    model = create_model((IMG_SIZE[0], IMG_SIZE[1], 3), NUM_CLASSES)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model and capture history
    history = model.fit(train_dataset, epochs=20)
    
    # Save the model
    model.save('model.keras')
    
    # Print the final training accuracy as a percentage
    final_accuracy = history.history['accuracy'][-1] * 100
    print(f"Final training accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
