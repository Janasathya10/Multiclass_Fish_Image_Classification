import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_path="data", img_size=(224,224), batch_size=8, augment=False, preprocess_fn=None):
    """Load train, validation, and test data generators."""
    if preprocess_fn is None:
        preprocess_fn = lambda x: x / 255.0

    if augment:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "val")
    test_dir = os.path.join(base_path, "test")

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    test_gen = val_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    print("Train classes:", train_gen.class_indices)
    print("Number of classes:", train_gen.num_classes)

    return train_gen, val_gen, test_gen