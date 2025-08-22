import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

BACKBONES = {
    "VGG16": {"model": VGG16, "img_size": (224,224), "preprocess": tf.keras.applications.vgg16.preprocess_input},
    "ResNet50": {"model": ResNet50, "img_size": (224,224), "preprocess": tf.keras.applications.resnet50.preprocess_input},
    "MobileNet": {"model": MobileNet, "img_size": (224,224), "preprocess": tf.keras.applications.mobilenet.preprocess_input},
    "InceptionV3": {"model": InceptionV3, "img_size": (299,299), "preprocess": tf.keras.applications.inception_v3.preprocess_input},
    "EfficientNetB0": {"model": EfficientNetB0, "img_size": (224,224), "preprocess": tf.keras.applications.efficientnet.preprocess_input}
}

def create_transfer_model(model_name, input_shape, num_classes):
    base_model = BACKBONES[model_name]["model"](
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base_model