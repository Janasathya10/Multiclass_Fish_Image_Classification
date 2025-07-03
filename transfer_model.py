from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNet, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import load_data
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

MODELS = {
    'VGG16': VGG16,
    'ResNet50': ResNet50,
    'InceptionV3': InceptionV3,
    'MobileNet': MobileNet,
    'EfficientNetB0': EfficientNetB0
}

train_gen, val_gen, _ = load_data()
input_shape = (224, 224, 3)
num_classes = train_gen.num_classes
labels = train_gen.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

for name, base_fn in MODELS.items():
    base = base_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ckpt_path = f'models/{name.lower()}.h5'
    checkpoint = ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, verbose=1)

    print(f"Training {name}...")
    model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[checkpoint, early_stop, reduce_lr], class_weight=class_weights_dict)