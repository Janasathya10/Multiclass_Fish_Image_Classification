import os, json, gc, random, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from data_loader import load_data
from cnn_model import create_cnn_model
from transfer_model import BACKBONES, create_transfer_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)

def save_metadata(model_name, img_size, class_indices, preprocess_name, path):
    meta = {
        "model_name": model_name,
        "img_size": list(img_size),
        "class_indices": class_indices,
        "preprocess": preprocess_name
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def train_model(model_name, batch_size=8, epochs_initial=8, epochs_finetune=5):
    if model_name == "CNN":
        train_gen, val_gen, _ = load_data(img_size=(224,224), batch_size=batch_size, augment=True)
        model = create_cnn_model((224,224,3), train_gen.num_classes)
        model.fit(train_gen, validation_data=val_gen, epochs=epochs_initial)
        model_path = "models/cnn_baseline.h5"
        model.save(model_path)
        meta_path = "models/cnn_baseline_metadata.json"
        save_metadata("CNN", (224,224), train_gen.class_indices, "rescale_0_1", meta_path)

    elif model_name in BACKBONES:
        cfg = BACKBONES[model_name]
        img_size = cfg["img_size"]
        preprocess_fn = cfg["preprocess"]
        train_gen, val_gen, _ = load_data(img_size=img_size, batch_size=batch_size, augment=True, preprocess_fn=preprocess_fn)
        model, base_model = create_transfer_model(model_name, (img_size[0], img_size[1], 3), train_gen.num_classes)

        class_weights = compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
        cw = {i: w for i, w in enumerate(class_weights)}

        ckpt_path = f"models/{model_name.lower()}_best.h5"
        callbacks = [
            ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max"),
            EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=epochs_initial, class_weight=cw, callbacks=callbacks)

        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_gen, validation_data=val_gen, epochs=epochs_finetune, class_weight=cw, callbacks=callbacks)

        meta_path = f"models/{model_name.lower()}_metadata.json"
        save_metadata(model_name, img_size, train_gen.class_indices, preprocess_fn.__name__, meta_path)

    # Save also as best_model for deployment
    model.save("models/best_model.h5")
    save_metadata(model_name, (224,224) if model_name=="CNN" else img_size,
                  train_gen.class_indices,
                  "rescale_0_1" if model_name=="CNN" else preprocess_fn.__name__,
                  "models/best_model_metadata.json")

    # Clear memory
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()
    if args.only:
        train_model(args.only)
    else:
        for model_name in ["CNN"] + list(BACKBONES.keys()):
            train_model(model_name)