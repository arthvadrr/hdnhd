"""
Ducky credits. Clean exportable model.
Augment in the dataset. Model has no aug layers.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG = 160
BS = 32

def make_ds(root, shuffle, aug=False):
    ds = keras.utils.image_dataset_from_directory(
        root, labels="inferred", label_mode="binary",
        image_size=(IMG, IMG), batch_size=BS, shuffle=shuffle
    )
    AUTOTUNE = tf.data.AUTOTUNE
    # normalize 0..1
    ds = ds.map(lambda x,y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=AUTOTUNE)
    if aug:
        a = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.25),
        ])
        ds = ds.map(lambda x,y: (a(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.cache().shuffle(1000).prefetch(AUTOTUNE) if shuffle else ds.cache().prefetch(AUTOTUNE)

train = make_ds("data/train", shuffle=True, aug=True)
val   = make_ds("data/val",   shuffle=False, aug=False)

def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inp = keras.Input((IMG, IMG, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out), base

model, base = build_model()
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

ckpt = keras.callbacks.ModelCheckpoint("out/best.keras", save_best_only=True,
                                       monitor="val_auc", mode="max")
model.fit(train, validation_data=val, epochs=10, callbacks=[ckpt])

# fine tune
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.fit(train, validation_data=val, epochs=5, callbacks=[ckpt])
print("✅ Training done. Saved out/best.keras")