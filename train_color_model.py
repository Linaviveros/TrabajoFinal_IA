# Pico_placa_ia/train_color_model.py
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carpeta donde está este archivo: .../Pico_placa_ia/
ROOT_DIR = Path(__file__).resolve().parent

# Dataset: .../Pico_placa_ia/dataset/placas_colombia/car_color_dataset/train
DATASET_DIR = ROOT_DIR / "dataset" / "placas_colombia" / "car_color_dataset" / "train"

# Carpeta Models: .../Pico_placa_ia/Models  (la que ves en la captura)
MODELS_DIR = ROOT_DIR / "Models"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

def main():
    print(f"Usando dataset en: {DATASET_DIR}")
    print(f"Guardando modelo en: {MODELS_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=123,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        seed=123,
    )

    class_names = train_ds.class_names
    print("Clases detectadas (EN):", class_names)

    normalization_layer = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    num_classes = len(class_names)

    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
    )

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "car_color_model.h5"
    model.save(model_path)
    print(f"✅ Modelo guardado en: {model_path}")

if __name__ == "__main__":
    main()
