import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2





st.set_page_config(page_title="CNN Dashboard", layout="wide")


# Sidebar
menu = st.sidebar.radio("Go to", [
    "🏠 Home","📊 Train","📈 Graphs","🧪 Test",
    "🔍 Prediction","🎥 Webcam"
])
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# Load model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("model.h5")
    except:
        return None

model = load_model()

@st.cache_resource
def load_digit_model():
    try:
        return tf.keras.models.load_model("digit_model.h5")
    except:
        return None

digit_model = load_digit_model()

# ---------------- HOME ----------------
if menu == "🏠 Home":
    st.title("🚀 CNN Dashboard")
    st.metric("🎯 Expected Accuracy", "85% - 92%")

# ---------------- TRAIN ----------------
elif menu == "📊 Train":
    st.title("📊 Train Model")

    epochs = st.slider("Epochs", 5, 30, 20)

    if st.button("Start Training"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(x_train)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(10,activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = int((epoch + 1) / epochs * 100)
                progress_bar.progress(progress)

                status_text.text(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Acc: {logs['accuracy']:.4f} | "
                    f"Val Acc: {logs['val_accuracy']:.4f}"
                )

        with st.spinner("Training..."):
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=64),
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=[StreamlitCallback()]
            )

        model.save("model.h5")
        np.save("history.npy", history.history)

        st.success("✅ Model Trained Successfully!")
        st.success(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
        st.success(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# ---------------- GRAPHS ----------------
elif menu == "📈 Graphs":
    st.title("📈 Accuracy & Loss Graphs")

    try:
        history = np.load("history.npy", allow_pickle=True).item()

        fig1 = plt.figure()
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title("Accuracy vs Epochs")
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title("Loss vs Epochs")
        st.pyplot(fig2)

    except:
        st.warning("Train model first!")

# ---------------- TEST ----------------
elif menu == "🧪 Test":
    st.title("🧪 Model Evaluation")

    if model is None:
        st.warning("Train model first!")
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test / 255.0

        loss, acc = model.evaluate(x_test, y_test)

        st.metric("📊 Test Accuracy", f"{acc*100:.2f}%")
        st.metric("📉 Loss", f"{loss:.4f}")

        pred = model.predict(x_test)
        y_pred = np.argmax(pred, axis=1)

        cm = confusion_matrix(y_test, y_pred)

        fig = plt.figure(figsize=(8,6))
        sns.heatmap(cm, cmap='Blues')
        plt.title("Confusion Matrix")
        st.pyplot(fig)

# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
elif menu == "🔍 Prediction":
    st.title("🔍 Smart Image Prediction")

    if model is None:
        st.warning("Train model first!")
    else:
        files = st.file_uploader("Upload Images", accept_multiple_files=True)

        if files:
            count = 0

            for file in files:
                img = Image.open(file)
                st.image(img, width=150)

                img_array = np.array(img)

                # 🔥 BETTER DIGIT DETECTION
                is_grayscale = len(img_array.shape) == 2 or img.mode == 'L'

                if is_grayscale:
                    # ✅ DIGIT MODEL
                    if digit_model is None:
                        st.warning("⚠️ digit_model.h5 not found!")
                        continue

                    img_resized = img.convert('L').resize((28,28))
                    img_resized = np.array(img_resized) / 255.0

                    # invert colors
                    img_resized = 1 - img_resized

                    img_resized = img_resized.reshape(1,28,28,1)

                    pred = digit_model.predict(img_resized)
                    result = np.argmax(pred)
                    confidence = np.max(pred) * 100

                    st.success(f"✍️ Digit: {result} ({confidence:.2f}%)")

                else:
                    # ✅ CIFAR MODEL
                    img_resized = img.resize((32,32))
                    img_resized = np.array(img_resized) / 255.0
                    img_resized = np.expand_dims(img_resized, axis=0)

                    pred = model.predict(img_resized)
                    result = classes[np.argmax(pred)]
                    confidence = np.max(pred) * 100

                    st.success(f"📷 {result} ({confidence:.2f}%)")

                count += 1

            st.info(f"📂 Total Images Predicted: {count}")
# ---------------- WEBCAM ----------------
elif menu == "🎥 Webcam":
    st.title("🎥 Real-Time Detection")

    run = st.checkbox("Start Webcam")

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (32,32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        label = classes[np.argmax(pred)]

        cv2.putText(frame, label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        st.image(frame, channels="BGR")

    cap.release()