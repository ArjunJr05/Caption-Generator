import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, concatenate
import numpy as np
import os
import pickle


class ImageCaptioningSystem:
    def __init__(self, model_path="caption_model.h5", tokenizer_path="tokenizer.pkl", max_length=34, vocab_size=5000):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.caption_model = None

    def load_image(self, image_path):
        try:
            img = load_img(image_path, target_size=(224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.vgg16.preprocess_input(img)
            return img
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

    def load_dataset(self, dataset_path):
        images, captions = [], []
        for filename in os.listdir(dataset_path):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(dataset_path, filename)
                caption_path = os.path.join(dataset_path, filename.split('.')[0] + '.txt')

                if os.path.exists(caption_path):
                    try:
                        with open(caption_path, 'r') as f:
                            caption = f.read().strip()
                        img = self.load_image(image_path)
                        images.append(img)
                        captions.append(caption)
                    except Exception as e:
                        st.error(f"Error loading {filename}: {e}")
        if not images:
            raise ValueError("No valid image-caption pairs found in the dataset.")
        return np.vstack(images), captions

    def create_tokenizer(self, captions):
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(captions)
        # Save tokenizer for later use
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def create_model(self):
        vgg_model = VGG16(weights="imagenet")
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

        inputs1 = Input(shape=(224, 224, 3))
        fe1 = vgg_model(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = concatenate([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        self.caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

    def prepare_sequences(self, captions):
        sequences = self.tokenizer.texts_to_sequences(captions)
        return pad_sequences(sequences, maxlen=self.max_length, padding='post')

    def train(self, images, captions, epochs=10, batch_size=32):
        sequences = self.prepare_sequences(captions)
        y = tf.keras.utils.to_categorical(sequences, num_classes=self.vocab_size)
        self.caption_model.fit([images, sequences], y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.caption_model.save(self.model_path)  # Save the trained model

    def load_pretrained_model(self):
        self.caption_model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def generate_caption(self, image_path):
        img = self.load_image(image_path)
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = self.caption_model.predict([img, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.tokenizer.index_word.get(yhat, None)
            if word is None or word == 'endseq':
                break
            in_text += ' ' + word
        return in_text.replace('startseq', '').replace('endseq', '').strip()


def main():
    st.title("Image Captioning System")
    captioning_system = ImageCaptioningSystem()

    # Sidebar setup
    st.sidebar.header("Model Training")
    dataset_path = st.sidebar.text_input("Dataset Path", "path_to_your_dataset")
    epochs = st.sidebar.slider("Epochs", 1, 50, 10)
    batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

    # Train model
    if st.sidebar.button("Train Model"):
        try:
            images, captions = captioning_system.load_dataset(dataset_path)
            captioning_system.create_tokenizer(captions)
            captioning_system.create_model()
            captioning_system.train(images, captions, epochs, batch_size)
            st.sidebar.success("Model trained and saved successfully!")
        except Exception as e:
            st.sidebar.error(f"Training Error: {e}")

    # Load pre-trained model
    if os.path.exists(captioning_system.model_path) and os.path.exists(captioning_system.tokenizer_path):
        captioning_system.load_pretrained_model()
        st.sidebar.success("Pre-trained model loaded!")

    # Generate captions
    st.header("Generate Caption")
    input_option = st.radio("Input Method", ["Upload Image", "Enter Path"])
    image_path = None

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file:
            image_path = os.path.join("temp_image.jpg")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    elif input_option == "Enter Path":
        image_path = st.text_input("Image Path")

    if st.button("Generate Caption"):
        if image_path and os.path.exists(image_path):
            try:
                caption = captioning_system.generate_caption(image_path)
                st.success(f"Caption: {caption}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Invalid image path or upload.")

if __name__ == "__main__":
    main()
