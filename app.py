import os
import torch
# import torch.nn as nn
import numpy as np
import cv2
# from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas


import torch.nn as nn

import torch.optim as optim

def train_model(model, images, labels, device, epochs=5, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for img, label in zip(images, labels):
            img = preprocess_image(img).to(device)
            label = torch.tensor([label], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(images):.4f}')

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # Use 'fc' to match the saved model

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)  # Use 'fc' to match the saved model

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path, device):
    model = LogisticRegressionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image):
    """
    Preprocess the image to match the input requirements of the model.
    """
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert colors: Canvas is white on black, MNIST is black on white
    image = cv2.bitwise_not(image)
    # Normalize to [0,1]
    image = image / 255.0
    # Add channel and batch dimensions
    image = image.reshape(1, 1, 28, 28)
    # Convert to tensor
    image = torch.tensor(image, dtype=torch.float32)
    return image


def main():
    st.title('Digit Recognizer')
    st.markdown('Draw a digit in the box below and click **Predict** to see the result.')

    # Sidebar for model path and mode selection
    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.text_input("Model Path", "best_model.pth")
    mode = st.sidebar.selectbox("Mode", ["Predict", "Train"])

    # Check if the model file exists
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file not found at `{model_path}`. Please provide a valid path.")
        st.stop()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = load_model(model_path, device)

    # Canvas settings
    SIZE = 192  # Canvas size
    canvas_result = st_canvas(
        fill_color='#FFFFFF',  # White background
        stroke_width=20,
        stroke_color='#000000',  # Black drawing
        background_color='#FFFFFF',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key='canvas'
    )

    if canvas_result.image_data is not None:
        # Convert the image to uint8
        img = canvas_result.image_data.astype('uint8')
        # Display the image on the app
        st.image(img, caption='Your Drawing', use_column_width=True)

    if mode == "Predict":
        if st.button('Predict'):
            if canvas_result.image_data is None:
                st.warning("Please draw a digit before predicting.")
            else:
                # Preprocess the image
                processed_image = preprocess_image(img)
                processed_image = processed_image.to(device)

                # Make prediction
                with torch.no_grad():
                    outputs = model(processed_image)
                    probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                    predicted_digit = np.argmax(probabilities)

                # Display the result
                st.write(f'### Predicted Digit: {predicted_digit}')
                st.bar_chart(probabilities)
    elif mode == "Train":
        label = st.sidebar.number_input("Label", min_value=0, max_value=9, step=1)
        if st.button('Train'):
            if canvas_result.image_data is None:
                st.warning("Please draw a digit before training.")
            else:
                train_model(model, [img], [label], device)
                torch.save(model.state_dict(), model_path)
                st.success("Model trained and saved successfully.")

if __name__ == "__main__":
    main()