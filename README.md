# Image Classification Project

Welcome to the Image Classification Project! This project leverages deep learning and a pre-trained VGG16 model to classify images. The project is implemented using TensorFlow, Keras, and several other Python libraries, and it includes a web application built with Streamlit for easy interaction with the model.

## Project Structure

The project consists of the following files and directories:

- `notebooks/`
  - `image_classification.ipynb`: Jupyter notebook containing code for loading, preprocessing, and predicting images using the VGG16 model.
- `app.py`: Streamlit application code for uploading images and displaying predictions.
- `model/`
  - `model.joblib`: Serialized model saved using joblib.
- `requirements.txt`: List of required Python packages.

## Requirements

To run this project, you will need the following Python packages:

- tensorflow
- keras
- joblib
- numpy
- image
- streamlit
- requests

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

1. Open the Jupyter notebook `image_classification.ipynb`.
2. Run the cells to load the VGG16 model, preprocess images, and make predictions.

### Running the Streamlit App

1. Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

2. This will start a local web server. Open your web browser and navigate to the provided URL (usually `http://localhost:8501`).

3. Upload an image using the file uploader. The app will preprocess the image, pass it through the pre-trained VGG16 model, and display the predicted label.

## Model Details

- **Model Architecture**: VGG16 (pre-trained on ImageNet dataset)
- **Libraries Used**:
  - TensorFlow and Keras: For deep learning model and preprocessing
  - Joblib: For model serialization
  - Numpy: For numerical operations
  - Image: For image handling
  - Streamlit: For web application
  - Requests: For accessing model labels in JSON format

## How It Works

1. **Image Loading and Preprocessing**:
   - Images are loaded using the `image` library.
   - Preprocessing includes resizing the image to the input size expected by VGG16 and normalizing pixel values.

2. **Model Prediction**:
   - The preprocessed image is fed into the VGG16 model.
   - The model outputs probabilities for each class label.
   - The class with the highest probability is selected as the prediction.

3. **Streamlit App**:
   - The app allows users to upload images.
   - The uploaded image is processed and passed to the model for prediction.
   - The predicted label is displayed on the web interface.

## Example

Here is a step-by-step example of using the Streamlit app:

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Upload an image using the file uploader in the web interface.
3. The app will display the predicted label along with the confidence score.

## Conclusion

This project demonstrates how to use a pre-trained deep learning model for image classification and how to build a simple web application for model inference. Feel free to explore the code and extend the project as needed.

If you have any questions or encounter any issues, please open an issue on the project's GitHub repository.

Happy coding!

---

**Note**: Ensure you have the correct versions of the libraries mentioned in `requirements.txt` to avoid any compatibility issues.
