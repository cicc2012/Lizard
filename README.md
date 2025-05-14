# Tutorial: Training a Model to Distinguish Individual Lizards

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Overview: Detection and Identification Workflow](#overview-detection-and-identification-workflow)
3. [Step 1: Data Preparation](#step-1-data-preparation)
   - [Bounding Box Usage for Training](#bounding-box-usage-for-training)
   - [Cropping Lizard Regions](#cropping-lizard-regions)
   - [Segmentation for Background Removal](#segmentation-for-background-removal)
     - [Best Practices for Segmentation](#best-practices-for-segmentation)
     - [Workflow for Segmentation Training Data](#workflow-for-segmentation-training-data)
     - [Using Pre-trained Segmentation Models](#using-pre-trained-segmentation-models)
   - [Data Augmentation](#data-augmentation)
4. [Step 2: Feature Extraction](#step-2-feature-extraction)
   - [Feature Extraction Program](#feature-extraction-program)
5. [Step 3: Model Training](#step-3-model-training)
   - [Model Architecture](#model-architecture)
   - [Training the Model](#training-the-model)
6. [Step 4: Model Evaluation](#step-4-model-evaluation)
7. [Step 5: Deployment](#step-5-deployment)
   - [Full Workflow: Detection and Identification](#full-workflow-detection-and-identification)
8. [Pre-trained Models](#pre-trained-models)
   - [Object Detection Models](#object-detection-models)
   - [Feature Extraction Models](#feature-extraction-models)
   - [Segmentation Models](#segmentation-models)
9. [Notes](#notes)

---

## Prerequisites

1. **Python Environment**:
   - Install Python 3.8 or later.
   - Recommended: Use a virtual environment.

2. **Dependencies**:
   Install the following libraries:
   ```bash
   pip install tensorflow opencv-python scikit-learn matplotlib
   ```

3. **Dataset**:
   - Collect high-resolution images of lizards.
   - Organize them into folders named after individual lizards:
     ```
     dataset/
       ├── lizard_1/
       │   ├── img1.jpg
       │   ├── img2.jpg
       ├── lizard_2/
       │   ├── img1.jpg
       │   ├── img2.jpg
     ```

---

## Overview: Detection and Identification Workflow

To distinguish individual lizards, we follow a two-step workflow:

### Step 1: Detecting the Presence of a Lizard
Before identifying an individual, the system must first detect whether a lizard is present in the image. This involves:
1. **Object Detection**:
   - Use models like YOLO, Faster R-CNN, or SSD to locate lizards in the image.
   - These models output bounding boxes around detected lizards.

2. **Lizard Localization**:
   - Crop the detected region (bounding box) to focus on the lizard.
   - This ensures the identification step works only on the relevant part of the image.

### Step 2: Identifying the Individual
Once a lizard is detected, the cropped image is passed to a classification model to identify the individual. This involves:
1. **Feature Extraction**:
   - Use a pre-trained model (e.g., MobileNetV2 or ResNet) to extract features from the cropped image.
   - These features represent unique patterns like scales, coloration, or body shape.

2. **Classification**:
   - The extracted features are fed into a classifier trained to distinguish between individuals.
   - The classifier outputs the identifier (e.g., "Lizard_1", "Lizard_2").

---

## Step 1: Data Preparation

### Bounding Box Usage for Training

Bounding boxes are essential for localizing lizards in images. For training, we use the following steps:

1. **Annotate Images**:
   - Use tools like LabelImg to annotate images with bounding boxes around lizards.
   - Save annotations in formats like XML (Pascal VOC) or JSON (COCO).

2. **Crop Images**:
   - Use the bounding box coordinates to crop the lizard region from the image.
   - This ensures the model focuses only on the lizard and not the background.

3. **Optional: Background Removal**:
   - Use segmentation techniques to remove the background and retain only the lizard.

---

### Cropping Lizard Regions

Here’s a Python function to crop lizard regions based on bounding box annotations:

```python
import cv2
import json

def crop_lizard(image_path, annotation_path):
    """
    Crops the lizard region from an image using bounding box annotations.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file (e.g., JSON or XML).

    Returns:
        numpy.ndarray: Cropped lizard image.
    """
    # Load image
    image = cv2.imread(image_path)

    # Load bounding box annotation (example assumes JSON format)
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    bbox = annotation['bbox']  # Example: [x1, y1, width, height]

    # Crop the bounding box region
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

# Example usage
cropped_image = crop_lizard('lizard.jpg', 'lizard_annotation.json')
cv2.imwrite('cropped_lizard.jpg', cropped_image)
```

---

### Segmentation for Background Removal

Background removal can improve recognition accuracy by isolating the lizard from the background. Below are best practices and a workflow for segmentation:

#### Best Practices for Segmentation

1. **Manual Annotation**:
   - Use tools like [LabelMe](https://github.com/wkentaro/labelme) or [CVAT](https://github.com/openvinotoolkit/cvat) to manually annotate segmentation masks.
   - Save masks in formats like PNG or JSON, where each pixel value corresponds to a class (e.g., 1 for lizard, 0 for background).

2. **Semi-Automatic Annotation**:
   - Use tools like [GrabCut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html) or [Deep Extreme Cut (DEXTR)](https://github.com/scaelles/DEXTR) to generate initial masks, then refine them manually.

3. **Pre-trained Segmentation Models**:
   - Use models like U-Net, DeepLab, or Mask R-CNN pre-trained on datasets like COCO or Pascal VOC.
   - Fine-tune these models on your specific dataset for better accuracy.

4. **Validation of Segmentation Masks**:
   - Visualize masks overlaid on the original images to ensure alignment.
   - Use metrics like Intersection over Union (IoU) or Dice Coefficient to evaluate mask quality.

5. **Consistency in Preprocessing**:
   - Resize both images and masks to the same dimensions.
   - Normalize pixel values for both images and masks.

---

#### Workflow for Segmentation Training Data

Here’s a Python function to preprocess images and their corresponding segmentation masks:

```python
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_segmentation_data(image_path, mask_path, target_size=(128, 128)):
    """
    Preprocesses an image and its corresponding segmentation mask.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the segmentation mask.
        target_size (tuple): Target size for resizing (height, width).

    Returns:
        tuple: Preprocessed image and mask.
    """
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize image and mask
    image = cv2.resize(image, target_size)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Normalize image
    image = image / 255.0

    # Convert mask to categorical (one-hot encoding)
    mask = to_categorical(mask, num_classes=2)  # Assuming 2 classes: lizard and background

    return image, mask

# Example usage
image, mask = preprocess_segmentation_data('lizard.jpg', 'lizard_mask.png')
```

---

#### Using Pre-trained Segmentation Models

To automate background removal, use pre-trained segmentation models like DeepLab or U-Net. Below is an example workflow:

```python
def remove_background_with_model(image, segmentation_model):
    """
    Removes the background from an image using a pre-trained segmentation model.

    Args:
        image (numpy.ndarray): Input image.
        segmentation_model: Pre-trained segmentation model.

    Returns:
        numpy.ndarray: Image with background removed.
    """
    # Resize image to model's input size
    input_size = (segmentation_model.input.shape[1], segmentation_model.input.shape[2])
    resized_image = cv2.resize(image, input_size)
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

    # Predict segmentation mask
    mask = segmentation_model.predict(input_image)[0]
    mask = (mask > 0.5).astype(np.uint8)  # Threshold mask

    # Resize mask back to original image size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Example usage
segmentation_model = ...  # Load your pre-trained segmentation model
segmented_image = remove_background_with_model(cv2.imread('lizard.jpg'), segmentation_model)
cv2.imwrite('segmented_lizard.jpg', segmented_image)
```

---

### Data Augmentation

Data augmentation is crucial for improving model robustness. Here are some augmentation techniques:

1. **Geometric Transformations**:
   - Rotate, flip, or scale the image to simulate different perspectives.

2. **Color Adjustments**:
   - Change brightness, contrast, or saturation to account for lighting variations.

3. **Background Variations**:
   - Overlay the lizard on different backgrounds to make the model robust to background changes.

Example using `ImageDataGenerator`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example usage
augmented_images = datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
```

---

## Step 2: Feature Extraction

### Feature Extraction Program

Here’s a Python program to extract these features:

```python
import cv2
import numpy as np
from skimage.feature import hog

def extract_features(image):
    """
    Extracts features from a lizard image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        dict: Extracted features including HOG, color histogram, and keypoints.
    """
    features = {}

    # Resize image
    image = cv2.resize(image, (128, 128))

    # 1. Extract HOG (Histogram of Oriented Gradients) for scale patterns
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, multichannel=False)
    features['hog'] = hog_features

    # 2. Extract color histogram for coloration and markings
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    features['color_histogram'] = cv2.normalize(hist, hist).flatten()

    # 3. Extract keypoints for geometric features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    features['keypoints'] = descriptors if descriptors is not None else np.array([])

    return features
```

---

## Step 3: Model Training

### Model Architecture

Use a pre-trained model (e.g., MobileNetV2) for feature extraction and fine-tune it for individual recognition.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Load pre-trained MobileNetV2
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')

# Add custom layers for classification
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### Training the Model

Split the dataset into training and validation sets, then train the model.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load dataset
data_dir = 'dataset/'
images, labels = [], []
label_map = {}

for label, lizard_name in enumerate(os.listdir(data_dir)):
    label_map[label] = lizard_name
    lizard_dir = os.path.join(data_dir, lizard_name)
    for img_file in os.listdir(lizard_dir):
        img_path = os.path.join(lizard_dir, img_file)
        images.append(preprocess_image(img_path))
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Split data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)
```

---

## Step 4: Model Evaluation

Evaluate the model's performance using accuracy and loss metrics.

```python
import matplotlib.pyplot as plt

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

---

## Step 5: Deployment

### Full Workflow: Detection and Identification

Here’s how to integrate detection and identification into a single function:

```python
def recognize_individual(image_path, detection_model, recognition_model, label_map):
    """
    Detects lizards in an image, extracts features, and identifies individuals.

    Args:
        image_path (str): Path to the input image.
        detection_model: Pre-trained object detection model (e.g., YOLO).
        recognition_model: Pre-trained individual recognition model.
        label_map (dict): Mapping of class indices to individual names.

    Returns:
        List of detected individuals with their bounding boxes and features.
    """
    import cv2
    import numpy as np

    # Step 1: Detect lizards
    image = cv2.imread(image_path)
    bounding_boxes = detection_model.detect(image)  # Replace with actual detection function

    results = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        cropped_lizard = image[y1:y2, x1:x2]

        # Step 2: Extract features
        features = extract_features(cropped_lizard)

        # Step 3: Preprocess and identify individual
        cropped_lizard = preprocess_image(cropped_lizard)
        cropped_lizard = np.expand_dims(cropped_lizard, axis=0)  # Add batch dimension
        prediction = recognition_model.predict(cropped_lizard)
        predicted_label = label_map[np.argmax(prediction)]

        # Append result
        results.append({
            "bounding_box": (x1, y1, x2, y2),
            "individual": predicted_label,
            "features": features
        })

    return results

# Example usage
detection_model = ...  # Load your object detection model (e.g., YOLO)
results = recognize_individual('test_image.jpg', detection_model, model, label_map)
for result in results:
    print(f"Detected {result['individual']} at {result['bounding_box']}")
    print(f"Features: {result['features']}")
```

---

## Pre-trained Models

### Object Detection Models

- **YOLO (You Only Look Once)**:
  - A fast and accurate object detection model.
  - Pre-trained weights and configuration files can be downloaded from the [YOLO GitHub repository](https://github.com/AlexeyAB/darknet).
  - Example usage:
    ```bash
    wget https://pjreddie.com/media/files/yolov3.weights
    ```

- **Faster R-CNN**:
  - A highly accurate object detection model.
  - Available in the [TensorFlow Model Zoo](https://github.com/tensorflow/models/tree/master/research/object_detection).

- **SSD (Single Shot Multibox Detector)**:
  - A lightweight object detection model.
  - Pre-trained models are available in the [TensorFlow Model Zoo](https://github.com/tensorflow/models/tree/master/research/object_detection).

---

### Feature Extraction Models

- **MobileNetV2**:
  - A lightweight model for feature extraction.
  - Available in TensorFlow and Keras:
    ```python
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(weights='imagenet', include_top=False)
    ```

- **ResNet**:
  - A powerful model for feature extraction.
  - Available in TensorFlow and Keras:
    ```python
    from tensorflow.keras.applications import ResNet50
    model = ResNet50(weights='imagenet', include_top=False)
    ```

---

### Segmentation Models

- **DeepLab**:
  - A state-of-the-art segmentation model.
  - Pre-trained models are available in the [TensorFlow Model Zoo](https://github.com/tensorflow/models/tree/master/research/deeplab).

- **U-Net**:
  - A popular model for biomedical and general image segmentation.
  - Pre-trained weights can be found in repositories like [U-Net GitHub](https://github.com/zhixuhao/unet).

---

## Notes

1. **Improving Accuracy**:
   - Use more data for training.
   - Fine-tune the base model by unfreezing some layers.

2. **Improving Feature Extraction**:
   - Experiment with different feature extraction techniques (e.g., deep learning-based embeddings).
   - Use domain-specific knowledge to refine feature selection.

3. **Feature Combination**:
   - Combine extracted features (e.g., HOG + color histogram) for better classification accuracy.

4. **Advanced Techniques**:
   - Use Siamese Networks for one-shot learning if the dataset is small.
   - Explore keypoint detection for unique patterns.

5. **Hardware**:
   - Use a GPU for faster training and feature extraction.

This concludes the tutorial. You now have a complete workflow to detect, crop, remove background, augment, and identify individual lizards!
