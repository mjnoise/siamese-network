# Visual Search
Allow customers to upload photos and find matching items in your catalog.
Given a query image the model will find matching items in the catalog. This is a common use-case in e-commerce, where customers want to find products similar to an image they have.

## Dataset

**clothing-dataset-small** collection, originally found [here](https://github.com/alexeygrigorev/clothing-dataset-small). For this assignment, a specific subset has been curated and organized into training and validation sets, focusing on the following distinct clothing categories: `dress`, `hat`, `longsleeve`, `pants`, `shoes`, `shorts`, and `t-shirt`.

## Model
This use-case requires a Siamese Network which can descern similar items in a catalog. To achieve this, we will first train a classifier to learn good feature representations of the images. Then we will use the trained backbone of the classifier to train a Siamese Network using triplet loss.

## Training
The training process is divided into two main stages:
1. **Classifier Training**: A convolutional neural network (CNN) is trained on the training dataset to classify images into their respective clothing categories. This step helps the model learn useful feature representations of the images.
2. **Siamese Network Training**: Using the trained backbone from the classifier, a Siamese Network is trained with triplet loss to learn embeddings that capture the similarity between items. This enables the model to find matching items in the catalog based on visual similarity. The training process is executed in the `nb.ipynb` notebook.

## Evaluation
After training, the model's performance is evaluated by selecting a random query image from the validation set and retrieving the most similar items from the catalog. The retrieved items are displayed along with their class labels to demonstrate the effectiveness of the visual search functionality.

## Conclusion
This project demonstrates how to implement a visual search system using a Siamese Network. By training a classifier to learn feature representations and then fine-tuning a Siamese Network with triplet loss, we can effectively find and retrieve similar items from a catalog based on a query image. This approach is particularly useful in e-commerce applications, enhancing the customer experience by allowing them to easily find products that match their preferences.

## Example result

Below is an example visualization of the retrieval result produced by the notebook using images from `nb_images/`.

**Query image (`nb_images/query_image.png`)**

![Query image](nb_images/query_image.png)

**Top match / positive (`nb_images/positive.png`)**

![Positive match](nb_images/positive.png)

In this example, the model embeds the query image and retrieves the nearest neighbor in the catalog embedding space. The retrieved *positive* image is the closest match, indicating high visual similarity (e.g., same clothing type and similar appearance).