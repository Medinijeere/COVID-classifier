from fastai.vision.all import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the path to your dataset
path = Path('C:/Users/medin/Documents/dataset')

# Load the trained model
export_path = path/'xray_classifier.pkl'
learn_inf = load_learner(export_path)

# Define the prediction function
def predict_image(img_path):
    img = PILImage.create(img_path)
    pred, _, probs = learn_inf.predict(img)
    return pred, probs.max().item()

# Use the prediction function on a new image
img_path = path/'train'/'has covid'/'covid5.jpg'  # Update this path to your test image
prediction, probability = predict_image(img_path)
print(f"Prediction: {prediction}, Probability: {probability:.4f}")

# Check if there are validation samples available
valid_files = get_image_files(path/'validation')
print("Number of validation samples:", len(valid_files))

# Inspect the validation dataloader
if valid_files:
    try:
        # Prepare the validation dataset
        def label_func(x):
            if 'has covid' in str(x):
                return 'has covid'
            else:
                return 'doesn\'t have covid'

        valid_dl = ImageDataLoaders.from_name_func(
            path,
            valid_files,
            label_func=label_func,
            item_tfms=Resize(224)  # Resize images to the same size
        )

        print("Validation dataloader:", valid_dl)

        # Get predictions for the validation set
        try:
            preds, targs = learn_inf.get_preds(dl=valid_dl[0])
            print("Predictions shape:", preds.shape)
            print("Predictions:", preds)

            # Convert tensor predictions to numpy arrays
            preds = np.array(preds)
            targs = np.array(targs)

            # Compute the confusion matrix
            confusion_matrix = np.zeros((2, 2))
            for pred, targ in zip(preds, targs):
                pred_class = np.argmax(pred)
                confusion_matrix[pred_class, targ] += 1

            # Plot the confusion matrix
            sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['doesn\'t have covid', 'has covid'], yticklabels=['doesn\'t have covid', 'has covid'])
            plt.xlabel('True Label')
            plt.ylabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.show()
        except Exception as e:
            print("Error occurred while getting predictions:", e)
    except Exception as e:
        print("Error occurred while preparing the validation dataset:", e)
else:
    print("Warning: No validation samples available.")
