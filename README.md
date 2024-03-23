# User Guide for Website: Arrhythmia Detection

## Getting Started

1. **Upload Data Files**: Begin by preparing your data files in ZIP or RAR format. Each data file intended for training should be placed at the root of the zip/rar file. Once ready, upload the compressed file using the provided form. Click on the 'Start Extraction' button to initiate the extraction process.
![image](https://github.com/GianEgaa/Arrhythmia-Detection/assets/90212308/166e9477-422e-46c7-845b-dfc333d1f7dd)

2. **Training Data**: After extraction, proceed with training the data by clicking on the 'Train Model' button. If there's an imbalance in the class distribution of the extracted features, a yellow warning and corresponding button will appear. You have the option to proceed with training without class balancing by clicking 'Train Model', or choose to balance the classes using SMOTE by selecting 'Continue with SMOTE'.
![image](https://github.com/GianEgaa/Arrhythmia-Detection/assets/90212308/c45e17fe-d3f7-4c84-a5f2-101f6d983e83)

3. **Visualize Feature Extraction**: To visualize the feature extraction results, you can view an example by clicking the green 'Plot' button next to the listed data files.
![image](https://github.com/GianEgaa/Arrhythmia-Detection/assets/90212308/059241b8-d003-44ad-a567-e6b5f9eeca2f)

## Post-Training

4. **Model Evaluation**: Once the training is completed, you'll be presented with the accuracy and classification report of the model.

5. **Prediction**: You can perform predictions using the trained model by uploading an .xlsx file via the prediction form. Simply hit 'Upload' and await the results, which will be displayed promptly in the provided section.
![image](https://github.com/GianEgaa/Arrhythmia-Detection/assets/90212308/c7436d36-1c4e-450f-a7c2-b93878dc949a)

## Additional Notes

- Ensure that your data files are properly formatted and organized within the compressed file.
- Pay attention to any warnings regarding class distribution imbalance and consider using SMOTE for better training results.
- Visualizing feature extraction results can provide insights into the data processing steps.
- After training, make use of the model evaluation metrics to assess its performance.
- For predictions, prepare your input data in .xlsx format and use the provided form for uploading.

Follow these steps to effectively utilize the Arrhythmia Detection website for data processing, model training, and prediction tasks.
