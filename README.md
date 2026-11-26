# CampusQuery AI — End-to-End College Query Classification

This repository is prepared as a new GitHub project and includes a working Streamlit website, a trained 10-category model, the 5,000-row dataset, model evaluation reports, and the Colab training notebook.

## Included features

- Multi-page Streamlit interface
- Single-query intent classification
- Top category probabilities and confidence estimate
- Low-confidence review warning
- Batch CSV classification with downloadable predictions
- Dataset analytics and keyword search
- Model benchmark dashboard
- Session query history export
- Feedback collection and CSV export

## Ten intent categories

Admission, Fees, Exam, Hostel, Scholarship, Timetable, Library, Transport, Placement, Attendance.

## Trained model

The included deployable model was selected from six candidates:
- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression
- Linear SVM
- SGD Classifier
- Random Forest

Selection was based on group-aware cross-validation macro-F1. See `model_comparison_cv.csv`, `model_comparison_test.csv`, `reports/`, and `model_metadata.json`.

## Dataset transparency

The dataset contains 5,000 balanced rows: 180 original labelled examples retained from the original project and 4,820 synthetic template-generated examples. The current held-out benchmark is approximately 84.3% accuracy and 0.841 macro-F1 on this constructed dataset. These figures are **not verified real-world accuracy**. Collect fresh, independently written and manually labelled student queries for a realistic evaluation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a new empty GitHub repository.
2. Extract this ZIP on your computer.
3. Upload all files and folders **inside** the extracted folder to the root of the new repository.
4. Commit the files.
5. Open https://share.streamlit.io/ and choose **Create app**.
6. Select your new repository, branch `main`, and main file path `app.py`.
7. Deploy.

## Retraining

Open `College_Query_Classification_5000_Colab.ipynb` in Google Colab and run cells from top to bottom. Upload the existing dataset CSV when prompted. The notebook will regenerate the dataset, compare models, save the best model and create a Streamlit deployment bundle.

The classifier predicts intent only. It does not generate verified college-specific answers or check current university policies.
