from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import json
import joblib
import os
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, f_classif

def train_and_save_models(df, label, model_save_path, ratio, num_features=5):
    # Separate features (X) and target variable (y)
    X = df.drop(label, axis=1)
    y = df[label]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    old_X_train = X_train
    old_y_train = y_train

    if not ratio:
        print('Training With SMOTE')
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("ini train asli =\n",old_y_train.value_counts(),"ini train smote =\n", y_train.value_counts())

    # # Feature Selection using Recursive Feature Elimination with Cross-Validation (RFECV)
    # rfecv = RFECV(estimator=RandomForestClassifier(), step=1, cv=5)
    # X_train_selected = rfecv.fit_transform(X_train, y_train)
    # X_test_selected = rfecv.transform(X_test)
        
    # print("Selected Features:", X.columns[rfecv.support_])
        
    # Feature Selection using SelectKBest with ANOVA F-statistic
    selector = SelectKBest(f_classif, k=num_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get the selected feature indices
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_feature_indices]
    print("Selected Features:", selected_features)

    # Save selected features to a JSON file
    selected_features_data = {
        'selected_features': selected_features.tolist()
    }

    # Train K-Nearest Neighbors (KNN) model
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_selected, y_train)
    knn_predictions = knn_model.predict(X_test_selected)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    knn_classification_report = classification_report(y_test, knn_predictions)
    
    # Train Support Vector Classifier (SVC) model
    svc_model = SVC()
    svc_model.fit(X_train_selected, y_train)
    svc_predictions = svc_model.predict(X_test_selected)
    svc_accuracy = accuracy_score(y_test, svc_predictions)
    svc_classification_report = classification_report(y_test, svc_predictions)

    # Train Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_selected, y_train)
    rf_predictions = rf_model.predict(X_test_selected)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_classification_report = classification_report(y_test, rf_predictions)

    classification_reports_data = {
        'knn_classification_report': knn_classification_report,
        'svc_classification_report': svc_classification_report,
        'rf_classification_report': rf_classification_report
    }
    os.makedirs(model_save_path, exist_ok=True)

    selected_features_json_filename = os.path.join(model_save_path, 'selected_features.json')
    with open(selected_features_json_filename, 'w') as outfile:
        json.dump(selected_features_data, outfile, indent=4)

    classification_reports_json = json.dumps(classification_reports_data, indent=4)

    json_filename = os.path.join(model_save_path, 'classification_report.json')
    # Writing to sample.json
    with open(json_filename, "w") as outfile:
        outfile.write(classification_reports_json)

    accuracy_data = {
        'knn_accuracy': knn_accuracy,
        'svc_accuracy': svc_accuracy,
        'rf_accuracy': rf_accuracy
    }

    combined_data = {
        'accuracy_data': accuracy_data,
        'classification_reports': classification_reports_data
    }

    # Save the models
    joblib.dump(knn_model, os.path.join(model_save_path, 'KNN.joblib'))
    joblib.dump(svc_model, os.path.join(model_save_path, 'SVC.joblib'))
    joblib.dump(rf_model, os.path.join(model_save_path, 'RF.joblib'))

    print("model_save :", model_save_path)
    print("json_file :", json_filename)

    return accuracy_data, classification_reports_data

