
🌸 Iris Flower Classification using Machine Learning
This project uses the classic Iris dataset to classify iris flowers into one of three species based on their sepal and petal measurements. It trains and compares three machine learning models — Decision Tree, Random Forest, and Support Vector Machine (SVM) — using scikit-learn.

📁 Files and Structure
Iris.csv → Dataset (input)
train_models.py → Trains the 3 ML models
predict_input.py → Accepts user input and predicts the species
README.md → This file

📊 Features
✅ Trains and evaluates:

🌳 Decision Tree Classifier

🌲 Random Forest Classifier

🌀 Support Vector Machine (SVM)

✅ Uses all input features:

SepalLengthCm

SepalWidthCm

PetalLengthCm

PetalWidthCm

✅ Encodes species into numerical labels
✅ Allows user to input custom measurements to predict the flower species
✅ Prints accuracy of each model and final prediction

🚀 Getting Started
1. Install Dependencies
Make sure you have Python 3.x installed, then run:

pip install pandas scikit-learn matplotlib seaborn

2. Train the Models
Run the script to train Decision Tree, Random Forest, and SVM:

python train_models.py

You’ll see output like:

Accuracy of Decision Tree: 1.0
Accuracy of Random Forest: 1.0
Accuracy of SVM: 1.0

3. Predict Species from User Input
After training, run:

python predict_input.py

You’ll be prompted to enter sepal and petal lengths and widths:

Enter the sepal length: 5.1
Enter the sepal width: 3.5
Enter the petal length: 1.4
Enter the petal width: 0.2
Predicted species: Iris-setosa

📌 Notes
The SVM model was trained using feature names from a pandas DataFrame. If you pass a raw list (like [5.1, 3.5, 1.4, 0.2]), you may see a warning. To avoid this, the prediction input is wrapped as a pandas DataFrame with column names.

All models achieve 100% accuracy because the Iris dataset is small, clean, and linearly separable — perfect for beginner ML tasks.

📚 Dataset Source
UCI Machine Learning Repository – Iris Dataset
https://archive.ics.uci.edu/ml/datasets/iris

