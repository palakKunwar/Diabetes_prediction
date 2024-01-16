import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetes_.csv')
df.head()

df1 = df.astype(float)
df1.head()

import numpy as np
# Replace 0 values in important features with median value
df1['Glucose'] = df1['Glucose'].replace(0, np.median(df1['Glucose']))
df1['BloodPressure'] = df1['BloodPressure'].replace(0, np.median(df1['BloodPressure']))
df1['SkinThickness'] = df1['SkinThickness'].replace(0, np.median(df1['SkinThickness']))
df1['Insulin'] = df1['Insulin'].replace(0, np.median(df1['Insulin']))
df1['BMI'] = df1['BMI'].replace(0, np.median(df1['BMI']))

# Split features and target
X = df1.drop('Outcome', axis=1)
y = df1['Outcome']

# Split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
# Random Forest without scaling or resampling
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print("\nRandom Forest - Original Data")
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
# Random Forest with scaling
rfd_scaled = RandomForestClassifier(random_state=42)
rfd_scaled.fit(X_train_scaled, y_train)
predictions_scaled = rfd_scaled.predict(X_test_scaled)
print("\nRandom Forest - Scaled Data")
print("Classification Report:")
print(classification_report(y_test, predictions_scaled))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_scaled))

'''import pandas as pd
input_datad = {
    "Pregnancies": 0,  # Modify based on relevant data for pregnancies
    "Glucose": 137,  # Use existing glucose level
    "BloodPressure": 40,  # Use existing blood pressure
    "SkinThickness": 35,  # Modify based on relevant data for skin thickness
    "Insulin": 168,  # Modify based on relevant data for insulin
    "BMI": 43.1,  # Use existing BMI
    "DiabetesPedigreeFunction": 2.288,  # Modify based on relevant data for diabetes pedigree function
    "Age": 33  # Use existing age
}
# Convert input data to a DataFrame
input_df = pd.DataFrame([input_datad])
# Predict using the trained Random Forest model
predicted_outcome = rf.predict(input_df)

# Print the predicted outcome
print("Predicted Outcome for Diabetes:", predicted_outcome)
if predicted_outcome[0] == 1:
    print( "The model predicts a high risk of Diabetes.")
else:
    print("The model predicts a low risk of Diabetes.")'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
from PyQt5 import QtGui, QtCore
import pandas as pd
from PyQt5.QtWidgets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class DiabetesGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "Diabetes Prediction"
        self.top = 300
        self.left = 600
        self.width = 600
        self.height = 600
        self.iconName = "C:/Users/user/Documents/pythonprog/ML/MLGUI/assets/python.png"
        self.initUI()


    def initUI(self):

        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createInputFields()
        self.createPredictionButton()
        self.show()

        self.svmButton = self.createButton("Run",self.runModelPrediction,330, 240,60, 30)

        self.show()
    def createPredictionButton(self):
        # Create a button for prediction
        self.predict_button = self.createButton("Predict", self.runModelPrediction, 200, 400, 100, 30)

    def createButton(self, text, function, x, y, width, height):
        # Function to create buttons
        button = QPushButton(text, self)
        button.setGeometry(QtCore.QRect(x, y, width, height))
        button.clicked.connect(function)
        return button

    def createInputFields(self):

        self.preg_label = QLabel("Preg:", self)
        self.preg_label.setGeometry(QtCore.QRect(40, 50, 80, 30))

        self.preg_lineEdit = QLineEdit(self)
        self.preg_lineEdit.setGeometry(QtCore.QRect(150, 50, 100, 30))

        self.gluc_label = QLabel("Glucose:", self)
        self.gluc_label.setGeometry(QtCore.QRect(40, 90, 100, 30))

        self.gluc_lineEdit = QLineEdit(self)
        self.gluc_lineEdit.setGeometry(QtCore.QRect(150, 90, 100, 30))

        self.BP_label = QLabel("BP:", self)
        self.BP_label.setGeometry(QtCore.QRect(40, 130, 80, 30))

        self.BP_lineEdit = QLineEdit(self)
        self.BP_lineEdit.setGeometry(QtCore.QRect(150, 130, 100, 30))

        self.SkinT_label = QLabel("SkinThickness:", self)
        self.SkinT_label.setGeometry(QtCore.QRect(40, 170, 80, 30))

        self.SkinT_lineEdit = QLineEdit(self)
        self.SkinT_lineEdit.setGeometry(QtCore.QRect(150, 170, 100, 30))

        self.Insulin_label = QLabel("Insulin:", self)
        self.Insulin_label.setGeometry(QtCore.QRect(260, 50, 100, 30))

        self.Insulin_lineEdit = QLineEdit(self)
        self.Insulin_lineEdit.setGeometry(QtCore.QRect(370, 50, 100, 30))

        self.BMI_label = QLabel("BMI:", self)
        self.BMI_label.setGeometry(QtCore.QRect(260, 90, 80, 30))

        self.BMI_lineEdit = QLineEdit(self)
        self.BMI_lineEdit.setGeometry(QtCore.QRect(370, 90, 100, 30))

        self.Diabetes_label = QLabel("DPF", self)
        self.Diabetes_label.setGeometry(QtCore.QRect(260, 130, 80, 30))

        self.Diabetes_lineEdit = QLineEdit(self)
        self.Diabetes_lineEdit.setGeometry(QtCore.QRect(370, 130, 100, 30))

        self.age_label = QLabel("Age", self)
        self.age_label.setGeometry(QtCore.QRect(260, 170, 100, 30))

        self.age_lineEdit = QLineEdit(self)
        self.age_lineEdit.setGeometry(QtCore.QRect(370, 170, 100, 30))


        self.predict_button = self.createButton("Predict", self.runModelPrediction, 200, 400, 100, 30)


    def setDefault(self):
        # self.fileName = ""
        self.splitSize = 20
        self.regParam = 1.0
        self.kernelType = 'rbf'
        self.degree = 3
        self.tol = 0.001


    def drawBrowser(self):
        self.centralwidget = QWidget(self)
        self.csv_label = QLabel(self.centralwidget)
        self.csv_label.setGeometry(QtCore.QRect(10, 10, 80, 30))
        self.csv_label.setText("csv file: ")

        self.csv_lineEdit = QLineEdit(self)
        self.csv_lineEdit.setGeometry(QtCore.QRect(90,10,300,30))
        self.svmButton = self.createButton("Browse",self.getFileName,330, 50,60, 30)


    def runModelPrediction(self):
        # Assuming you have loaded your trained model (rf_model) beforehand
        # Get user input data from GUI fields
        input_data = {
            "Pregnancies": int(self.preg_lineEdit.text()),
            "Glucose": int(self.gluc_lineEdit.text()),
            "BloodPressure": int(self.BP_lineEdit.text()),
            "SkinThickness": int(self.SkinT_lineEdit.text()),
            "Insulin": int(self.Insulin_lineEdit.text()),
            "BMI": int(self.BMI_lineEdit.text()),
            "DiabetesPedigreeFunction": int(self.Diabetes_lineEdit.text()),
            "Age": int(self.age_lineEdit.text()),

            # Extract data from other fields similarly
            # ...
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Assuming rf_model is your trained Random Forest model
        predicted_result = self.predictOutcome(input_df,rf)

        # Display the prediction result in a message box
        if predicted_result[0] == 1:
            QMessageBox.about(self, "Prediction Result", "The model predicts a high risk of Diabetes.")
        else:
            QMessageBox.about(self, "Prediction Result", "The model predicts a low risk of Diabetes.")



    def predictOutcome(self, input_data,model):
        # Load your trained Random Forest model
        # Replace this with your actual trained model
        predicted_outcome = model.predict(input_data)
        return predicted_outcome


def Main():
    m = DiabetesGUI()
    m.show()
    return m

if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    mWin = DiabetesGUI()
    sys.exit(app.exec_())