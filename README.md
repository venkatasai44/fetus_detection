# Fetal Abnormality Detection using MobileNetV2 Transfer Learning

This project focuses on detecting **fetal abnormalities** from ultrasound images using **MobileNetV2** and **Transfer Learning** in **TensorFlow 2.15**.  
The model classifies images into **normal** and **abnormal** categories based on training data.

---

##  Environment Setup (Using pip)

This project uses **pip** to manage its dependencies.  
It is recommended to create a **virtual environment** before installing the required packages.

---

###  Prerequisites

1. Ensure you have **Python 3.10+** installed.  
2. Clone this repository and make sure the following files are present in your project folder:
   - `train_model.py`
   - `requirements.txt`
   - `README.md`

  ### And download the training and testing datasets from kaggle or drive,where instructions are provided to download in dataset.txt file ###
  ** place the classification and classification_test folders in the project folder **
   classification folder for dataset
   classification_test to test on unseen data
---

###  Create a Virtual Environment

Open your **terminal** or **command prompt** in the project directory and run the following commands:

```bash
# Create virtual environment
python -m venv fetal_env

# For Windows
fetal_env\Scripts\activate

# For macOS/Linux
source fetal_env/bin/activate

# after activation enter this command in the terminal 

pip install -r requirements.txt

# now start training the model so you enter this command
python train_model.py

# During training you will see

Training and validation accuracy/loss values

Precision–Recall curve plots

Classification report with precision, recall, and F1-score for each class

#After successful training:

Precision, Recall, F1-Score for each class will be displayed in the terminal.

Precision–Recall Curve and Confusion Matrix will be displayed as images or saved automatically.

The trained model will be saved inside the saved_model/ directory.

###  Reproducibility

To ensure reproducible results, random seeds are fixed in the training script using:
```python
np.random.seed(42)
tf.random.set_seed(42)


