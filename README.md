### Housing Price Prediction Based on Socioeconomic and Geographical Factors

```markdown
# House Price Prediction Web App

This is a simple web application built using Flask, TensorFlow, and Scikit-learn for predicting house prices based on various features of the housing dataset. The model is trained on the California Housing dataset.

## Features

- User-friendly web interface to input housing features.
- Predicts house prices based on user input.
- Utilizes a neural network model for accurate predictions.
- Scales input features for better model performance.

## Tech Stack

- **Frontend**: HTML, CSS
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Scikit-learn, Pandas, NumPy
- **Data Storage**: Model saved as HDF5 (`.h5`) and Scikit-learn's `joblib` for scalers.

## Installation

### Prerequisites

Make sure you have Python installed on your machine. It is recommended to create a virtual environment for this project. You can create a virtual environment using:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```

- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### Install Required Packages

Use pip to install the required packages:

```bash
pip install Flask tensorflow scikit-learn pandas numpy joblib
```

### Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/house_price_prediction.git
```

Navigate to the project directory:

```bash
cd house_price_prediction
```

## Usage

1. **Train the Model**: Before running the web app, you need to train the model. Run the `model.py` script to create the model and scaler files.

   ```bash
   python model.py
   ```

2. **Run the Flask App**: Start the Flask server by running:

   ```bash
   python app.py
   ```

3. **Access the Web App**: Open your web browser and go to `http://127.0.0.1:5000/` to access the application.

4. **Input Data**: Fill in the required fields and click the "Predict" button to see the predicted house price.

## Project Structure

```
house_price_prediction/
│
├── app.py              # Main Flask application file
├── model.py            # Script to train and save the model
├── templates/          # Folder containing HTML templates
│   └── index.html      # Main HTML template for user input
└── static/             # Folder for static files like CSS
    └── style.css       # CSS styles for the web app
```

## Acknowledgments

- California Housing dataset from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).
- [Flask](https://flask.palletsprojects.com/) for creating the web application.
- [TensorFlow](https://www.tensorflow.org/) for building the machine learning model.
- [Scikit-learn](https://scikit-learn.org/) for data preprocessing.

## Made By

**Nishant Kumar**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Customization

Feel free to further customize any part of the README file, such as the repository link, license information, and any additional sections or details specific to your project.
