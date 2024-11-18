from fastapi import FastAPI

app = FastAPI()

def get_app_description():
    return(
        'Welcome to Iris Species Prediction API!'
        'This API allows you to predict the species of an iris flower based on its petal measurements'
        "Use the '/predict/' endpoint with a POST request to make predictions."
        "Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
    )

@app.get('/')
async def root():
    return{'message' : get_app_description()}

# logistic regression classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# train a logistic regression model
model = LogisticRegression(max_iter=100000)
model.fit(X, y)

# define a function to predict species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
     features = [[sepal_length, sepal_width, petal_length, petal_width]]
     prediction = model.predict(features)
     return iris.target_names[prediction[0]]

# define the Pydantic model for your input data
from pydantic import BaseModel

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
# create API endpoint
@app.post('/predict')
async def predict_species_api(iris_data : IrisData):
         species = predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)
         return {'species' : species}
