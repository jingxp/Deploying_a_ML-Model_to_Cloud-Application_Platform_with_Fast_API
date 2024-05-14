from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

pos_case = {"age": 52,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States"
            }

neg_case = {"age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }

def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World !"}


def test_pos_outcome():
    prediction = client.post("/predict", json=pos_case)
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] in [">=50K", "<=50K"]


def test_neg_outcome():
    prediction = client.post("/predict", json=neg_case)
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] in [">=50K", "<=50K"]