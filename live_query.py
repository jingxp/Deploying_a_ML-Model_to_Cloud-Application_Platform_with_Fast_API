import requests

endpoint = 'https://mlops-final-project-3609b5418d04.herokuapp.com/predict'
#endpoint = 'https://mlops-final-project-3609b5418d04.herokuapp.com'

test_case = {"age": 52,
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

response = requests.post(endpoint, json=test_case)
#response = requests.get(endpoint)
print(response.status_code)
print("prediction: ")
print(response.json())