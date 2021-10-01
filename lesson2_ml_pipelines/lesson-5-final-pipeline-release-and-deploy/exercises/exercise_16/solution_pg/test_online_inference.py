import requests
import json

with open("data_sample.json") as fp:
    data = json.load(fp)

results = requests.post("http://localhost:5000/invocations", json=data)

print(results.json())