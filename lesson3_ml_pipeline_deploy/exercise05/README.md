# Exercise: Local API Testing
Install FastAPI and uvicorn in a conda environment if you haven't already. Copy the following main.py to your local directory:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}
```

Write a unit test that tests the status code and response of the defined GET method. Be thorough and test with and without the query parameter, and test a malformed URL.