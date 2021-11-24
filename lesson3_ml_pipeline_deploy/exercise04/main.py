
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

from typing import Optional

class Body(BaseModel):
    single_item: str

# Instantiate the app.
app = FastAPI()

# Define a POST on the specified endpoint.
@app.post('/{path}')
async def exercise_function(path: str, query: str, body: Body):
    return {"path": path, "query": query, "body": body}

# Define a POST on the specified endpoint.
@app.get('/')
async def hello_world(name: Optional[str] = None):
    return f'{name} Hola mundo!!!' if name is not None else 'Hola mundo!!!'
