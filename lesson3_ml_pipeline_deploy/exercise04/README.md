# Exercise: Parameters and Input in FastAPI
Write a simple script that creates a FastAPI app and defines a POST method that takes one path parameter, one query parameter, and a request body containing a single field. Have this function return all three in a dict. To get started, you can use the following snippet -- note the missing imports:

```python
@app.post(...)
async def exercise_function(...):
  return {"path": path, "query": query, "body": body}
```