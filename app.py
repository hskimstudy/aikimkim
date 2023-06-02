from fastapi import FastAPI, Request, Form, HTTPException
import joblib
import numpy as np

app = FastAPI()

@app.get("/")
async def get_prediction():
    return {"message": "Send a POST request to this endpoint to make predictions."}

@app.post("/")
async def create_user(request: Request, A: float = Form(...), B: float = Form(...), C: float = Form(...), D: float = Form(...), E: float = Form(...)):
    try:
        loaded_model = joblib.load('finalized_model_softmax.sav')
        input_data = np.array([[float(A), float(B), float(C), float(D), float(E)]])
        prediction = loaded_model.predict(input_data)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

