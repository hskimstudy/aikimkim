from fastapi import FastAPI, HTTPException, Form
import joblib
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate('apikim-17704-firebase-adminsdk-ncp61-6afeee53d7.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://apikim-17704-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# Create FastAPI instance
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Define API endpoin
@app.post("/")
async def make_prediction(A: float = Form(...), B: float = Form(...), C: float = Form(...),
                          D: float = Form(...), E: float = Form(...)):
    try:
        loaded_model = joblib.load('finalized_model_softmax.sav')
        input_data = np.array([[A, B, C, D, E]])
        prediction = loaded_model.predict(input_data)
        
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
