from fastapi import FastAPI
import pandas as pd
import joblib
import uvicorn


app = FastAPI()

model = joblib.load("my_model.pkl")

@app.get("/")
def root():
    return {"status": "API is running", "model_loaded": True}

@app.post("/predict")
def predict_house_price(data: dict):


    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)


    return {"estimated_value": float(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)