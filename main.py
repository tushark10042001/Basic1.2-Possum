from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict", response_class=HTMLResponse)
def predict(
    site: int = Form(...),
    Pop: str = Form(...),
    sex: str = Form(...),
    hdlngth: float = Form(...),
    skullw: float = Form(...),
    totlngth: float = Form(...),
    taill: float = Form(...),
    footlgth: float = Form(...),
    earconch: float = Form(...),
    eye: float = Form(...),
    chest: float = Form(...),
    belly: float = Form(...)
):

    data = pd.DataFrame([{
        "site": site,
        "Pop": Pop,
        "sex": sex,
        "hdlngth": hdlngth,
        "skullw": skullw,
        "totlngth": totlngth,
        "taill": taill,
        "footlgth": footlgth,
        "earconch": earconch,
        "eye": eye,
        "chest": chest,
        "belly": belly
    }])

    prediction = model.predict(data)[0]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Age Prediction</title>
    </head>

    <body>

        <h1>Age Prediction</h1>

        <h2>Predicted Age: {prediction:.2f}</h2>

        <a href="/">Make another prediction</a>

    </body>
    </html>
    """
