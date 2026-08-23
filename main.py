from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    form = await request.form()

    data = {
        "site": form["site"],
        "Pop": form["Pop"],
        "sex": form["sex"],
        "hdlngth": float(form["hdlngth"]),
        "skullw": float(form["skullw"]),
        "totlngth": float(form["totlngth"]),
        "taill": float(form["taill"]),
        "footlgth": float(form["footlgth"]),
        "earconch": float(form["earconch"]),
        "eye": float(form["eye"]),
        "chest": float(form["chest"]),
        "belly": float(form["belly"])
    }

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction, 2)
        }
    )