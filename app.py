from fastapi import FastAPI, Form, HTTPException, Request
import requests
import re
import nltk
import spacy
from nltk.corpus import stopwords
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qnlpmodel import model_load, load_vectorier
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

PICARTA_API_TOKEN = os.getenv("PICARTA_API_TOKEN", "U7XIEFTM9APVVD1IR4C6")
GEONAMES_USERNAME = os.getenv("GEONAMES_USERNAME", "istatieh")

model = model_load("classifer.cfl")
vectorier = load_vectorier("data.csv")

nltk.download("stopwords")
nltk.download("punkt")

nlp = spacy.load("en_core_web_sm")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def text_processing(input_text):
    input_text = re.sub(r"http\S+|www\S+|https\S+", " ", input_text, flags=re.MULTILINE)
    input_text = re.sub(r"#\s+(\w+)", r"#\1", input_text)
    hashtags = re.findall(r"#(\w+)", input_text)
    input_text = input_text.lower()
    input_text = re.sub(r"@\w+|[^a-z\s]", " ", input_text)
    words = input_text.split()
    filtered_words = [word for word in words if word not in stopwords.words("english")]
    cleaned_text = " ".join(filtered_words)
    return cleaned_text.strip(), hashtags

def apply_ner(input_text):
    doc = nlp(input_text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Function to classify location using Picarta API
def classify_location_with_media(media_url):
    headers = {"Content-Type": "application/json"}
    payload = {"TOKEN": PICARTA_API_TOKEN, "IMAGE": media_url, "TOP_K": 3}
    response = requests.post("https://picarta.ai/classify", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail=f"Picarta API error: {response.text}")

# Function to find coordinates using GeoNames API
def get_coordinates(location):
    params = {"q": location, "maxRows": 1, "username": GEONAMES_USERNAME}
    response = requests.get("http://api.geonames.org/searchJSON", params=params)

    if response.status_code == 200:
        data = response.json()
        if data["totalResultsCount"] > 0:
            result = data["geonames"][0]
            return result["lat"], result["lng"]
        return None, None
    else:
        raise HTTPException(status_code=500, detail=f"GeoNames API error: {response.text}")

@app.post("/process_input", response_class=HTMLResponse)
def process_input(request: Request, text: str = Form(...), image_url: str = Form(...)):
    try:
        # Step 1: Preprocess text and extract hashtags
        cleaned_text, hashtags = text_processing(text)

        # Step 2: Use the model to predict if there's a disaster in the text
        is_disaster = model.predict(vectorier.transform([cleaned_text]))[0][0] == 1

        if is_disaster:
            # Step 3: Apply NER to the cleaned text
            ner_entities = apply_ner(cleaned_text)

            # Step 4: Use Picarta API for media location classification
            media_locations = classify_location_with_media(image_url)

            # Step 5: Include the top Picarta result in coordinates and find others
            coordinates = []
            if "predictions" in media_locations and media_locations["predictions"]:
                top_location = media_locations["predictions"][0]["label"]
                lat, lng = get_coordinates(top_location)
                if lat and lng:
                    coordinates.append({"entity": top_location, "latitude": lat, "longitude": lng})

            # Add coordinates from named entities
            for entity, label in ner_entities:
                if label == "GPE":  # GPE = Geo-Political Entity
                    lat, lng = get_coordinates(entity)
                    if lat and lng:
                        coordinates.append({"entity": entity, "latitude": lat, "longitude": lng})

            return templates.TemplateResponse(
                "results.html",
                {
                    "request": request,
                    "text": text,
                    "cleaned_text": cleaned_text,
                    "hashtags": hashtags,
                    "ner_entities": ner_entities,
                    "coordinates": coordinates,
                    "is_disaster": True,
                },
            )
        else:
            # No disaster detected
            return templates.TemplateResponse(
                "results.html",
                {
                    "request": request,
                    "text": text,
                    "cleaned_text": cleaned_text,
                    "hashtags": hashtags,
                    "ner_entities": [],
                    "coordinates": [],
                    "is_disaster": False,
                },
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))