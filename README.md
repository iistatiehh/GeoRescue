### README: Text and Image Disaster Detection Application

---

#### **Project Overview**
This project is a web-based application designed to process text and image inputs to detect potential disasters. It uses machine learning, natural language processing (NLP), and external APIs to identify disaster-related keywords, classify locations from images, and map detected entities to geographical coordinates.

---

#### **Features**
- **Text Processing**: Cleans user-provided text to remove noise (e.g., URLs, special characters) and extracts hashtags.
- **Disaster Detection**: Predicts whether the input text indicates a disaster using a trained classifier.
- **Named Entity Recognition (NER)**: Identifies geographical entities (locations) in the text using SpaCy.
- **Image Classification**: Classifies the location depicted in an image using the Picarta API.
- **Coordinate Mapping**: Finds latitude and longitude for detected locations using the GeoNames API.
- **User-Friendly Interface**: Interactive web forms with real-time feedback and styled result pages.

---

#### **Tech Stack**
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend**: HTML, CSS (with `styles.css` for styling), Jinja2 templates
- **NLP Libraries**: NLTK, SpaCy
- **APIs**:
  - [Picarta API](https://picarta.ai/) for image classification
  - [GeoNames API](http://www.geonames.org/) for geolocation
- **Modeling**: Custom classifier (`qnlpmodel`) with a vectorizer for text classification
- **Environment Management**: Python `os` module for secure token handling

---

#### **Installation and Setup**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required NLTK Data**:
   ```python
   import nltk
   nltk.download("stopwords")
   nltk.download("punkt")
   ```

5. **Configure API Tokens**:
   - Add the following environment variables to your `.env` file or system configuration:
     ```
     PICARTA_API_TOKEN=your-picarta-token
     GEONAMES_USERNAME=your-geonames-username
     ```

6. **Run the Application**:
   ```bash
   uvicorn main:app --reload
   ```
   Access the app at `http://127.0.0.1:8000`.

---

#### **File Structure**
```
project/
│
├── main.py                # FastAPI application code
├── qnlpmodel.py           # Custom model and vectorizer loading utilities
├── templates/             # HTML templates for the frontend
│   ├── index.html         # Homepage template
│   └── results.html       # Results page template
├── static/                # Static files (CSS/JS)
│   └── styles.css         # Custom styling
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

#### **Usage**
1. Navigate to the homepage (`http://127.0.0.1:8000`).
2. Enter text and an image URL in the provided form.
3. Click "Process" to:
   - Analyze the text for disaster indicators.
   - Classify the image for potential location matches.
   - Display the results, including:
     - Disaster detection status.
     - Extracted hashtags.
     - Named entities and their geographical coordinates (if applicable).

---

#### **Key Outputs**
- **Disaster Detected**:
  - Displays an alert with a red background.
  - Shows identified locations and coordinates in a table.
- **No Disaster Detected**:
  - Displays an alert with a green background.
  - Skips displaying the tables for entities and coordinates.

---

#### **APIs Used**
1. **Picarta API**: Classifies locations in image URLs.
   - Input: Image URL
   - Output: Top location predictions
2. **GeoNames API**: Retrieves latitude and longitude for detected locations.
   - Input: Location name
   - Output: Latitude and longitude

---

#### **Improvements**
- Enhance the classifier for better disaster detection accuracy.
- Add support for multiple languages in text processing.
- Improve error handling for API requests.
- Extend functionality for multimedia inputs like videos.

---

#### **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
