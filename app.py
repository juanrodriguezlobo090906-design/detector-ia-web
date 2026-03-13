from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No se subió ninguna imagen"

    file = request.files["image"]
    if file.filename == "":
        return "No se seleccionó archivo"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # --- Leer imagen ---
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- ANALISIS DE BORDES ---
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges)

    # --- ANALISIS DE RUIDO ---
    noise = np.std(gray)

    # --- ANALISIS DE TEXTURA ---
    texture = np.mean(cv2.Laplacian(gray, cv2.CV_64F))

    # --- SCORE FINAL ---
    score = (edge_density*0.4) + (noise*0.3) + (texture*0.3)
    probabilidad = int((score/255)*100)
    probabilidad = max(5, min(probabilidad, 95))  # evita 0% o >95%

    # --- HEATMAP FORENSE ---
    hf = cv2.subtract(gray, cv2.GaussianBlur(gray, (3,3), 0))
    hf = cv2.normalize(hf, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(hf, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # --- DETECCION DE ROSTROS ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,0,255), 2)

    # --- GUARDAR HEATMAP ---
    result_image = os.path.join(app.config["RESULT_FOLDER"], file.filename)
    cv2.imwrite(result_image, overlay)

    result = f"{probabilidad}% de probabilidad de que la imagen sea generada por IA"

    return render_template("index.html",
                           result=result,
                           filename=file.filename,
                           result_image=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    # Ejecutar Flask en localhost y puerto 5000
    app.run(host="0.0.0.0", port=5000, debug=True)