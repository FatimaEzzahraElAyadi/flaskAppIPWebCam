from flask import Flask, render_template, request, session, flash, redirect,url_for, jsonify, Response
import PIL
from ultralytics import YOLO
import base64
import os
import cv2
import easyocr
import time
import numpy as np
import datetime
from datetime import datetime

import psycopg2 #pip install psycopg2 
import psycopg2.extras
import re
from werkzeug.security import generate_password_hash, check_password_hash
import glob


app = Flask(__name__)
app.secret_key = "8e1c58670df72ee926c77e8f890ea304542b1d08e299d622042c75e2e28e693f"

DB_HOST = "localhost"
DB_NAME = "webcamdetectiondb"
DB_USER = "postgres"
DB_PASS = "fatima"
 
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


# Setting page layout
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

MODEL_DIRECTORY = 'C:\\Users\\hp\\Desktop\\flaskAppIPWebCam\\static\\models'

def find_model_paths(directory):
    model_paths = {}
    for filepath in glob.glob(os.path.join(directory, "*.pt")):
        model_name = os.path.splitext(os.path.basename(filepath))[0].replace("_modele", "").replace("_", " ").title()
        model_paths[model_name] = filepath
    return model_paths

model_paths = find_model_paths(MODEL_DIRECTORY)

def load_model(selected_model):
    model_path = model_paths.get(selected_model, '')
    try:
        return YOLO(model_path)
    except Exception as ex:
        return None
    
        

def apply_ocr(image, box):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the box coordinates from the Boxes object
    x, y, w, h = map(int, box.xyxy[0].tolist())

    # Extract the region of interest (ROI) using the box coordinates
    roi = image_rgb[y:h, x:w]

    # Convert the ROI to gray scale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Apply OCR to the gray scale ROI
    reader = easyocr.Reader(['en'])
    result = reader.readtext(roi_gray)

    return result

def format_detection_result(detected_classes,model):
    formatted_result = ""
    class_counts = {}
    
    # Compter le nombre d'occurrences de chaque classe détectée
    for class_id in detected_classes:
        class_id = int(class_id)
        if class_id not in class_counts:
            class_counts[class_id] = 0
        class_counts[class_id] += 1
    
    # Convertir le résultat formaté en une chaîne de caractères
    for class_id, count in class_counts.items():
        class_name = model.names[class_id]
        formatted_result += f"{count} {class_name} "
    
    
    return formatted_result

def find_class_index(classes, class_name):
    for key, value in classes.items():
        if value == class_name:
            return key
    return None

def post_process_detections(outs, classes, conf_threshold):
    boxes = outs[0].boxes
    names = boxes.cls
    epi_detections = []

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        bbox = boxes.xyxy[i]
        x_min, y_min, x_max, y_max = bbox.tolist()

        if confidence > conf_threshold:
            if classes[class_id] != 'human':
                epi_detections.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min), classes[class_id]))
            else:
                epi_detections.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min), classes[class_id]))

    return epi_detections

def filter_person_detections(boxes, person_id, conf_threshold):
    person_detections = []

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        bbox = boxes.xyxy[i]
        x_min, y_min, x_max, y_max = bbox.tolist()

        if class_id == person_id and confidence > conf_threshold:
            person_detections.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))

    return person_detections

def filter_epi_near_person(person_detections, epi_detections):
    person_epis = []
    for person in person_detections:
        person_x, person_y, person_w, person_h = person
        person_center_x = person_x + person_w / 2
        person_center_y = person_y + person_h / 2
        person_area = person_w * person_h

        person_epis_for_this_person = []
        for epi in epi_detections:
            epi_x, epi_y, epi_w, epi_h, epi_class = epi
            epi_center_x = epi_x + epi_w / 2
            epi_center_y = epi_y + epi_h / 2
            epi_area = epi_w * epi_h

            distance = np.sqrt((person_center_x - epi_center_x)**2 + (person_center_y - epi_center_y)**2)

            if distance < 1.5 * np.sqrt(person_area + epi_area):
                person_epis_for_this_person.append(epi)

        person_epis.extend(person_epis_for_this_person)

    return person_epis

def count_epi_classes(person_epis):
    epi_counts = {}
    unique_epis = set()

    for epi in person_epis:
        epi_id = (epi[0], epi[1], epi[2], epi[3], epi[4])  
        unique_epis.add(epi_id)

    for epi_id in unique_epis:
        epi_class = epi_id[4]
        epi_counts[epi_class] = sum(1 for epi in person_epis if (epi[0], epi[1], epi[2], epi[3], epi[4]) == epi_id)

    return epi_counts

def display_epi_counts(epi_counts):
    output = ""
    for epi_class, count in epi_counts.items():
        output += f"{count} {epi_class}, "
    return output

def detect_and_draw(net, frame, conf_threshold,classes,person_id):
    # Détection des objets
    outs = net.predict(frame)

    # Post-traitement des détections
    epi_detections = post_process_detections(outs, classes, conf_threshold)

    # Filtrage des détections pour ne garder que celles près des personnes
    person_detections = filter_person_detections(outs[0].boxes, person_id, conf_threshold)

    # Filtrage des détections d'EPI près des personnes
    person_epis = filter_epi_near_person(person_detections, epi_detections)

    # Dessiner les détections sur le frame
    for person in person_detections:
        person_x, person_y, person_w, person_h = person
        cv2.rectangle(frame, (person_x, person_y), (person_x + person_w, person_y + person_h), (0, 255, 0), 2)

    for epi in person_epis:
        epi_x, epi_y, epi_w, epi_h, epi_class = epi
        cv2.rectangle(frame, (epi_x, epi_y), (epi_x + epi_w, epi_y + epi_h), (0, 255, 0), 2)
        cv2.putText(frame, epi_class, (epi_x, epi_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Fonction pour calculer l'IoU entre deux boîtes
def iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1.xyxy[0]
    x1_2, y1_2, x2_2, y2_2 = box2.xyxy[0]

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

@app.route('/register', methods=['GET', 'POST'])
def register():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
 
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        _hashed_password = generate_password_hash(password)

        #Check if account exists using MySQL
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        print(account)
        # If account exists show error and validation checks
        if account:
            flash('Account already exists!')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers!')
        elif not username or not password or not email:
            flash('Please fill out the form!')
        else:
            # Account doesnt exists and the form data is valid, now insert new account into users table
            cursor.execute("INSERT INTO users (fullname, username, password, email) VALUES (%s,%s,%s,%s)", (fullname, username, _hashed_password, email))
            conn.commit()
            flash('You have successfully registered!')
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash('Please fill out the form!')
    # Show registration form with message (if any)
    return render_template('register.html')

@app.route('/login/', methods=['GET', 'POST'])
def login():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        print(password)
 
        # Check if account exists using MySQL
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        # Fetch one record and return result
        account = cursor.fetchone()
 
        if account:
            password_rs = account['password']
            print(password_rs)
            # If account exists in users table in out database
            if check_password_hash(password_rs, password):
                # Create session data, we can access this data in other routes
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                # Redirect to home page
                return redirect(url_for('index'))
            else:
                # Account doesnt exist or username/password incorrect
                flash('Incorrect username/password')
        else:
            # Account doesnt exist or username/password incorrect
            flash('Incorrect username/password')
 
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    global cap  # Utiliser la variable globale cap

    # Vérifier si l'utilisateur est connecté
    if 'loggedin' not in session:
        # L'utilisateur n'est pas connecté, rediriger vers la page de connexion
        return redirect(url_for('login'))

    # Initialiser la liste d'alertes en dehors de la session
    if 'alertes' not in session:
        session['alertes'] = []

    alertes = session['alertes']  # Charger les alertes existantes

    model_paths = find_model_paths(MODEL_DIRECTORY)

    if request.method == 'POST':    
        selected_model = request.form['model']
        model = load_model(selected_model)

        if model is not None:
            # Créer un objet VideoWriter pour enregistrer la vidéo
            upload_folder = 'static/uploads'
            video_filename = f"{selected_model}_detection.avi"
            video_path = os.path.join(upload_folder, video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))  # Ajustez la résolution 

            if 'start_detection' in request.form:
                #url = "http://192.168.19.211:8080/video"
                # Capturer la vidéo depuis ip webcam
                cap = cv2.VideoCapture(0)

                # Définir le seuil de mouvement
                movement_threshold = 20  
                

                # Dictionnaire pour stocker les boîtes et leurs identifiants
                previous_boxes= {}

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Effectuer la détection sur le frame actuel
                    res = model.predict(frame)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]

                   

                    # Insérer les résultats de détection dans la base de données
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                    cursor.execute("INSERT INTO inputs (modele) VALUES (%s) RETURNING id", (selected_model,))
                    input_id = cursor.fetchone()[0]

                    
                                    
 
                    # Check detected classes against model classes
                    detected_classes = boxes.cls
                    model_classes = model.names if hasattr(model, 'names') else None

                    # Trouver l'indice de la classe 'human'
                    person_id = find_class_index(model_classes, 'human')

                    conf_threshold = 0.5

                    epi_detections = post_process_detections(res, model_classes, conf_threshold)

                    # Filtrage des détections pour ne garder que celles près des personnes
                    person_detections = filter_person_detections(res[0].boxes, person_id, conf_threshold)

                    # Filtrage des détections d'EPI près des personnes
                    person_epis = filter_epi_near_person(person_detections, epi_detections)

                    # Compter le nombre d'EPI de chaque classe détectés
                    epi_counts = count_epi_classes(person_epis)

                    # Afficher le nombre d'EPI de chaque classe
                    formatted_detection_result_ppe= display_epi_counts(epi_counts)

                    # Traitement de la frame actuelle
                    processed_frame = detect_and_draw(model, frame, conf_threshold,model_classes,person_id)

                    if model_classes:
                        if selected_model == "Ppe Detection":
                            for person_index, person in enumerate(person_detections, 1):
                                epi_personne = [epi[4] for epi in epi_detections if person_epis]  # Liste des EPI portés par la personne
                                if "helmet" not in epi_personne and "vest" not in epi_personne:
                                    alerte = f"Person {person_index} detected without helmet and vest"
                                elif "helmet" not in epi_personne:
                                    alerte = f"Person {person_index} detected without helmet"
                                elif "vest" not in epi_personne:
                                    alerte = f"Person {person_index} detected without vest"
                                else:
                                    continue  # Si les deux équipements sont présents, passer à la personne suivante
                                alertes.append(alerte)
                                session['alertes'] = alertes
                                current_time = datetime.now()
                                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                                cursor.execute("INSERT INTO alertes (inputid,alertedescription,timestamp) VALUES (%s,%s,%s)", (input_id, alerte,current_time))
                        else:
                            missing_classes = [model_classes[class_id] for class_id in model_classes if class_id not in detected_classes]
                            if missing_classes:
                                new_alerts = {', '.join(missing_classes)}
                                alertes.extend(new_alerts)

                                session['alertes'] = alertes

                                for alerte in new_alerts:
                                    current_time = datetime.now()
                                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                                    cursor.execute("INSERT INTO alertes (inputid,alertedescription,timestamp) VALUES (%s,%s,%s)", (input_id,alerte,current_time))

                    if selected_model == "Ppe Detection":
                        # Enregistrer le frame traité dans la vidéo de sortie
                        out.write(processed_frame)

                        # Affichage du frame traité
                        cv2.imshow('Processed Frame', processed_frame)
                    
                    elif selected_model == "Forklift Detection":

                        # Si nous avons des boîtes du frame précédent et du frame actuel
                        if previous_boxes and boxes:
                            for i, box in enumerate(boxes):
                                # Obtenir les coordonnées de la boîte actuelle
                                x1, y1, x2, y2 = box.xyxy[0]  # Coin supérieur gauche et coin inférieur droit

                                # Chercher une correspondance dans le frame précédent
                                matching_box = previous_boxes.get(i)

                                # Si on trouve une correspondance
                                if matching_box:
                                    # Obtenir les coordonnées de la boîte précédente
                                    prev_x1, prev_y1, prev_x2, prev_y2 = matching_box.xyxy[0]

                                    # Calculer le déplacement
                                    displacement = np.sqrt((x1 - prev_x1) ** 2 + (y1 - prev_y1) ** 2)

                                    # Afficher le statut en fonction du seuil de mouvement
                                    if displacement > movement_threshold:
                                        print(f"Forklift {i+1} is moving!")
                                        current_time = datetime.now()
                                        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                                        cursor.execute("INSERT INTO infos (inputid,infodescription,timestamp) VALUES (%s,%s,%s)", (input_id,f"Forklift {i+1} is moving!",current_time))
                                    else:
                                        print(f"Forklift {i+1} is stationary.")
                                        current_time = datetime.now()
                                        cursor.execute("INSERT INTO infos (inputid,infodescription,timestamp) VALUES (%s,%s,%s)", (input_id,f"Forklift {i+1} is stationary",current_time))
                                else:
                                    # Si c'est une nouvelle détection, ajouter à la liste des détections précédentes
                                    previous_boxes[i] = box

                        # Mettre à jour les boîtes précédentes avec les boîtes actuelles
                        previous_boxes = {i: box for i, box in enumerate(boxes)}
                                    

                        out.write(frame)
                        cv2.imshow('Real-time Detection', res_plotted)
                                
       
                    else:

                        # Insérer les résultats de détection dans la vidéo
                        out.write(frame)

                        # Afficher la vidéo avec les résultats de la détection en temps réel
                        cv2.imshow('Real-time Detection', res_plotted)

                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

                    formatted_detection_result = format_detection_result(detected_classes, model)
                    # Insérer les résultats dans la table correspondante en fonction du modèle sélectionné
                    if selected_model == "Ppe Detection":
                        cursor.execute("INSERT INTO ppedetection (inputid, videodetected, detectionresult) VALUES (%s, %s, %s)", (input_id, video_path, formatted_detection_result_ppe))
                    elif selected_model == "Plate Number Detection":
                        for box in boxes:
                            ocr_result = apply_ocr(res_plotted, box)
                            plate_number_text = ocr_result[0][1] if ocr_result else "N/A"
                            cursor.execute("INSERT INTO platenumberdetection (inputid, videodetected, detectionresult) VALUES (%s, %s, %s)", (input_id, video_path, plate_number_text))
                    elif selected_model == "Forklift Detection":
                        cursor.execute("INSERT INTO forkliftdetection (inputid, videodetected, detectionresult) VALUES (%s, %s, %s)", (input_id, video_path, formatted_detection_result))
                            
                    elif selected_model == "TrainCars Detection":
                        cursor.execute("INSERT INTO traincarsdetection (inputid, videodetected, detectionresult) VALUES (%s, %s, %s)", (input_id, video_path, formatted_detection_result))
                            
                    elif selected_model == "Carre Detection":
                        cursor.execute("INSERT INTO carredetection (inputid, videodetected, detectionresult) VALUES (%s, %s, %s)", (input_id, video_path, formatted_detection_result))
                        if [model_classes[class_id] for class_id in model_classes if class_id in detected_classes]:
                            cursor.execute("INSERT INTO infos (inputid,infodescription,timestamp) VALUES (%s,%s,%s)", (input_id,"Lane is busy!",current_time))

                    conn.commit()
                    cursor.close()
                
                 # Libérer la capture vidéo et détruire toutes les fenêtres ouvertes
                cap.release()
                out.release()
                cv2.destroyAllWindows()

            # Libérer la capture vidéo et détruire toutes les fenêtres ouvertes
            elif 'stop_detection' in request.form:
                if 'cap' in globals() and cap.isOpened():
                    cap.release()

                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # Récupérer les alertes et infos depuis la base de données
                if selected_model == "Ppe Detection":
                    cursor.execute("SELECT alertedescription, timestamp FROM alertes")
                    alertes_from_db = cursor.fetchall()
                    alertes = [{"alertedescription": alerte['alertedescription'], "timestamp": alerte["timestamp"]} for alerte in alertes_from_db]
                    cursor.close()
                    return render_template('index.html', alertes=alertes)
                elif selected_model == "Forklift Detection" or selected_model =="Carre Detection":
                    cursor.execute("SELECT infodescription, timestamp FROM infos")
                    infos_from_db = cursor.fetchall()
                    infos = [{"infodescription": info["infodescription"], "timestamp": info["timestamp"]} for info in infos_from_db]
                    cursor.close()
                    return render_template('index.html',infos=infos)
                
            
            out.release()
            cv2.destroyAllWindows()

            return render_template('index.html', video_filename=video_filename, alertes=alertes)

    # Pass the file paths, encoded image data, and OCR result to the template
        
    return render_template('index.html', alertes=alertes)


@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

@app.route('/profile')
def profile(): 
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    # Check if user is loggedin
    if 'loggedin' in session:
        cursor.execute('SELECT * FROM users WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/tables')
def tables():
    return render_template('tables.html')


if __name__ == '__main__':
    app.run(debug=True)