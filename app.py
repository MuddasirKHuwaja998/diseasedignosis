import os
import gdown
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Path to download the trained model from Dropbox
MODEL_DROPBOX_LINK = "https://www.dropbox.com/scl/fi/m28vkfcb4xe24hrbpswg1/model.h5?rlkey=8dy2rnavoktedgidv3jys7fic&st=crfpl3y1&dl=1"
MODEL_PATH = "saved_model/model.h5"

# Ensure the model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Download the model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    try:
        print("Attempting to download the model from Dropbox...")
        gdown.download(MODEL_DROPBOX_LINK, MODEL_PATH, quiet=False)
        if os.path.exists(MODEL_PATH):
            print("Model downloaded successfully!")
        else:
            raise IOError("Model download failed: File not found after download.")
    except Exception as e:
        print(f"Error during model download: {e}")
        raise IOError("Model download failed. Please check the Dropbox link.")

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    raise IOError("Model loading failed. Ensure the model file is valid and compatible.")

# Disease details
class_details_alternate = {
    "Acute Otitis Media": [
        {
            "description": "La membrana timpanica appare arrossata e tesa a causa dell'accumulo di liquido infetto nell'orecchio medio. L'otite media acuta è caratterizzata da dolore auricolare acuto, febbre e riduzione temporanea dell'udito.",
            "causes": [
                "Infezioni virali delle vie aeree superiori",
                "Infezioni batteriche (Streptococcus pneumoniae, Haemophilus influenzae)",
                "Disfunzione della tuba di Eustachio (ad es. per allergie o raffreddore)"
            ]
        },
        {
            "description": "L'otite media acuta è un'infiammazione improvvisa dell'orecchio medio con raccolta di muco purulento dietro il timpano. Si manifesta con sensazione di pressione nell'orecchio e dolore intenso che può peggiorare di notte o durante cambiamenti di altitudine.",
            "causes": [
                "Infezioni virali del tratto respiratorio superiore",
                "Batteri ototossici comuni (Pneumococco, Moraxella catarrhalis)",
                "Anomalie anatomiche o funzionali della tuba di Eustachio"
            ]
        },
        {
            "description": "L'otite media acuta si verifica quando virus o batteri colonizzano lo spazio aerato dell'orecchio medio, spesso dopo un raffreddore. Il timpano appare sporgente o retratto e l'infiammazione può causare febbre e irritabilità, con dolore alla pressione dell'orecchio.",
            "causes": [
                "Infezioni respiratorie virali o batteriche",
                "Congestione delle vie aeree (raffreddore, allergie) con ostruzione tubarica",
                "Ventilazione tubarica inadeguata dell'orecchio medio"
            ]
        }
    ],
    "Chronic Otitis Media": [
        {
            "description": "È un'infezione cronica persistente dell'orecchio medio caratterizzata da perforazione timpanica e secrezione continua. Alla otoscopia, il timpano appare perforato con tessuto cicatriziale e spesso colesteatoma. Può causare ipoacusia stabile.",
            "causes": [
                "Infezioni acute ricorrenti dell'orecchio medio",
                "Perforazione cronica della membrana timpanica",
                "Malfunzionamento prolungato della tuba di Eustachio"
            ]
        },
        {
            "description": "Nell'otite media cronica il processo infettivo dell'orecchio medio persiste per mesi o anni. La membrana timpanica è spesso perforata o retratta con muco purulento che fuoriesce nell'orecchio esterno. I pazienti possono avvertire sensazione di orecchio tappato e perdita uditiva.",
            "causes": [
                "Infezioni non completamente guarite (batteriche)",
                "Danno timpanico da traumi o tubicini di ventilazione multipli",
                "Colesteatoma come complicanza di otiti ricorrenti"
            ]
        },
        {
            "description": "La condizione comporta un'infiammazione cronica dell'orecchio medio che porta a danni permanenti. Il timpano può apparire nascosto da tessuto di granulazione o cicatriziale. Questa patologia porta comunemente a ipoacusia conduttiva e a scarico persistente dal condotto uditivo.",
            "causes": [
                "Infezioni batteriche croniche (Pseudomonas, Stafilococco)",
                "Ostacolo cronico delle tube di Eustachio",
                "Ripetute perforazioni timpaniche con cicatrizzazione"
            ]
        }
    ],
    "Earwax Plug": [
        {
            "description": "Il tappo di cerume è un accumulo di cerume indurito che ostruisce il condotto uditivo esterno. All'otoscopia si osserva una massa brunastra o giallastra che copre il timpano parzialmente o totalmente. Può provocare ovattamento dell'udito, prurito o dolore auricolare leggero.",
            "causes": [
                "Eccessiva produzione di cerume",
                "Pulizia auricolare impropria (cotton-fioc)",
                "Struttura stretta del condotto uditivo esterno"
            ]
        },
        {
            "description": "Un deposito ceruminoso compatto blocca il normale passaggio del suono nel condotto uditivo, simulando un'ipoacusia conduttiva. Il timpano può risultare occluso alla vista a causa del cerume. Questo fenomeno può essere indotto da sovrapproduzione di cerume o da rimozioni scorrette.",
            "causes": [
                "Anomalie anatomiche del condotto uditivo (es. canale stretto)",
                "Uso scorretto di cotton-fioc o strumenti auricolari",
                "Aumento della produzione di cerume (iperproduzione ceruminosa)"
            ]
        }
    ],
    "Myringosclerosis": [
        {
            "description": "La miringosclerosi presenta piccole placche bianche calcificate sulla membrana timpanica, conseguenza di infiammazioni croniche o interventi timpanici (es. tubicini di ventilazione). Tali depositi rendono opaco il timpano all'otoscopia. Solitamente non provoca sintomi evidenti, ma può ridurre lievemente la mobilità timpanica.",
            "causes": [
                "Infiammazioni croniche ripetute dell'orecchio medio",
                "Ripetuti interventi di tubicini di ventilazione timpanica"
            ]
        },
        {
            "description": "Si osserva un ispessimento localizzato e aree di sclerosi (opacità bianche) sul timpano, esito di infezioni o traumi timpanici precedenti. Questi cambiamenti possono provocare una ridotta conduttività sonora, benché spesso non siano accompagnati da sintomi gravi.",
            "causes": [
                "Processi infiammatori cronici del timpano (es. otiti ricorrenti)",
                "Drenaggi timpanici (tubi) multipli"
            ]
        }
    ],
    "Normal": [
        {
            "description": "L'orecchio appare normale: membrana timpanica integra, trasparente e di colore madreperlaceo, senza segni di infiammazione, perforazioni o accumuli di cerume. Il condotto uditivo è pulito e senza secrezioni. Questa condizione riflette un funzionamento uditivo ottimale.",
            "causes": [
                "Assenza di patologie otologiche",
                "Condizione fisiologica dell'orecchio medio"
            ]
        },
        {
            "description": "In un orecchio sano il timpano è mobile e privo di lesioni, e il condotto uditivo non presenta anomalie. Non si rilevano infezioni né irritazioni. Questa normalità corrisponde a integrità delle strutture uditive e a percezione uditiva regolare.",
            "causes": [
                "Tuba di Eustachio funzionante correttamente",
                "Integrità delle strutture dell'orecchio"
            ]
        }
    ],
    "Otitis Externa": [
        {
            "description": "L'otite esterna è un'infiammazione del condotto uditivo esterno. Il canale appare arrossato e gonfio, spesso con desquamazione o secrezione purulenta. Il dolore aumenta alla pressione sul padiglione auricolare. Sono comuni prurito intenso e sensazione di orecchio tappato (nota come 'otite del nuotatore').",
            "causes": [
                "Esposizione prolungata all'umidità (nuoto, docce frequenti)",
                "Infezione batterica (es. Pseudomonas aeruginosa, Staphylococcus aureus)",
                "Traumi o irritazione del condotto (uso eccessivo di cotton-fioc)"
            ]
        },
        {
            "description": "L'infiammazione del condotto uditivo esterno si manifesta con rossore, gonfiore e possibile secrezione giallastra. Il paziente prova forte dolore auricolare e prurito. Questo quadro è comunemente dovuto a proliferazione di batteri o funghi nell'ambiente umido del canale.",
            "causes": [
                "Infezione fungina del condotto uditivo (Aspergillus, Candida)",
                "Infezione batterica del condotto uditivo",
                "Dermatite o eccessiva macerazione cutanea (es. saponi aggressivi)"
            ]
        },
        {
            "description": "La pelle del condotto uditivo esterno è infiammata e spesso desquamata; possono formarsi piccole lesioni dolorose. L'otoscopia rivela ostruzione parziale da detriti o esudato. L'otite esterna è favorita da fattori come acqua stagnante o pulizia traumatica del canale.",
            "causes": [
                "Accumulo di acqua nel canale uditivo esterno",
                "Pulizia del canale con strumenti non sterilizzati",
                "Eczema o altre patologie dermatologiche del condotto"
            ]
        }
    ],
    "Tymphanosclerosis": [
        {
            "description": "La timpanosclerosi comporta depositi di tessuto fibroso e calcareo sulla membrana timpanica e sugli ossicini. Il timpano presenta aree opache e ispessite all'otoscopia. Questa condizione, esito di infiammazioni croniche, può causare ipoacusia conduttiva dovuta alla ridotta mobilità della catena ossiculare.",
            "causes": [
                "Infiammazione cronica ricorrente dell'orecchio medio",
                "Perforazioni timpaniche guarite con cicatrici",
                "Interventi chirurgici o tubi timpanici ripetuti"
            ]
        },
        {
            "description": "I depositi sclerotici sulla membrana timpanica e sullo spazio timpanico sono tipici della timpanosclerosi. Si osservano placche bianche e fibrose sul timpano. Tale patologia è conseguenza di otiti medie croniche, più comuni in soggetti con ripetute perforazioni timpaniche.",
            "causes": [
                "Otiti medie croniche con sclerosi timpanica",
                "Drenaggio timpanico o perforazioni ricorrenti",
                "Processi infiammatori prolungati nell'orecchio medio"
            ]
        }
    ]
}
# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = list(class_details_alternate.keys())[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file is uploaded
        if "file" not in request.files:
            flash("No file uploaded!")
            return redirect(request.url)

        file = request.files["file"]

        # Handle single image upload
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_dir = os.path.join("static", "uploads")
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, filename)
            file.save(file_path)

            # Predict the uploaded image
            predicted_class, confidence = predict_image(file_path)

            # Get details for the predicted class
            details = class_details_alternate.get(predicted_class, [{}])[0]

            return render_template(
                "index.html",
                uploaded_image=url_for("static", filename=f"uploads/{filename}"),
                prediction=predicted_class,
                confidence=f"{confidence:.2f}%",
                description=details.get("description", "No description available."),
                causes=details.get("causes", [])
            )
        else:
            flash("Invalid file type! Please upload a valid image.")
            return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)