from keras.models import model_from_json
import numpy as np
import customtkinter as ctk
import os
from PIL import Image, ImageTk
import cv2

## Estetica de la interfaz

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

## Manejo de las diferentes ventanas que se crean - Se diseña para que se remplazen y no se abran simultaneamente.
class Main(ctk.CTk):

    def __init__(self):
        ctk.CTk.__init__(self)

        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.geometry("900x800")

        self.CTkFrame = {}

        for F in (ArkangelAI, CameraFunc):

            frame = F(container, self)

            self.CTkFrame[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(ArkangelAI)

    def show_frame(self, cont):
        frame = self.CTkFrame[cont]
        frame.tkraise()

## Inicio de Sesion - Interfaz - Funcionalidad

class ArkangelAI(ctk.CTkFrame):

    def __init__(self, parent, controller):

        ctk.CTkFrame.__init__(self, parent)

        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Images")
        self.FinisTC = ctk.CTkImage(Image.open(os.path.join(image_path, "logo.png")),size=(600, 200))

        self.controller = controller

## Top Frame
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(expand=True, fill='both')


        label = ctk.CTkLabel(self, text="", image=self.FinisTC)
        label.pack(pady=12, padx=10)

        label1 = ctk.CTkLabel(self, text="Emotions Detector App", font=("Perpetua", 18, "bold"))
        label1.pack(pady=12, padx=10)

        logbutton = ctk.CTkButton(self, text="Start", command=lambda: controller.show_frame(CameraFunc))
        logbutton.pack(pady=12, padx=10)

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(expand=True, fill='both')

class CameraFunc(ctk.CTkFrame):

    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)

        ## Grid_Config

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        middle_frame = ctk.CTkFrame(self)
        middle_frame.pack(side='top', expand=True, anchor='center' ,fill='both', padx=580, pady=120, ipady=80)

        label1 = ctk.CTkLabel(self, text="Emotions Detector App", font=("Perpetua", 18, "bold"))
        label1.pack(pady=12, padx=10)

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(expand=True, fill='both')

        self.camera_canvas = ctk.CTkCanvas(middle_frame)
        self.camera_canvas.pack(expand=True, fill='both')

        self.cap = cv2.VideoCapture(0)


        self.init_emotion_detection()

    def init_emotion_detection(self):
        json_file = open("emotionsdetectormodel.json", "r")
        model_json = json_file.read()
        json_file.close()

        self.emotion_model = model_from_json(model_json)
        self.emotion_model.load_weights("emotionsdetectormodel.h5")

        self.haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.haar_file)

        self.labels = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}

        # Start webcam capture
        self.webcam = cv2.VideoCapture(0)

        # Call the function to start emotion detection
        self.detect_emotions()

    def detect_emotions(self):
        ret, frame = self.webcam.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)

            try:
                for (p, q, r, s) in faces:
                    image = gray[q:q + s, p:p + r]
                    cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 222, 0), 2)
                    image = cv2.resize(image, (48, 48))
                    img = self.ef(image)
                    pred = self.emotion_model.predict(img)
                    prediction_label = self.labels[pred.argmax()]

                    cv2.putText(frame, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                                (0, 222, 255))

                # Convert the OpenCV frame to a format that Tkinter can display (PIL Image)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((500, 500))
                img = ImageTk.PhotoImage(image=img)

                # Update the canvas with the new image
                self.camera_canvas.create_image(250, 250, image=img)
                self.camera_canvas.image = img  # Save a reference to prevent garbage collection

            except cv2.error:
                pass

        # Call this function recursively to update the camera feed and emotion detection continuously
        self.after(10, self.detect_emotions)

    def ef(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
app = Main()

app.mainloop()
