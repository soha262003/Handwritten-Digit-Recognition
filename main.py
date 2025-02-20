from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("mnist_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        img_data = np.array(request.form["img_data"]).reshape(1, 28, 28, 1)
        prediction = np.argmax(model.predict(img_data))
    return render_template("main.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
