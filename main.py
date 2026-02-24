from flask import Flask, render_template, request

from model_utils import predict

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    debug_mode = False

    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        debug_mode = request.form.get("debug") == "on"
        if input_text:
            result = predict(input_text, debug=debug_mode)

    return render_template("index.html", result=result, input_text=input_text, debug_mode=debug_mode)


if __name__ == "__main__":
    app.run(debug=True)
