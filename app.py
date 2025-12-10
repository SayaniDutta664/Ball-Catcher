from flask import Flask, render_template, Response, jsonify
import cv2
from hand_tracker import HandTracker  # your custom hand tracking module

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize hand tracker
tracker = HandTracker()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/game")
def game():
    return render_template("game.html")

@app.route("/how-to-play")
def how_to_play():
    return render_template("how_to_play.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Webcam video generator
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue  # skip if frame not read

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Process frame for hand tracking
        tracker.process(frame)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/bucket_position")
def bucket_position():
    x = tracker.bucket_x
    return jsonify({"x": x})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
