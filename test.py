import base64
import cv2
from flask import Flask, Response
from flask import jsonify,stream_with_context,Flask,render_template,Response

app = Flask(__name__)

@app.route('/stream_test')
def stream_test():
    return render_template('stream_test.html')

@app.route('/')
def index():
  # Open the webcam
  cap = cv2.VideoCapture('videos/highway2.mp4')

  def generate_frames():
    while True:
      # Read a frame from the webcam
      ret, frame = cap.read()

      # Encode the frame as a JPEG image
      ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 15])

      # Convert the image to a bytes object
      img_bytes = buffer.tobytes()

      # Encode the bytes object as a base64 string
      img_base64 = base64.b64encode(img_bytes).decode()

      # Yield the base64 string as a 'message' event
      yield 'event: message\ndata: ' + img_base64 + '\n\n'

  # Return the streaming response
  return Response(generate_frames(), mimetype='text/event-stream')


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080 )  
  
