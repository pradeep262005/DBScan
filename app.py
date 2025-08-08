
from flask import Flask, render_template, request
from model import generate_dbscan_plot  

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    eps = float(request.form['eps'])
    min_samples = int(request.form['min_samples'])
    generate_dbscan_plot(eps, min_samples)

    return render_template('result.html',
                           eps=eps,
                           min_samples=min_samples,
                           image_path='static/dbscan_result.png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

