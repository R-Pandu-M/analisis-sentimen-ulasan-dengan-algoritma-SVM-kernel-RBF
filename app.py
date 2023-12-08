from flask import Flask, render_template, request, send_from_directory, jsonify
import pickle
import nltk
import sqlite3
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
cvt = pickle.load(open("count_vect.pkl", "rb"))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'

def connect_to_database():
    connection = sqlite3.connect('mydatabase.db')
    cursor = connection.cursor()
    return connection, cursor

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index_prediksi')
def index_prediksi():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/wordcloud')
def wordcloud():
    return render_template('wordcloud.html')

@app.route('/timeseries')
def timeseries():
    return render_template('timeseries.html')

@app.route('/api/data/<path:table_name>', methods=['GET', 'POST'])
def get_data_by_table(table_name):
    try:
        connection, cursor = connect_to_database()
        cursor.execute(f'SELECT * FROM {table_name} LIMIT 5')
        data = cursor.fetchall()
        

        response = []
        for row in data:
            item = {
                'Review' : row[0],
                'Label Predicted' : row[4],
                'Actual Label'  : row[5]
            }
            response.append(item)
        
        connection.close()
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}) 

@app.route('/get_data_ampel')
def get_data_ampel():
    try:
        connection, cursor = connect_to_database()
        cursor.execute('''
            SELECT "Actual Label", COUNT(*) as count
            FROM ampel
            GROUP BY "Actual Label"
        ''')
        data = cursor.fetchall()
        connection.close()

        # Proses data jika diperlukan
        # Contoh: Membuat format data sesuai dengan kebutuhan Anda
        formatted_data = [{'Actual Label': row[0], 'count': row[1]} for row in data]

        return jsonify(formatted_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/static/<path:filename>')
def images(filename):
    return send_from_directory('static', filename)

@app.route('/prediksi', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['ulasan'].lower()

        stop_words = stopwords.words('indonesian')
        stop_words.extend(['tempat','wisata','religi','sunan','ziarah','ziaroh',
                  'ibrahim','gunung','jati','kalijaga','kudus','muria','surabaya',
                  'translated','by','gogle','more','asli','original','diterjemahkan','jiaroh', 'makam',
                   'wali', 'songo', 'walisongo','tuban','cirebon','google'])
        tokens = nltk.word_tokenize(str(text))
        filtered = []
        for w in tokens:
            if w not in stop_words:
                filtered.append(w)
        hasil = ' '.join(filtered)

        factory_stem = StemmerFactory()
        stemmer = factory_stem.create_stemmer()
        text_fix = stemmer.stem(hasil)

        text_cv = cvt.transform([text_fix])
        text_cvt = text_cv.toarray()
        prediction = model.predict(text_cvt)

    return render_template('result.html', prediction_text="{}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)