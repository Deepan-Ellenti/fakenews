from flask import Flask, render_template,jsonify, request
import pickle

app = Flask(__name__)
loaded_model = pickle.load(open('model.pkl', 'rb'))
tfvect=pickle.load(open('tfidfvect2.pkl','rb'))
def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)