
import flask
import joblib
from flask import request
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import jsonify
from flask_cors import CORS
import firebase
import service

from com.shariful.prediction.scrubbing.scrub import addMissingColumn

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

appointment_path= '../../../data/db/appointment.csv'
user_path= '../../../data/db/user.csv'

@app.route('/', methods=['GET'])
def home():
    return "home"

@app.route('/get-appointment')
def get_appointment_by_id():
    id = request.args['id']
    return service.get_appointment(id)

@app.route('/get-all-appointment')
def get_all():
    return service.get_all_appointment()

@app.route('/get-all-user')
def ret_all_user():
    return service.get_all_user()

@app.route('/add-appointment')
def add_new_appointment():
    id = request.args['id']
    name = request.args['name']
    date = request.args['date']
    time = request.args['time']
    hospital = request.args['hospital']
    service.add_appointment(id, name, date, time, hospital)
    return service.get_appointment(id)


    # df.to_csv(data_path, index=df.size + 1, mode='w', header=False)
@app.route('/register')
def register():
    id = request.args['id']
    name = request.args['name']
    phone = request.args['phone']
    service.add_user( id,name, phone)
    return service.get_user(phone)

@app.route('/device-register',methods=['POST'])
def device_register():
    content = request.json
    id = content['id']
    device_id = content['device_id']
    token = content['token']
    service.add_device( id,device_id, token)
    return jsonify("added successfully")


@app.route('/login')
def login():
    phone = request.args['phone']
    js=service.get_user(phone)
    return js
    # if()




@app.route('/ask')
def askQuestion():
    chatbot = ChatBot('Helper Bot')
    question = request.args['question']
    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train("../../../data/dictionary/bangla.yml")
    trainer.train("../../../data/dictionary/english.yml")
    response = chatbot.get_response(question)
    text = response.text
    return jsonify(text)



@app.route('/predict', methods=['POST'])
def givePrediction():
    content = request.json
    id=content['id']
    df = pd.json_normalize(content)
    df = addMissingColumn(df)
    loaded_model = joblib.load('../../../data/saved_model/model.bpk', )
    res = loaded_model.predict(df)
    print(res[0])
    message=""
    if (res[0] == 1):
        message=service.generate_message(loaded_model,df)
        firebase.sendPush(id=id,title="risk alert", msg=message)
    else:
        message= "It is advisable to visit Antenatal care each three month during pregnancy."
        firebase.sendPush(id=id,title="info", msg=message)
    return jsonify(message)

@app.route('/noti')
def sentNoti():
    # firebase.send_all()
    firebase.sendPush(id=id,title="sample", msg="ok")
    return jsonify("success")


app.run()
