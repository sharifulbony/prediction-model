import dice_ml
import flask
import joblib
import shap
from flask import request
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import jsonify

from com.shariful.prediction.scrubbing.scrub import addMissingColumn

app = flask.Flask(__name__)
app.config["DEBUG"] = True

appointment_path= '../../../data/db/appointment.csv'
user_path= '../../../data/db/user.csv'

@app.route('/', methods=['GET'])
def home():
    return "home"

def get_appointment(id):
    df = pd.read_csv(appointment_path, delimiter='|')
    temp = df.loc[df['id'].isin([int(id)])]
    js = temp.to_json(orient='records')
    return js

def get_all_appointment():
    df = pd.read_csv(appointment_path, delimiter='|')
    js = df.to_json(orient='records')
    return js

@app.route('/get-appointment')
def get_appointment_by_id():
    id = request.args['id']
    return get_appointment(id)

@app.route('/get-all-appointment')
def get_all():
    return get_all_appointment()


def add_appointment(id, name, date, time, hospital):
    df = pd.read_csv(appointment_path, delimiter='|')
    index = df.index[-1]
    df2 = pd.DataFrame({
        'index': [index + 1],
        'id': [id],
        'name': [name],
        'date': [date],
        'time': [time],
        'hospital': [hospital]
    })
    df2.to_csv(appointment_path, mode='a', index=False, index_label='index', header=False, sep="|")
    df.append(df2)
    # df.to_csv(data_path, index=df.size + 1, mode='w', header=False)


@app.route('/add-appointment')
def add_new_appointment():
    id = request.args['id']
    name = request.args['name']
    date = request.args['date']
    time = request.args['time']
    hospital = request.args['hospital']
    add_appointment(id, name, date, time, hospital)
    return get_appointment(id)

def add_user( id,name, phone ):
    df = pd.read_csv(user_path, delimiter='|')
    index = df.index[-1]
    df2 = pd.DataFrame({
        'index': [index + 1],
        'id': [id],
        'name' : [name],
        'phone': [phone],
    })
    df2.to_csv(user_path, mode='a', index=False, index_label='index', header=False, sep="|")
    df.append(df2)
    # df.to_csv(data_path, index=df.size + 1, mode='w', header=False)
@app.route('/register')
def register():
    id = request.args['id']
    name = request.args['name']
    phone = request.args['phone']
    add_user( id,name, phone)
    return get_user(phone)

def get_user(phone):
    df = pd.read_csv(user_path, delimiter='|')
    temp = df.loc[df['phone'].isin([int(phone)])]
    # if(temp==)
    js = temp.to_json(orient='records')
    return js


@app.route('/login')
def login():
    phone = request.args['phone']
    js=get_user(phone)
    return js
    # if()




@app.route('/ask')
def askQuestion():
    chatbot = ChatBot('Helper Bot')
    question = request.args['question']
    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train("chatterbot.corpus.bangla.emotions")
    trainer.train("chatterbot.corpus.bangla.greetings")
    trainer.train("chatterbot.corpus.english.conversations")
    trainer.train("chatterbot.corpus.english.emotion")
    trainer.train("chatterbot.corpus.english.greetings")
    # Get a response to an input statement
    response = chatbot.get_response(question)
    text = response.text
    return jsonify(text)


@app.route('/predict', methods=['POST'])
def givePrediction():
    content = request.json
    df = pd.json_normalize(content)
    df = addMissingColumn(df)
    loaded_model = joblib.load('../../../data/saved_model/model.bpk', )

    # d = dice_ml.Data(dataframe=df, continuous_features=[
    #     'age',
    #     'mother_age_when_baby_was_born',
    #     'pregnant_month'
    # ], outcome_name='outcome_pregnancy')
    # m = dice_ml.Model(model=loaded_model, backend="sklearn")
    # exp = dice_ml.Dice(d, m, method="random")
    # e1 = exp.generate_counterfactuals(df, total_CFs=2, desired_class="opposite")
    # e1.visualize_as_dataframe(show_only_changes=True)

    res = loaded_model.predict(df)

    # return jsonify(shap_values.tolist())
    val="please contact nearest hospital"
    if(res==1):
        val = "You are in the risk zone. please book appointment to nearest hospital"
    elif(res==0):
        val = "It is advisable to visit Antenatal care each three month"
    # return pd.Series(res).to_json(orient='records')
    return jsonify(val)



app.run()
