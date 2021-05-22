import dice_ml
import flask
import joblib
import shap
from flask import request
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import jsonify
from flask_cors import CORS
import firebase

from com.shariful.prediction.scrubbing.scrub import addMissingColumn

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

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
    trainer.train("../../../data/dictionary/bangla.yml")
    trainer.train("../../../data/dictionary/english.yml")
    # trainer.train("chatterbot.corpus.bangla.emotions")
    # trainer.train("chatterbot.corpus.bangla.greetings")
    # trainer.train("chatterbot.corpus.english.conversations")
    # trainer.train("chatterbot.corpus.english.emotion")
    # trainer.train("chatterbot.corpus.english.greetings")
    # Get a response to an input statement
    response = chatbot.get_response(question)
    text = response.text
    return jsonify(text)

def generate_message(model,df):
    d = dice_ml.Data(dataframe=df, continuous_features=[
        'age',
        # 'mother_age_when_baby_was_born',
        'age_at_first_conception',
        'wt'
    ], outcome_name='outcome_pregnancy')
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    e1 = exp.generate_counterfactuals(df, total_CFs=2, desired_class="opposite")
    e1.visualize_as_dataframe(show_only_changes=True)
    data2 = e1.cf_examples_list[0].final_cfs_df_sparse
    # data=data.append(data2)
    message=""

    if( not data2.empty):
        nunique = data2.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        data2 = data2.drop(cols_to_drop, axis=1)
        column_name = data2.columns.tolist()
        message = "You are identified as risk group for your pregnancy. The reason of the risk is identified as "
        for column in column_name:
            print(column)
            column = column.replace("_", " ")
            if (column == "wt"):
                column = column.replace("wt", "weight")
            if (column == "during pregnancy"):
                column = column.replace("during pregnancy", " status of  antenatal care visit during pregnancy")
            if (column == "during lactation"):
                column = column.replace("during lactation", " status of  antenatal care visit during lactation")
            if ("anm" in column):
                column = column.replace("anm", "antenatal care")
            if ("is" in column):
                column = column.replace("is", "status of")

            message += column + ", "
            print(column)
        new = " and"
        message = new.join(message.rsplit(",", 1))
        print(message)
    else:
        message="You are identified as risk group for your pregnancy. please contact your nearest hospital"

    return message


@app.route('/predict', methods=['POST'])
def givePrediction():
    content = request.json
    df = pd.json_normalize(content)
    df = addMissingColumn(df)
    loaded_model = joblib.load('../../../data/saved_model/model.bpk', )
    res = loaded_model.predict(df)
    print(res[0])
    message=""
    if (res[0] == 1):
        message=generate_message(loaded_model,df)
        firebase.sendPush(title="risk alert", msg=message)
    else:
        message= "It is advisable to visit Antenatal care each three month during pregnancy."
        firebase.sendPush(title="info", msg=message)
    return jsonify(message)

@app.route('/noti')
def sentNoti():
    # firebase.send_all()
    firebase.sendPush(title="sample", msg="ok")
    return jsonify("success")


app.run()
