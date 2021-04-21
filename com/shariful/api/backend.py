import flask
from flask import request
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def get_rows(id):
    data_path = '../../../data/test/test.csv'
    df = pd.read_csv(data_path, delimiter='|')
    temp=df.loc[df['id'].isin([int(id)])]
    js=temp.to_json(orient='records')
    return js

def add_rows(id,name,date,time,hospital):
    data_path = '../../../data/test/test.csv'
    df = pd.read_csv(data_path, delimiter='|')
    index=df.index[-1]
    df2=pd.DataFrame({
        'index':[index+1],
        'id':[id],
        'name':[name],
        'date':[date],
        'time':[time],
        'hospital':[hospital]
    })
    df2.to_csv(data_path,mode='a',index=False,index_label='index',header=False,sep="|")
    df.append(df2)
    # df.to_csv(data_path, index=df.size + 1, mode='w', header=False)



@app.route('/', methods=['GET'])
def home():
    return "home"

@app.route('/get-appointment')
def get_appointment():
    id = request.args['id']
    return get_rows(id)

@app.route('/add-appointment')
def add_appointment():
    id = request.args['id']
    name = request.args['name']
    date= request.args['date']
    time = request.args['time']
    hospital = request.args['hospital']
    add_rows(id,name,date,time,hospital)
    return get_rows(id)

@app.route('/ask')
def askQuestion():
    chatbot = ChatBot('Helper Bot')
    question=request.args['question']
    # Create a new trainer for the chatbot
    trainer = ChatterBotCorpusTrainer(chatbot)

    trainer.train("chatterbot.corpus.bangla")
    trainer.train("chatterbot.corpus.english")
    trainer.train("chatterbot.corpus.custom")

    # Get a response to an input statement
    response=chatbot.get_response(question)
    text=response.text
    return jsonify(text)




app.run()
