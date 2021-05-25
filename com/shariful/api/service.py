import dice_ml
import pandas as pd

appointment_path= '../../../data/db/appointment.csv'
user_path= '../../../data/db/user.csv'
device_path= '../../../data/db/device.csv'
def get_appointment(id):
    df = pd.read_csv(appointment_path, delimiter='|')
    temp = df.loc[df['id'].isin([int(id)])]
    js = temp.to_json(orient='records')
    return js

def get_all_appointment():
    df = pd.read_csv(appointment_path, delimiter='|')
    js = df.to_json(orient='records')
    return js


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

def get_user(phone):
    df = pd.read_csv(user_path, delimiter='|')
    temp = df.loc[df['phone'].isin([int(phone)])]
    # if(temp==)
    js = temp.to_json(orient='records')
    return js

def get_all_user():
    df = pd.read_csv(user_path, delimiter='|')
    js = df.to_json(orient='records')
    return js

def add_device( id,device_id, token ):
    df = pd.read_csv(device_path, delimiter='|')
    index = df.index[-1]
    df2 = pd.DataFrame({
        'index': [index + 1],
        'id': [id],
        'name' : [device_id],
        'phone': [token],
    })
    df2.to_csv(device_path, mode='a', index=False, index_label='index', header=False, sep="|")
    df.append(df2)

def get_device(id):
    df = pd.read_csv(device_path, delimiter='|')
    temp = df.loc[df['id'].isin([int(id)])]
    # if(temp==)
    return temp


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
