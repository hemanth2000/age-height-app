from flask import Flask,render_template,request

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

from joblib import load

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello_world():

    request_type_str=request.method
    if request_type_str=='GET':
        return render_template("index.html",href="static/base_pic.svg")  
    else:
        text=request.form['text']
        random_string=uuid.uuid4().hex
        path='static/'+random_string+'.svg'
        input_np_arr=floats_string_to_np_arr(text)
        model=load("model.joblib")
        make_picture('AgesAndHeights.pkl',model,input_np_arr,path)

        return render_template("index.html",href=path)
    
def is_float(s):
  
  try:
    float(s)
    return True
  
  except:
    return False

def floats_string_to_np_arr(floats_str):

  floats=[float(x) for x in floats_str.split(',') if is_float(x)]

  return np.array(floats).reshape(-1,1)

def make_picture(training_data_filename,model,new_inp_np_arr,output_file):

  data=pd.read_pickle(training_data_filename)
  data=data[data['Age']>0]
  ages=data['Age']
  heights=data['Height']

  x_new=np.arange(0,19,1).reshape(-1,1)
  preds=model.predict(x_new)

  fig=px.scatter(x=ages,y=heights,title="Height vs Age of people",labels={'x':"Age (years)",
                                                                      'y':"Height (inches)"})
  fig.add_trace(go.Scatter(x=x_new.reshape(-1),y=preds,mode="lines",name="Model"))

  new_preds=model.predict(new_inp_np_arr)

  fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(-1),y=new_preds,
                           name="New Outputs",mode="markers",
                           marker=dict(color='purple',size=10)))

  fig.write_image(output_file,width=800,engine='kaleido')
  fig.show()
