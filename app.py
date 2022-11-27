from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods =["GET", "POST"])
def output():
    if request.method == "POST":
      df = pd.DataFrame(columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'])
      lst = []
      time = request.form.get("Time")
      lst.append(time)
      for i in range(1,29):
            s = 'V'+str(i)
            x = request.form.get(s)
            lst.append(x)
      amount = request.form.get("Amount")
      lst.append(amount)
      df.loc[len(df)] = lst
      filename = 'trainedModel.pth'
      loaded_model = pickle.load(open(filename, 'rb'))
      prediction = loaded_model.predict(df)
      if prediction[0]:
            output = "It is Fraud"
      else:
            output = "It's not Fraud"
      lst.append(output)
      return render_template("output.html", result = lst)

if __name__ == "__main__":
    app.run(debug=True)