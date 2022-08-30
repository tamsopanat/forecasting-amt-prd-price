import requests
from flask import Flask, request,render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from sklearn.linear_model import LinearRegression

current_year = date.today().year

app = Flask(__name__)
app.config['DEBUG'] = True

cvt_eng = {'ข้าวนาปี' : 'NAPEE',
           'ข้าวนาปรัง' : 'NAPRUNG',
           'ข้าวโพดเลี้ยงสัตว์' : 'CORN',
           'อ้อยโรงงาน' : 'AOI'}
cvt_mth = {'มกราคม' : 1,
           'กุมภาพันธ์' : 2,
           'มีนาคม' : 3,
           'เมษายน' : 4,
           'พฤษภาคม' : 5,
           'มิถุนายน' : 6,
           'กรกฎาคม' : 7,
           'สิงหาคม' : 8,
           'กันยายน' : 9,
           'ตุลาคม' : 10,
           'พฤศจิกายน' : 11,
           'ธันวาคม' : 12}

def create_model_pred(prd,prv,mth):
  # Predict AOP
  data = pd.read_csv("C:/Users/HP/Desktop/RS_FINAL/amt_of_prd_per_area/all_for_"+ cvt_eng[prd].lower()+".csv")
  data.index = data['year']
  del data['year']
  data = data[data.province == prv]
  data_corr = data[['humid_AM', 'humid_PM','Amount_of_product_per_area']]
  corr = data_corr.corr()['Amount_of_product_per_area']
  exogenous_features =  list(corr[(abs(corr)>=0.8) & (corr.index != 'Amount_of_product_per_area')].index)
  df = data[exogenous_features +  ['temp_AM', 'rain_AM', 'humid_AM', 'temp_PM', 'rain_PM', 'Amount_of_product_per_area']]

  # normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  df['Amount_of_product_per_area']= scaler.fit_transform(np.array(df['Amount_of_product_per_area']).reshape(-1,1))
  print(df.shape)
  reg = LinearRegression().fit(df.drop(columns = ['Amount_of_product_per_area']), df['Amount_of_product_per_area'])

  # Predict
  fore_var = pd.read_csv("C:/Users/HP/Desktop/RS_FINAL/For_Forecast/all_forecast_aop.csv")
  pred_var = fore_var[(fore_var['province'] == prv) & (fore_var['year'] == current_year)]
  pred_var = pred_var[exogenous_features +  ['temp_AM', 'rain_AM', 'humid_AM', 'temp_PM', 'rain_PM']]
  predict_aop = scaler.inverse_transform(reg.predict(pred_var).reshape(-1,1))

  # Predict Price
  file_prd = "C:/Users/HP/Desktop/RS_FINAL/price/ALL_FOR_" + cvt_eng[prd] + "_PRICE.csv"
  data = pd.read_csv(file_prd)
  data['year'] = data['year'] -543
  data.year = data.year.astype(str)
  data.month = data.month.astype(str)
  data['Date'] = data['year'] + '-' + data['month']
  data.index = pd.to_datetime(data.Date, format='%Y-%m')
  del data['Date']
  data = data[data['price'] != 0]
  corr = data.corr(method="spearman")['price']
  exogenous_features = list(corr[(abs(corr)>=0.2) & (corr.index != 'price')].index)
  df = data[exogenous_features + ['price']]
  # normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  df['price']= scaler.fit_transform(np.array(df['price']).reshape(-1,1))
  # Model
  reg2 = LinearRegression().fit(df.drop(columns = ['price']), df['price'])
  # Predict
  fore_var = pd.read_csv("C:/Users/HP/Desktop/RS_FINAL/For_Forecast/all_forecast_price_new.csv")
  pred_var = fore_var[(fore_var['year'] == current_year) & (fore_var['month'] == cvt_mth[mth])]
  pred_var = pred_var[exogenous_features]
  predict_price = scaler.inverse_transform(reg2.predict(pred_var).reshape(-1,1))
  return float(predict_aop), float(predict_price)

cvt_kg = {'ข้าวนาปี' : 1000,
           'ข้าวนาปรัง' : 1000,
           'ข้าวโพดเลี้ยงสัตว์' : 1,
           'อ้อยโรงงาน' : 1000}

@app.route('/', methods = ["GET","POST"])
def index():
  if(request.method == "POST"):
    prd = request.form['product']
    prv = request.form['province']
    area = request.form['area']
    mth = request.form['month']
    if prd == "อ้อยโรงงาน":
      if cvt_mth[mth] in np.arange(4,12):
        return render_template("mainpage copy.html",year = current_year, prediction = "ในช่วงเดือนเมษายน ถึงเดือนพฤศจิกายนจะไม่มีการเก็บเกี่ยวอ้อย")
      else:
        aop, price = create_model_pred(prd,prv,mth)
        price = price/cvt_kg[prd]
        profit = aop*float(area)*price
        return render_template("mainpage copy.html",year = current_year, prediction = f'''เดือน{mth} {prd}ของจังหวัด{prv}จะมีปริมาณผลผลิต {round(aop,2)} กิโลกรัมต่อไร่ 
        และจะมีราคาประมาณ {round(price,2)} บาทต่อกิโลกรัม 
        พื้นที่ {area} ไร่ จะขายได้ {round(profit,2)} บาท''')
    else:
      aop, price = create_model_pred(prd,prv,mth)
      price = price/cvt_kg[prd]
      profit = aop*float(area)*price
      return render_template("mainpage copy.html",year = current_year, prediction = f'''เดือน{mth} {prd}ของจังหวัด{prv}จะมีปริมาณผลผลิต {round(aop,2)} กิโลกรัมต่อไร่ 
      และจะมีราคาประมาณ {round(price,2)} บาทต่อกิโลกรัม 
      พื้นที่ {area} ไร่ จะขายอ้อยได้ {round(profit,2)} บาท''')
  else:
    return render_template("mainpage copy.html", year = current_year, prediction = "")


if __name__ == '__main__':
    app.run(debug = True)