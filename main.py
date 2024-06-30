# main.py
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import os
import base64
import cv2
import pandas as pd
import numpy as np
import imutils
from flask import send_file
from werkzeug.utils import secure_filename
import shutil
import json
import csv
import subprocess
import mysql.connector
import hashlib
from datetime import datetime
from datetime import date
import datetime
import time
from random import seed
from random import randint
from PIL import Image
import stepic
import urllib.parse
from urllib.request import urlopen
import webbrowser

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="fisherman"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'


@app.route('/')
def index():
        
    return render_template('web/index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM fm_admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('web/login.html',msg=msg)

@app.route('/login_auth', methods=['GET', 'POST'])
def login_auth():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM fm_authority WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
           
            
            session['username'] = uname

            ff=open("static/auth.txt","w")
            ff.write(uname)
            ff.close()
            ###
            now = datetime.datetime.now()
            sdate=now.strftime("%Y-%m-%d")
            edate=sdate            
            st="1"           
            #lat1="12.738352"
            #lon1="80.474805"
            lat1="7.745089"
            lon1="77.525925"
            loc=lat1+","+lon1
            print(loc)

            shutil.copy("static/weather.csv","static/weather_data.csv")
            responsetext="https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"+loc+"/"+sdate+"/"+edate+"?unitGroup=metric&include=days&key=8STCAMRSRTEZ77JA2XRP7FMNC&contentType=csv"

            cdata=[]
            data1 = pd.read_csv(responsetext, header=0)
            for ss in data1.values:
                dt=[]
                print(ss[0])
                
                with open("static/weather_data.csv",'a',newline='') as outfile:
                    writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
                    writer.writerow(ss)

                cdata.append(dt)
            ##
            return redirect(url_for('auth_home'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('web/login_auth.html',msg=msg)

@app.route('/login_fm', methods=['GET', 'POST'])
def login_fm():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM fm_fisher WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            ff=open("static/uname.txt","w")
            ff.write(uname)
            ff.close()
            ###
            now = datetime.datetime.now()
            sdate=now.strftime("%Y-%m-%d")
            edate=sdate
            #if request.method=='POST':
            st="1"
            
            #t1=request.form["t1"]
            #sdate=request.form["sdate"]
            #edate=request.form["edate"]

            #t11=t1.split("),")
            #t2=t11[0].split("(")
            #t3=t2[1].split(",")

            #lat=t3[0].strip()
            #lon=t3[1].strip()

            
            #lat1=lat[0:8]
            #lon1=lon[0:8]
            lat1="7.745089"
            lon1="77.525925"
            loc=lat1+","+lon1
            print(loc)

            shutil.copy("static/weather.csv","static/weather_data.csv")
            responsetext="https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"+loc+"/"+sdate+"/"+edate+"?unitGroup=metric&include=days&key=8STCAMRSRTEZ77JA2XRP7FMNC&contentType=csv"

            cdata=[]
            data1 = pd.read_csv(responsetext, header=0)
            for ss in data1.values:
                dt=[]
                print(ss[0])
                
                with open("static/weather_data.csv",'a',newline='') as outfile:
                    writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
                    writer.writerow(ss)

                cdata.append(dt)
            ##
            return redirect(url_for('fm_home'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('web/login_fm.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    fn=""
    email=""
    mess=""
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM fm_fisher")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1
    
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    if request.method=='POST':
        
        name=request.form['name']
        gender=request.form['gender']
        dob=request.form['dob']
        address=request.form['address']
        mobile=request.form['mobile']
        email=request.form['email']
        pass1=request.form['pass']
        
        file = request.files['file']
        file2 = request.files['file2']
        
        uname="F"+str(maxid)
        #rn=randint(1000,9999)
        #pass1=str(rn)

        filename=file.filename
        aadhar="A"+str(maxid)+filename
        file.save(os.path.join("static/upload", aadhar))

        filename2=file2.filename
        fcard="F"+str(maxid)+filename2
        file2.save(os.path.join("static/upload", fcard))


        mess="Dear "+name+", Fisherman ID:"+uname+", Password:"+pass1

        
        sql = "INSERT INTO fm_fisher(id,name,gender,dob,address,mobile,email,aadhar,fisher_card,uname,pass,rdate,approve_st) VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)"
        val = (maxid,name,gender,dob,address,mobile,email,aadhar,fcard,uname,pass1,rdate,'0')
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "Registered Success")
        msg="success"


    return render_template('web/register.html',msg=msg,act=act,mess=mess,email=email)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    if 'username' in session:
        uname = session['username']

    ff=open("static/det.txt","w")
    ff.write("")
    ff.close()

  
                                        
    return render_template('admin.html',msg=msg)
@app.route('/process1', methods=['GET', 'POST'])
def process1():
    uname=""
    msg=""
   
    path='dataset/'
    df=pd.read_csv('static/indian-ocean-data.csv')
    
    dat1=df.head(100)
    ##
    data1=[]
    for ss1 in dat1.values:
        data1.append(ss1)
    

    
    return render_template('process1.html',data1=data1)
##LSTM
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

def train(train_iter, dev_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()

    if args.Adam is True:
        print("Adam Training......")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
    elif args.SGD is True:
        print("SGD Training.......")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay,
                                    momentum=args.momentum_value)
    elif args.Adadelta is True:
        print("Adadelta Training.......")
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)

    steps = 0
    model_count = 0
    best_accuracy = Best_Result()
    model.train()
    for epoch in range(1, args.epochs+1):
        steps = 0
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        for batch in train_iter:
            feature, target = batch.text, batch.label.data.sub_(1)
            if args.cuda is True:
                feature, target = feature.cuda(), target.cuda()

            target = autograd.Variable(target)  # question 1
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm_(model.parameters(), max_norm=args.init_clip_max_norm)
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                            train_size,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                print("\nDev  Accuracy: ", end="")
                eval(dev_iter, model, args, best_accuracy, epoch, test=False)
                print("Test Accuracy: ", end="")
                eval(test_iter, model, args, best_accuracy, epoch, test=True)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model.state_dict(), save_path)
                if os.path.isfile(save_path) and args.rm_model is True:
                    os.remove(save_path)
                model_count += 1
    return model_count


def eval(data_iter, model, args, best_accuracy, epoch, test=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        target.data.sub_(1)
        if args.cuda is True:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.item()/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss, accuracy, corrects, size))
    if test is False:
        if accuracy >= best_accuracy.best_dev_accuracy:
            best_accuracy.best_dev_accuracy = accuracy
            best_accuracy.best_epoch = epoch
            best_accuracy.best_test = True
    if test is True and best_accuracy.best_test is True:
        best_accuracy.accuracy = accuracy

    if test is True:
        print("The Current Best Dev Accuracy: {:.4f}, and Test Accuracy is :{:.4f}, locate on {} epoch.\n".format(
            best_accuracy.best_dev_accuracy, best_accuracy.accuracy, best_accuracy.best_epoch))
    if test is True:
        best_accuracy.best_test = False
        
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model


@app.route('/add_auth', methods=['GET', 'POST'])
def add_auth():
    msg=""
    mess=""
    email=""
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM fm_authority")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1
    
        
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        location=request.form['location']

        uname="M"+str(maxid)
        rn=randint(1000,9999)
        pass1=str(rn)

        mess="Dear "+name+", Maritime Authority ID:"+uname+", Password:"+pass1
        
        sql = "INSERT INTO fm_authority(id,name,mobile,email,location,uname,pass) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,location,uname,pass1)
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "Registered Success")
        msg="success"
        #return redirect(url_for('add_rto',act='1'))

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from fm_authority where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_auth'))
        

    mycursor.execute("SELECT * FROM fm_authority")
    data = mycursor.fetchall()

    return render_template('add_auth.html',msg=msg,act=act,data=data,mess=mess,email=email)


@app.route('/auth_home', methods=['GET', 'POST'])
def auth_home():
    msg=""
    uname=""
    data4=[]
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']

    ff=open("static/auth.txt","r")
    uname=ff.read()
    ff.close()

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_authority where uname=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM fm_fisher")
    data1 = mycursor.fetchall()

    if act=="ok":
        fid=request.args.get("fid")
        mycursor.execute("update fm_fisher set approve_st=1 where id=%s",(fid,))
        mydb.commit()
        msg="yes"
  
    ##
    data11 = pd.read_csv("static/weather_data.csv", header=0)
    r=0
    for ss in data11.values:
        dt=[]

        if ss[13]=="rain":
            if float(ss[12])>0 and float(ss[11])>0:
                r+=1
                
        dt.append(ss[1])
        dt.append(ss[2])
        dt.append(ss[3])
        dt.append(ss[4])
        dt.append(ss[9])
        dt.append(ss[29])
        dt.append(ss[30])
        dt.append(ss[31])
        data4.append(dt)
        
    return render_template('auth_home.html',msg=msg,act=act,data=data,data1=data1,data4=data4)


@app.route('/auth_request', methods=['GET', 'POST'])
def auth_request():
    msg=""
    uname=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("static/auth.txt","r")
    uname=ff.read()
    ff.close()

    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_authority where uname=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM fm_trip order by id desc")
    data1 = mycursor.fetchall()

    if act=="ok":
        tid=request.args.get("tid")
        mycursor.execute("update fm_trip set request_st=1 where id=%s",(tid,))
        mydb.commit()
        msg="ok"
    if act=="no":
        tid=request.args.get("tid")
        mycursor.execute("update fm_trip set request_st=2 where id=%s",(tid,))
        mydb.commit()
        msg="no"

    if act=="trip":
        subprocess.call(["myfile2.bat"])
                                        
    return render_template('auth_request.html',msg=msg,act=act,data=data,data1=data1)

@app.route('/fm_home', methods=['GET', 'POST'])
def fm_home():
    msg=""
    uname=""
    data4=[]
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("static/uname.txt","r")
    uname=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_fisher where uname=%s",(uname,))
    rs = mycursor.fetchone()

    
    ##
    data1 = pd.read_csv("static/weather_data.csv", header=0)
    r=0
    for ss in data1.values:
        dt=[]

        if ss[13]=="rain":
            if float(ss[12])>0 and float(ss[11])>0:
                r+=1
                
        dt.append(ss[1])
        dt.append(ss[2])
        dt.append(ss[3])
        dt.append(ss[4])
        dt.append(ss[9])
        dt.append(ss[29])
        dt.append(ss[30])
        dt.append(ss[31])
        data4.append(dt)
        
        
        
                                        
    return render_template('fm_home.html',msg=msg,act=act,rs=rs,data4=data4)


@app.route('/fm_trip', methods=['GET', 'POST'])
def fm_trip():
    msg=""
    uname=""
    data1=[]
    s1=""
    tid=""
    fid=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("static/uname.txt","r")
    uname=ff.read()
    ff.close()

    
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_fisher where uname=%s",(uname,))
    rs = mycursor.fetchone()
    fid=str(rs[0])

    if request.method=='POST':        
        sdate=request.form['sdate']
        sh=request.form['sh']
        sm=request.form['sm']

        sd=sdate.split("-")
        sdate1=sd[2]+"-"+sd[1]+"-"+sd[0]
        stime=sh+":"+sm

        mycursor.execute("SELECT max(id)+1 FROM fm_trip")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO fm_trip(id,fisher_id,sdate,stime,edate,etime,request_st,trip_st) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,uname,sdate1,stime,'','','0','0')
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "Registered Success")
        msg="success"

    mycursor.execute("SELECT count(*) FROM fm_trip where fisher_id=%s order by id desc",(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        s1="1"
        mycursor.execute("SELECT * FROM fm_trip where fisher_id=%s order by id desc",(uname,))
        data1 = mycursor.fetchall()

    if act=="trip":
        tid=request.args.get("tid")
        mycursor.execute("update fm_trip set trip_st=1 where id=%s",(tid,))
        mydb.commit()
        msg="trip"

        ff=open("fid.txt","w")
        ff.write(fid)
        ff.close()

        ff=open("tid.txt","w")
        ff.write(tid)
        ff.close()

        mycursor.execute("update fm_admin set tid=%s",(tid,))
        mydb.commit()
        
        subprocess.call(["myfile.bat"])

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    rtime=now.strftime("%H-%M")
    if act=="end":
        ff=open("tid.txt","r")
        tid=ff.read()
        ff.close()
        mycursor.execute("update fm_trip set edate=%s,etime=%s,trip_st=2 where id=%s",(rdate,rtime,tid))
        mydb.commit()
        return redirect(url_for('fm_trip'))
        
    return render_template('fm_trip.html',msg=msg,act=act,rs=rs,data1=data1,s1=s1,fid=fid,tid=tid)

@app.route('/fm_history', methods=['GET', 'POST'])
def fm_history():
    msg=""
    uname=""
    data1=[]
    s1=""
    tid=request.args.get("tid")
    fid=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("static/uname.txt","r")
    uname=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_fisher where uname=%s",(uname,))
    rs = mycursor.fetchone()
    fid=str(rs[0])

    mycursor.execute("SELECT * FROM fm_history where tid=%s",(tid,))
    data1 = mycursor.fetchall()

    return render_template('fm_history.html',msg=msg,act=act,rs=rs,data1=data1)

@app.route('/auth_history', methods=['GET', 'POST'])
def auth_history():
    msg=""
    uname=""
    data1=[]
    s1=""
    tid=request.args.get("tid")
    fid=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("static/uname.txt","r")
    uname=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_authority where uname=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM fm_history where tid=%s",(tid,))
    data1 = mycursor.fetchall()

    return render_template('auth_history.html',msg=msg,act=act,data=data,data1=data1)


@app.route('/single', methods=['GET', 'POST'])
def single():
    msg=""

    ff=open("fid.txt","r")
    fid=ff.read()
    ff.close()
    mycursor.execute("SELECT * FROM fm_fisher where id=%s",(fid,))
    rs = mycursor.fetchone()

    ff=open("tid.txt","r")
    tid=ff.read()
    ff.close()

    return render_template('single.html',msg=msg,act=act,rs=rs,data1=data1,s1=s1,fid=fid,tid=tid)

@app.route('/multi', methods=['GET', 'POST'])
def multi():
    msg=""
    uname=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("static/auth.txt","r")
    uname=ff.read()
    ff.close()

        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_authority where uname=%s",(uname,))
    data = mycursor.fetchone()


    return render_template('multi.html',msg=msg,act=act,data=data)



@app.route('/fm_travel', methods=['GET', 'POST'])
def fm_travel():
    msg=""
    uname=""
    data1=[]
    s1=""
    tid=request.args.get("tid")
    fid=""
    act=request.args.get("act")
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM fm_fisher where uname=%s",(uname,))
    rs = mycursor.fetchone()
    fid=str(rs[0])

    return render_template('fm_travel.html',msg=msg,act=act,rs=rs,fid=fid,tid=tid)








@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
