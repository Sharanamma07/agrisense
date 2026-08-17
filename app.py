import json, os, sqlite3, uuid
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, send_file
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MODEL_PATH=os.path.join('models','agrisense_model.tflite')
UPLOAD_DIR=os.path.join('static','uploads')
DATABASE='agrisense.db'
IMG_SIZE=(224,224)
app=Flask(__name__)
app.secret_key=os.environ.get('SECRET_KEY','change-this-agrisense-secret')
os.makedirs(UPLOAD_DIR,exist_ok=True)

interpreter=input_details=output_details=None
class_names=[]
disease_info={}

def db():
    c=sqlite3.connect(DATABASE); c.row_factory=sqlite3.Row; return c

def init_db():
    c=db()
    c.execute('CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT UNIQUE NOT NULL,password TEXT NOT NULL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,image_path TEXT,disease TEXT NOT NULL,confidence REAL NOT NULL,description TEXT,treatment TEXT,prevention TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS crops(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,name TEXT NOT NULL,planted_date TEXT,notes TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    c.commit(); c.close()
init_db()

try:
    interpreter=Interpreter(model_path=MODEL_PATH); interpreter.allocate_tensors()
    input_details=interpreter.get_input_details(); output_details=interpreter.get_output_details()
    print('[INFO] TFLite model loaded successfully.')
    print('[INFO] Input:',input_details[0]['shape'],'Output:',output_details[0]['shape'])
except Exception as e: print('[ERROR] TFLite model:',e)
try:
    with open(os.path.join('models','class_names.json'),encoding='utf-8') as f: class_names=json.load(f)
    print('[INFO] Number of classes:',len(class_names))
except Exception as e: print('[ERROR] class_names.json:',e)
if os.path.exists('disease_info.json'):
    try:
        with open('disease_info.json',encoding='utf-8') as f: disease_info=json.load(f)
        print('[INFO] Disease information loaded successfully.')
    except Exception as e: print('[ERROR] disease_info.json:',e)

def logged(): return 'user_id' in session

def info_for(label):
    x=disease_info.get(label,{})
    return {'description':x.get('description','No description available.'),'treatment':x.get('treatment','Consult a local agricultural extension office.'),'prevention':x.get('prevention','Remove infected material, keep foliage dry where possible, and monitor the crop regularly.'),'severity':x.get('severity','Moderate')}

def prep(path):
    img=Image.open(path).convert('RGB').resize(IMG_SIZE)
    a=np.asarray(img,dtype=np.float32); a=(a/127.5)-1.0
    return np.expand_dims(a,0)

@app.route('/')
def home(): return redirect(url_for('dashboard') if logged() else url_for('login'))

@app.route('/register',methods=['GET','POST'])
def register():
    if logged(): return redirect(url_for('dashboard'))
    if request.method=='POST':
        name=request.form.get('name','').strip(); email=request.form.get('email','').strip().lower(); pw=request.form.get('password',''); cpw=request.form.get('confirm_password','')
        if not name or not email or not pw: return render_template('register.html',error='Please fill in all fields.')
        if pw!=cpw: return render_template('register.html',error='Passwords do not match.')
        if len(pw)<6: return render_template('register.html',error='Password must contain at least 6 characters.')
        c=db()
        try: c.execute('INSERT INTO users(name,email,password) VALUES(?,?,?)',(name,email,generate_password_hash(pw))); c.commit()
        except sqlite3.IntegrityError: c.close(); return render_template('register.html',error='Email already registered.')
        c.close(); return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if logged(): return redirect(url_for('dashboard'))
    if request.method=='POST':
        email=request.form.get('email','').strip().lower(); pw=request.form.get('password',''); c=db(); u=c.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone(); c.close()
        if u and check_password_hash(u['password'],pw):
            session.clear(); session['user_id']=u['id']; session['user_name']=u['name']; session['user_email']=u['email']; return redirect(url_for('dashboard'))
        return render_template('login.html',error='Invalid email or password.')
    return render_template('login.html')

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not logged(): return redirect(url_for('login'))
    c=db(); total=c.execute('SELECT COUNT(*) n FROM scans WHERE user_id=?',(session['user_id'],)).fetchone()['n']; attention=c.execute("SELECT COUNT(*) n FROM scans WHERE user_id=? AND lower(disease) NOT LIKE '%healthy%'",(session['user_id'],)).fetchone()['n']; crops=c.execute('SELECT COUNT(*) n FROM crops WHERE user_id=?',(session['user_id'],)).fetchone()['n']; recent=c.execute('SELECT * FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 5',(session['user_id'],)).fetchall(); c.close()
    return render_template('dashboard.html',user_name=session['user_name'],total=total,attention=attention,crops=crops,recent=recent)

@app.route('/scan')
def scan():
    if not logged(): return redirect(url_for('login'))
    return render_template('scan.html',user_name=session['user_name'])

@app.route('/predict',methods=['POST'])
def predict():
    if not logged(): return jsonify(error='Please login before using AgriSense.'),401
    if interpreter is None: return jsonify(error='TFLite model is not loaded.'),503
    if 'image' not in request.files: return jsonify(error='No image uploaded.'),400
    f=request.files['image']
    if not f.filename: return jsonify(error='No image selected.'),400
    filename=uuid.uuid4().hex+'_'+os.path.basename(f.filename).replace(' ','_'); path=os.path.join(UPLOAD_DIR,filename)
    try:
        f.save(path); x=prep(path); interpreter.set_tensor(input_details[0]['index'],x); interpreter.invoke(); preds=interpreter.get_tensor(output_details[0]['index'])[0]
        idx=int(np.argmax(preds)); conf=float(preds[idx]); label=class_names[idx]; info=info_for(label)
        c=db(); cur=c.execute('INSERT INTO scans(user_id,image_path,disease,confidence,description,treatment,prevention) VALUES(?,?,?,?,?,?,?)',(session['user_id'],path.replace('\\','/'),label,round(conf*100,2),info['description'],info['treatment'],info['prevention'])); scan_id=cur.lastrowid; c.commit(); c.close()
        return jsonify(scan_id=scan_id,**{'class':label,'confidence':round(conf*100,2),'description':info['description'],'treatment':info['treatment'],'prevention':info['prevention'],'severity':info['severity'],'image_url':'/'+path.replace('\\','/')})
    except Exception as e:
        print('[ERROR] Prediction:',e); return jsonify(error=str(e)),500

@app.route('/history')
def history():
    if not logged(): return redirect(url_for('login'))
    c=db(); rows=c.execute('SELECT * FROM scans WHERE user_id=? ORDER BY id DESC',(session['user_id'],)).fetchall(); c.close(); return render_template('history.html',scans=rows,user_name=session['user_name'])

@app.route('/history/delete/<int:scan_id>',methods=['POST'])
def delete_scan(scan_id):
    if not logged(): return redirect(url_for('login'))
    c=db(); row=c.execute('SELECT image_path FROM scans WHERE id=? AND user_id=?',(scan_id,session['user_id'])).fetchone(); c.execute('DELETE FROM scans WHERE id=? AND user_id=?',(scan_id,session['user_id'])); c.commit(); c.close()
    if row and row['image_path']:
        try: os.remove(row['image_path'])
        except OSError: pass
    return redirect(url_for('history'))

@app.route('/crops',methods=['GET','POST'])
def crops():
    if not logged(): return redirect(url_for('login'))
    c=db()
    if request.method=='POST':
        name=request.form.get('name','').strip()
        if name: c.execute('INSERT INTO crops(user_id,name,planted_date,notes) VALUES(?,?,?,?)',(session['user_id'],name,request.form.get('planted_date',''),request.form.get('notes','').strip())); c.commit()
    rows=c.execute('SELECT * FROM crops WHERE user_id=? ORDER BY id DESC',(session['user_id'],)).fetchall(); c.close(); return render_template('crops.html',crops=rows,user_name=session['user_name'])

@app.route('/crops/delete/<int:crop_id>',methods=['POST'])
def delete_crop(crop_id):
    if not logged(): return redirect(url_for('login'))
    c=db(); c.execute('DELETE FROM crops WHERE id=? AND user_id=?',(crop_id,session['user_id'])); c.commit(); c.close(); return redirect(url_for('crops'))

@app.route('/doctor')
def doctor():
    if not logged(): return redirect(url_for('login'))
    return render_template('doctor.html',user_name=session['user_name'])

@app.route('/doctor/ask',methods=['POST'])
def doctor_ask():
    if not logged(): return jsonify(error='Login required'),401
    q=(request.json or {}).get('question','').lower().strip()
    answer='Tell me the crop, detected disease, symptoms, and what you want to know.'
    rules=[(['treat','medicine','treatment'],'For treatment, use the disease-specific advice shown by AgriSense and consult a local agricultural expert before applying chemicals.'),(['prevent','avoid'],'Improve field hygiene, remove infected material, avoid unnecessary leaf wetness, and monitor plants regularly.'),(['spread','spreading'],'Disease spread depends on the disease and may involve water splash, tools, insects, infected material, or environmental conditions.'),(['water','watering'],'Prefer watering the soil/root zone rather than keeping leaves wet for long periods.'),(['safe','eat'],'AgriSense is a screening tool; follow local agricultural guidance before deciding whether produce is safe to consume.')]
    for words,text in rules:
        if any(w in q for w in words): answer=text; break
    return jsonify(answer=answer)

@app.route('/weather')
def weather():
    if not logged(): return jsonify(error='Login required'),401
    city=request.args.get('city','Puttur').strip() or 'Puttur'
    try:
        g=requests.get('https://geocoding-api.open-meteo.com/v1/search',params={'name':city,'count':1,'language':'en','format':'json'},timeout=8).json(); r=g.get('results',[None])[0]
        if not r: return jsonify(error='Location not found.'),404
        w=requests.get('https://api.open-meteo.com/v1/forecast',params={'latitude':r['latitude'],'longitude':r['longitude'],'current':'temperature_2m,relative_humidity_2m,precipitation,weather_code','timezone':'auto'},timeout=8).json().get('current',{})
        return jsonify(city=r.get('name',city),country=r.get('country',''),temperature=w.get('temperature_2m'),humidity=w.get('relative_humidity_2m'),precipitation=w.get('precipitation'),weather_code=w.get('weather_code'))
    except Exception as e: return jsonify(error='Weather service unavailable.',details=str(e)),502

@app.route('/report/<int:scan_id>')
def report(scan_id):
    if not logged(): return redirect(url_for('login'))
    c=db(); r=c.execute('SELECT * FROM scans WHERE id=? AND user_id=?',(scan_id,session['user_id'])).fetchone(); c.close()
    if not r: return 'Report not found',404
    path=os.path.join('/tmp',f'agrisense_report_{scan_id}.pdf'); p=canvas.Canvas(path,pagesize=A4); W,H=A4; y=H-60
    p.setFont('Helvetica-Bold',20); p.drawString(50,y,'AgriSense Crop Health Report'); y-=30; p.setFont('Helvetica',11); p.drawString(50,y,f"Farmer: {session['user_name']}"); y-=18; p.drawString(50,y,f"Date: {r['created_at']}"); y-=30; p.setFont('Helvetica-Bold',15); p.drawString(50,y,f"Disease: {r['disease']}"); y-=20; p.setFont('Helvetica',11); p.drawString(50,y,f"AI confidence: {r['confidence']}%"); y-=30
    for title,text in [('Description',r['description']),('Treatment',r['treatment']),('Prevention',r['prevention'])]:
        p.setFont('Helvetica-Bold',13); p.drawString(50,y,title); y-=18; p.setFont('Helvetica',10); words=str(text).split(); line=''
        for word in words:
            if len(line)+len(word)>90: p.drawString(55,y,line); y-=14; line=''
            line+=word+' '
        if line: p.drawString(55,y,line); y-=22
        if y<70: p.showPage(); y=H-60
    p.setFont('Helvetica-Oblique',8); p.drawString(50,45,'AgriSense is an AI screening tool. Consult an agricultural professional for diagnosis and treatment decisions.'); p.save(); return send_file(path,as_attachment=True,download_name=f'agrisense_report_{scan_id}.pdf')

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=False)
