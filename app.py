from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3, os, pandas as pd
from model_utils import run_experiment, load_demo_data

BASE=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(BASE,"faultsense.db")
UPLOAD=os.path.join(BASE,"uploads")
os.makedirs(UPLOAD,exist_ok=True)
app=Flask(__name__)
app.secret_key=os.environ.get("FLASK_SECRET_KEY","change-this-development-secret")
app.config["MAX_CONTENT_LENGTH"]=12*1024*1024

def connect():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
with connect() as c:
    c.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT UNIQUE, password_hash TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY,user_id INTEGER,model TEXT,accuracy REAL,f1 REAL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

def auth(): return "user_id" in session
@app.route("/")
def index(): return redirect(url_for("dashboard")) if auth() else render_template("landing.html")
@app.route("/signup",methods=["GET","POST"])
def signup():
    if request.method=="POST":
        u=request.form.get("username","").strip(); e=request.form.get("email","").strip().lower(); p=request.form.get("password","")
        if not u or not e or len(p)<8:
            flash("Complete all fields; password must contain at least 8 characters.","error")
        else:
            try:
                with connect() as c: c.execute("INSERT INTO users(username,email,password_hash) VALUES(?,?,?)",(u,e,generate_password_hash(p)))
                flash("Account created. Please sign in.","success"); return redirect(url_for("signin"))
            except sqlite3.IntegrityError: flash("Username or email already exists.","error")
    return render_template("auth.html",mode="signup")
@app.route("/signin",methods=["GET","POST"])
def signin():
    if request.method=="POST":
        identity=request.form.get("identity","").strip().lower(); p=request.form.get("password","")
        with connect() as c: user=c.execute("SELECT * FROM users WHERE lower(username)=? OR email=?",(identity,identity)).fetchone()
        if user and check_password_hash(user["password_hash"],p):
            session.clear(); session["user_id"]=user["id"]; session["username"]=user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid username/email or password.","error")
    return render_template("auth.html",mode="signin")
@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("index"))
@app.route("/dashboard")
def dashboard():
    if not auth(): return redirect(url_for("signin"))
    with connect() as c: runs=c.execute("SELECT * FROM runs WHERE user_id=? ORDER BY created_at DESC LIMIT 8",(session["user_id"],)).fetchall()
    d=load_demo_data()
    return render_template("dashboard.html",username=session["username"],records=len(d),classes=sorted(d.fault_label.unique()),runs=runs)
@app.route("/dataset",methods=["GET","POST"])
def dataset():
    if not auth(): return redirect(url_for("signin"))
    d=None
    if request.method=="POST":
        f=request.files.get("dataset")
        if f and f.filename.lower().endswith(".csv"):
            try:
                d=pd.read_csv(f)
                if "fault_label" not in d.columns: flash("CSV must contain target column fault_label.","error"); d=None
                elif len(d)<20: flash("Please upload at least 20 rows.","error"); d=None
                else:
                    name=secure_filename(f.filename); path=os.path.join(UPLOAD,f"{session['user_id']}_{name}"); d.to_csv(path,index=False)
                    session["dataset_path"]=path; session["dataset_name"]=name; flash(f"Uploaded {name} ({len(d)} rows).","success")
            except Exception as ex: flash(f"Could not read CSV: {ex}","error")
        else: flash("Choose a CSV file.","error")
    if d is None:
        p=session.get("dataset_path"); d=pd.read_csv(p) if p and os.path.exists(p) else load_demo_data()
    return render_template("dataset.html",df=d.head(12).to_html(index=False,classes="data-table"),rows=len(d),cols=list(d.columns),name=session.get("dataset_name","Bundled synthetic demo dataset"))
@app.route("/train",methods=["GET","POST"])
def train():
    if not auth(): return redirect(url_for("signin"))
    results=None
    if request.method=="POST":
        p=session.get("dataset_path"); d=pd.read_csv(p) if p and os.path.exists(p) else load_demo_data()
        try:
            results=run_experiment(d,float(request.form.get("label_ratio",.1)),float(request.form.get("threshold",.8)))
            with connect() as c: c.execute("INSERT INTO runs(user_id,model,accuracy,f1) VALUES(?,?,?,?)",(session["user_id"],results["model"],results["semi"]["accuracy"],results["semi"]["f1"]))
        except Exception as ex: flash(f"Training failed: {ex}","error")
    return render_template("train.html",results=results)
@app.route("/prediction",methods=["GET","POST"])
def prediction():
    if not auth(): return redirect(url_for("signin"))
    pred=None
    if request.method=="POST":
        try:
            from sklearn.preprocessing import StandardScaler
            vals=[float(request.form[k]) for k in ["vibration","temperature","pressure","speed","current"]]
            demo=load_demo_data(); cols=["vibration","temperature","pressure","speed","current"]
            sc=StandardScaler().fit(demo[cols]); x=sc.transform(pd.DataFrame([vals],columns=cols))
            distances={}
            for label in demo.fault_label.unique(): distances[label]=float(((x-sc.transform(demo.loc[demo.fault_label==label,cols]).mean(axis=0))**2).sum()**.5)
            pred=min(distances,key=distances.get)
            flash("Demo nearest-centroid result only; not validated industrial diagnosis.","info")
        except Exception as ex: flash(f"Prediction failed: {ex}","error")
    return render_template("prediction.html",prediction_result=pred)
if __name__=="__main__": app.run(debug=True)
