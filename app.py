import os
import time
import datetime
import random
import pickle
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
from river import anomaly
from river import compose
from river import preprocessing
import hashlib

def ske_encrypt(plaintext, keyword):
    key = hashlib.sha256(keyword.encode('utf-8')).digest()
    plaintext_bytes = plaintext.encode('utf-8')
    ciphertext_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(plaintext_bytes)])
    return ciphertext_bytes.hex()

def ske_decrypt(ciphertext_hex, keyword):
    try:
        key = hashlib.sha256(keyword.encode('utf-8')).digest()
        ciphertext_bytes = bytes.fromhex(ciphertext_hex)
        plaintext_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(ciphertext_bytes)])
        return plaintext_bytes.decode('utf-8')
    except Exception:
        return "[Decryption Error]"

def simulate_pom_consensus(tx_id, tx_type, gap_type, score):
    nodes = ["EC-Node-1", "EC-Node-2", "EC-Node-3", "EC-Node-4", "EC-Node-5"]
    votes = {}
    
    # Federated model offsets representing local training variances
    if gap_type == "Gap 1":
        # Isolation Forest decision score (anomaly is < 0.0)
        offsets = [-0.04, 0.02, -0.01, 0.03, -0.02]
        confirms = 0
        aborts = 0
        for i, n in enumerate(nodes):
            local_score = score + offsets[i]
            if local_score < 0.0:
                votes[n] = "ABORT"
                aborts += 1
            else:
                votes[n] = "CONFIRM"
                confirms += 1
                
    elif gap_type == "Gap 2":
        # LSTM reconstruction MAE (anomaly is > 0.2)
        offsets = [-0.03, 0.01, -0.01, 0.02, -0.02]
        confirms = 0
        aborts = 0
        for i, n in enumerate(nodes):
            local_score = score + offsets[i]
            if local_score > 0.2:
                votes[n] = "ABORT"
                aborts += 1
            else:
                votes[n] = "CONFIRM"
                confirms += 1
                
    elif gap_type == "Gap 3":
        # Isolation Forest decision score (anomaly is < -0.491)
        offsets = [-0.003, 0.001, -0.001, 0.002, -0.002]
        confirms = 0
        aborts = 0
        for i, n in enumerate(nodes):
            local_score = score + offsets[i]
            if local_score < -0.491:
                votes[n] = "ABORT"
                aborts += 1
            else:
                votes[n] = "CONFIRM"
                confirms += 1
                
    elif gap_type == "Gap 4":
        # Data quality. score is the predicted class (1 for anomaly, 0 for normal).
        if score == 1:
            votes = {
                "EC-Node-1": "ABORT",
                "EC-Node-2": "ABORT",
                "EC-Node-3": "ABORT",
                "EC-Node-4": "CONFIRM", # slight local threshold variance
                "EC-Node-5": "ABORT"
            }
            confirms = 1
            aborts = 4
        else:
            votes = {
                "EC-Node-1": "CONFIRM",
                "EC-Node-2": "CONFIRM",
                "EC-Node-3": "CONFIRM",
                "EC-Node-4": "CONFIRM",
                "EC-Node-5": "ABORT" # slight false alarm
            }
            confirms = 4
            aborts = 1
            
    elif gap_type == "Gap 5":
        # River online HalfSpaceTrees anomaly score (anomaly is > 0.95)
        offsets = [-0.02, 0.01, -0.01, 0.02, -0.02]
        confirms = 0
        aborts = 0
        for i, n in enumerate(nodes):
            local_score = score + offsets[i]
            if local_score > 0.95:
                votes[n] = "ABORT"
                aborts += 1
            else:
                votes[n] = "CONFIRM"
                confirms += 1
    else:
        # Fallback
        votes = {n: "CONFIRM" for n in nodes}
        confirms = 5
        aborts = 0
        
    score_pom = (confirms - aborts) / 5.0
    status = "COMMITTED" if score_pom > 0 else "ABORTED"
    
    consensus_col.insert_one({
        'timestamp': datetime.datetime.now(),
        'tx_id': tx_id,
        'tx_type': tx_type,
        'gap_type': gap_type,
        'votes': votes,
        'confirms': confirms,
        'aborts': aborts,
        'score': score_pom,
        'status': status
    })
    
    return status

app = Flask(__name__)
app.secret_key = 'spcbac_super_secret'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

# --- MongoDB Setup ---
client = MongoClient('mongodb://localhost:27017/')
db = client['spcbac_db']
auth_logs = db['auth_logs']
events_col = db['events']
tasks_col = db['tasks']
readings_col = db['sensor_readings']
consensus_col = db['consensus_logs']

# Clean startup events to prevent stale demo state
events_col.delete_many({})
consensus_col.delete_many({})

# --- Load Pre-trained Models ---
MODELS_DIR = 'models'

# Gap 1
with open(os.path.join(MODELS_DIR, 'gap1_model_real.pkl'), 'rb') as f:
    gap1_model = pickle.load(f)

# Gap 2
gap2_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'gap2_model.keras'))
gap2_scaler = joblib.load(os.path.join(MODELS_DIR, 'gap2_scaler.pkl'))

# Gap 3
gap3_model = joblib.load(os.path.join(MODELS_DIR, 'gap3_model.pkl'))
gap3_scaler = joblib.load(os.path.join(MODELS_DIR, 'gap3_scaler.pkl'))

# Gap 4
gap4_model = joblib.load(os.path.join(MODELS_DIR, 'gap4_rf_model.pkl'))
gap4_scaler = joblib.load(os.path.join(MODELS_DIR, 'gap4_scaler.pkl'))
gap4_features = joblib.load(os.path.join(MODELS_DIR, 'gap4_features.pkl'))

# Gap 5 - HalfSpaceTrees from River
gap5_model = compose.Pipeline(
    preprocessing.StandardScaler(),
    anomaly.HalfSpaceTrees(
        n_trees=25,
        height=8,
        window_size=100,
        seed=42
    )
)

def init_gap5():
    if auth_logs.count_documents({}) < 230:
        print("Seeding/re-seeding auth_logs collection for Gap 5...")
        auth_logs.delete_many({})
        # 200 Normal logs
        for _ in range(200):
            record = {
                'failed_attempts_last_60s': random.randint(0, 1),
                'unique_ips_last_60s': 1,
                'time_since_last_attempt_s': random.uniform(60, 3600),
                'is_success': 1,
                'fail_rate': random.uniform(0, 0.05)
            }
            gap5_model.learn_one(record)
            auth_logs.insert_one(record)
            
        # 30 Attack logs
        for _ in range(30):
            record = {
                'failed_attempts_last_60s': random.randint(5, 20),
                'unique_ips_last_60s': random.randint(2, 5),
                'time_since_last_attempt_s': random.uniform(0.1, 5.0),
                'is_success': 0,
                'fail_rate': random.uniform(0.8, 1.0)
            }
            gap5_model.learn_one(record)
            auth_logs.insert_one(record)
    else:
        print("Loading existing auth_logs for Gap 5...")
        for doc in auth_logs.find():
            record = {k: v for k, v in doc.items() if k != '_id'}
            gap5_model.learn_one(record)

# Initialize Gap 5
init_gap5()

# State variables for simulation tracking
user_stats = {
    'user': {'session_count': 0, 'queries': 0, 'failed_auth': 0}
}
VALID_USERS = {'user': 'user123'}

def log_event(entity, action, gap, score, is_anomaly, details=""):
    events_col.insert_one({
        'timestamp': datetime.datetime.now(),
        'entity': entity,
        'action': action,
        'gap_triggered': gap,
        'score': float(score),
        'is_anomaly': bool(is_anomaly),
        'details': details
    })

# --- Background Task: Gap 4 (Simulate Device Uploading) ---
def simulate_device():
    # Find a random Pending task (ignoring T-115 to keep it permanently pending)
    pending_task = tasks_col.find_one({'status': 'Pending', 'task_id': {'$ne': 'T-115'}})
    if not pending_task:
        return
        
    task_id = pending_task['task_id']
    device_id = pending_task.get('assigned_device_id') or f"D-{random.randint(100, 999)}"
    data_type = pending_task['data_type']
    
    # Generate value and unit based on data type
    val, unit = 0.0, ""
    if data_type == 'Temperature':
         val, unit = round(random.uniform(22.0, 39.5), 1), "°C"
    elif data_type == 'Air Quality':
        val, unit = int(random.uniform(30, 280)), "AQI"
    elif data_type == 'Traffic Density':
        val, unit = int(random.uniform(10, 95)), "vehicles/min"
    elif data_type == 'Noise Level':
        val, unit = int(random.uniform(50, 100)), "dB"
    elif data_type == 'Humidity':
        val, unit = int(random.uniform(15, 95)), "%"

    # Generate 31 random features for Gap 4
    data = pd.DataFrame(np.random.randn(1, 31), columns=gap4_features)
    
    # Scale and predict with Gap 4 (Random Forest)
    scaled_data = gap4_scaler.transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=gap4_features)
    pred = gap4_model.predict(scaled_df)[0]
    
    # 1 is Low quality / fake data, 0 is Good data
    is_anomaly = bool(pred == 1)
    verdict = "LOW QUALITY" if is_anomaly else "GOOD"
    score = 1.0 if is_anomaly else 0.0
    
    # Save sensor reading (SKE Encrypted)
    task_keyword = f"secret_key_{task_id}"
    raw_payload = f"{val} {unit}"
    encrypted_value = ske_encrypt(raw_payload, task_keyword)
    readings_col.insert_one({
        'task_id': task_id,
        'stakeholder_id': pending_task['stakeholder_id'],
        'device_id': device_id,
        'data_type': data_type,
        'value': encrypted_value,
        'unit': unit,
        'timestamp': datetime.datetime.now(),
        'quality_verdict': verdict,
        'gap4_score': score,
        'retrieved': False
    })
    
    # Update task status to Completed
    tasks_col.update_one({'_id': pending_task['_id']}, {'$set': {'status': 'Completed', 'assigned_device_id': device_id}})
    
    # Log device upload event
    log_event(
        entity=device_id,
        action="Sensor_Upload",
        gap="Gap 4: Data Quality",
        score=score,
        is_anomaly=is_anomaly,
        details=f"Device uploaded data for Task {task_id}. Verdict: {verdict}"
    )
    # Trigger PoM consensus (audited by Gap 4 Data Quality)
    simulate_pom_consensus(task_id, "Sensor_Upload", "Gap 4", int(pred))

scheduler = BackgroundScheduler()
scheduler.add_job(func=simulate_device, trigger="interval", seconds=30)
scheduler.start()

# --- Routes ---
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username:
            username = username.strip()
        if password:
            password = password.strip()
            
        # GAP 5 Tracking
        if username not in user_stats:
            user_stats[username] = {'session_count': 0, 'queries': 0, 'failed_auth': 0}
            
        success = 1 if (username in VALID_USERS and VALID_USERS[username] == password) else 0
        
        if not success:
            user_stats[username]['failed_auth'] += 1
            
        # Gap 5 prediction
        record = {
            'failed_attempts_last_60s': user_stats[username]['failed_auth'],
            'unique_ips_last_60s': 1,
            'time_since_last_attempt_s': random.uniform(180, 600) if success else random.uniform(0.5, 4.0),
            'is_success': success,
            'fail_rate': user_stats[username]['failed_auth'] / max(1, user_stats[username]['session_count'])
        }
        
        # Gap 5 online prediction using River HalfSpaceTrees (calibrated threshold)
        gap5_score = float(gap5_model.score_one(record))
        is_gap5_anomaly = bool(gap5_score > 0.95)
                
        gap5_model.learn_one(record)
        auth_logs.insert_one(record)
        
        log_event(username, "Login_Attempt", "Gap 5: Brute Force", gap5_score, is_gap5_anomaly)
        
        # Trigger PoM consensus for Gap 5 Brute Force
        pom_status_g5 = simulate_pom_consensus(username, "Login_Auth", "Gap 5", gap5_score)
        if pom_status_g5 == "ABORTED":
            return render_template('login.html', error="Login blocked: Multiple login failures flagged by consensus nodes.")
            
        if success:
            session.permanent = True
            session['user'] = username
            user_stats[username]['session_count'] += 1
            user_stats[username]['failed_auth'] = 0 # reset on success
            
            # GAP 3: Triggered on successful login using Isolation Forest
            # Logon features: ['logon_count', 'logoff_count', 'unique_pcs', 'first_logon_hour', 'last_logoff_hour', 'active_hours']
            session_count = user_stats[username]['session_count']
            if session_count <= 2:
                # Normal behavior
                g3_features = pd.DataFrame([[session_count, session_count, 1, 9, 17, 8]], 
                                           columns=['logon_count', 'logoff_count', 'unique_pcs', 'first_logon_hour', 'last_logoff_hour', 'active_hours'])
            else:
                # Anomaly behavior (multiple logins, off-hours, high count)
                g3_features = pd.DataFrame([[10, 10, 5, 2, 23, 21]], 
                                           columns=['logon_count', 'logoff_count', 'unique_pcs', 'first_logon_hour', 'last_logoff_hour', 'active_hours'])
            
            scaled_g3 = gap3_scaler.transform(g3_features)
            score_g3 = float(gap3_model.decision_function(scaled_g3)[0])
            is_g3_anomaly = bool(score_g3 < -0.491)
            display_score_g3 = 1.0 if is_g3_anomaly else float(abs(score_g3))
            
            log_event(username, "Login_Success", "Gap 3: Federated Insider Threat", display_score_g3, is_g3_anomaly)
            
            # Trigger PoM consensus for Gap 3
            pom_status_g3 = simulate_pom_consensus(username, "Login_Auth", "Gap 3", score_g3)
            if pom_status_g3 == "ABORTED":
                session.clear()
                return render_template('login.html', error="Login blocked: Abnormal logging activity flagged by consensus nodes.")
            
            # Seed mock tasks (delete existing first to prevent duplicates)
            readings_col.delete_many({'stakeholder_id': username})
            tasks_col.delete_many({'stakeholder_id': username})
            
            mock_tasks = [
                {
                    'task_id': 'T-841',
                    'stakeholder_id': username,
                    'data_type': 'Temperature',
                    'location': 'Sector 4, Block B',
                    'description': 'Measure ambient temperature at core junctions.',
                    'reward': 150,
                    'status': 'Completed',
                    'created_at': datetime.datetime.now() - datetime.timedelta(hours=2),
                    'assigned_device_id': 'D-221'
                },
                {
                    'task_id': 'T-219',
                    'stakeholder_id': username,
                    'data_type': 'Air Quality',
                    'location': 'Metro Station Exit',
                    'description': 'Check particulate concentration during rush hours.',
                    'reward': 300,
                    'status': 'Pending',
                    'created_at': datetime.datetime.now() - datetime.timedelta(hours=1),
                    'assigned_device_id': 'D-502'
                },
                {
                    'task_id': 'T-904',
                    'stakeholder_id': username,
                    'data_type': 'Traffic Density',
                    'location': 'Central Ring Road',
                    'description': 'Gather traffic pass count to feed into smart traffic systems.',
                    'reward': 220,
                    'status': 'Pending',
                    'created_at': datetime.datetime.now() - datetime.timedelta(minutes=30),
                    'assigned_device_id': 'D-641'
                },
                {
                    'task_id': 'T-115',
                    'stakeholder_id': username,
                    'data_type': 'Humidity',
                    'location': 'City Park Greenhouses',
                    'description': 'Monitor relative humidity values.',
                    'reward': 100,
                    'status': 'Pending',
                    'created_at': datetime.datetime.now() - datetime.timedelta(minutes=10),
                    'assigned_device_id': 'D-781'
                }
            ]
            for task in mock_tasks:
                tasks_col.insert_one(task)
                # For Completed tasks, seed corresponding sensor readings (SKE Encrypted)
                if task['status'] == 'Completed':
                    task_keyword = f"secret_key_{task['task_id']}"
                    encrypted_value = ske_encrypt("34.2 °C", task_keyword)
                    readings_col.insert_one({
                        'task_id': task['task_id'],
                        'stakeholder_id': username,
                        'device_id': task['assigned_device_id'],
                        'data_type': task['data_type'],
                        'value': encrypted_value,
                        'unit': "°C",
                        'timestamp': datetime.datetime.now() - datetime.timedelta(minutes=45),
                        'quality_verdict': 'GOOD',
                        'gap4_score': 0.0,
                        'retrieved': False
                    })
            
            return redirect(url_for('dashboard', user=username))
        else:
            return render_template('login.html', error="Invalid credentials")
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    user = session.get('user') or request.args.get('user')
    if not user:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=user)

# --- Tasks API ---
@app.route('/api/tasks', methods=['GET', 'POST'])
def api_tasks():
    username = session.get('user') or request.args.get('user') or (request.json.get('user') if (request.is_json and request.json) else None)
    if not username:
        return jsonify({"error": "Unauthorized"}), 401
    
    if request.method == 'POST':
        data = request.json
        data_type = data.get('data_type')
        location = data.get('location')
        description = data.get('description')
        reward = int(data.get('reward', 0))
        
        task_id = f"T-{random.randint(100, 999)}"
        device_id = f"D-{random.randint(100, 999)}"
        
        task = {
            'task_id': task_id,
            'stakeholder_id': username,
            'data_type': data_type,
            'location': location,
            'description': description,
            'reward': reward,
            'status': 'Pending',
            'created_at': datetime.datetime.now(),
            'assigned_device_id': device_id
        }
        tasks_col.insert_one(task)
        
        # Trigger Gaps 1 & 2 for stakeholder task posting activity
        user_stats[username]['queries'] += 1
        
        # GAP 1: Stakeholder Behavior (Isolation Forest)
        # Features: ['avg_hour', 'total_duration', 'unique_pcs', 'device_connects']
        queries = user_stats[username]['queries']
        current_hour = datetime.datetime.now().hour
        
        # We read the actual system hour but align it with the model's training boundaries (5-12) to avoid false positives.
        if queries < 6:
            avg_hour = float(current_hour) if 5 <= current_hour <= 12 else 10.0
            device_connects = 0.0
            total_duration = 300.0
            unique_pcs = 1
        else:
            avg_hour = 3.0  # Attack hour (3 AM)
            device_connects = 20.0
            total_duration = 10000.0
            unique_pcs = 5
            
        g1_features = pd.DataFrame([[avg_hour, total_duration, unique_pcs, device_connects]], columns=['avg_hour', 'total_duration', 'unique_pcs', 'device_connects'])
        
        pred_g1 = gap1_model.predict(g1_features)[0]
        score_g1 = float(gap1_model.decision_function(g1_features)[0])
        is_g1_anomaly = bool(pred_g1 == -1)
        display_score_g1 = 1.0 if is_g1_anomaly else float(abs(score_g1))
        log_event(username, f"Post_Task_{task_id}", "Gap 1: Behavior", display_score_g1, is_g1_anomaly)
        
        # GAP 2: Insider Threat (LSTM Autoencoder)
        # Features: ['session_count', 'off_hours', 'login_hour', 'device_connects', 'unique_pcs', 'failed_auth_attempts']
        session_count = user_stats[username]['session_count']
        if queries < 6:
            login_hour = float(current_hour) if 4 <= current_hour <= 23 else 14.0
            off_hours = 0
            device_connects = 0.0
            unique_pcs = 1
            failed_auth = 0
        else:
            login_hour = 3.0
            off_hours = 1
            device_connects = 25.0
            unique_pcs = 5
            failed_auth = 5
            
        g2_input = pd.DataFrame([[session_count, off_hours, login_hour, device_connects, unique_pcs, failed_auth]], columns=['session_count', 'off_hours', 'login_hour', 'device_connects', 'unique_pcs', 'failed_auth_attempts'])
        scaled_g2 = gap2_scaler.transform(g2_input).reshape(1, 1, 6)
        recon_g2 = gap2_model.predict(scaled_g2, verbose=0)
        mae_g2 = float(np.mean(np.abs(scaled_g2 - recon_g2)))
        is_g2_anomaly = bool(mae_g2 > 0.2)
        log_event(username, f"Post_Task_{task_id}", "Gap 2: Insider", mae_g2, is_g2_anomaly)
        
        return jsonify({"status": "success", "task_id": task_id})

    # GET Request
    tasks = list(tasks_col.find({'stakeholder_id': username}))
    for t in tasks:
        t['_id'] = str(t['_id'])
        t['created_at'] = t['created_at'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(t['created_at'], datetime.datetime) else t['created_at']
    return jsonify(tasks)

# --- Retrieve Sensing Data API ---
@app.route('/api/tasks/retrieve', methods=['POST'])
def api_retrieve_data():
    username = session.get('user') or request.args.get('user') or (request.json.get('user') if (request.is_json and request.json) else None)
    if not username:
        return jsonify({"error": "Unauthorized"}), 401
    
    task_id = request.json.get('task_id')
    task = tasks_col.find_one({'task_id': task_id, 'stakeholder_id': username})
    if not task:
        return jsonify({"error": "Task not found"}), 404
        
    # Increment query counts (simulating cryptographic trapdoor matching request)
    user_stats[username]['queries'] += 1
    
    # GAP 1: Stakeholder Behavior (Isolation Forest)
    # Features: ['avg_hour', 'total_duration', 'unique_pcs', 'device_connects']
    queries = user_stats[username]['queries']
    current_hour = datetime.datetime.now().hour
    
    # We read the actual system hour but align it with the model's training boundaries (5-12) to avoid false positives.
    if queries < 6:
        avg_hour = float(current_hour) if 5 <= current_hour <= 12 else 10.0
        device_connects = 0.0
        total_duration = 300.0
        unique_pcs = 1
    else:
        avg_hour = 3.0  # Attack hour (3 AM)
        device_connects = 20.0
        total_duration = 10000.0
        unique_pcs = 5
        
    g1_features = pd.DataFrame([[avg_hour, total_duration, unique_pcs, device_connects]], columns=['avg_hour', 'total_duration', 'unique_pcs', 'device_connects'])
    
    pred_g1 = gap1_model.predict(g1_features)[0]
    score_g1 = float(gap1_model.decision_function(g1_features)[0])
    is_g1_anomaly = bool(pred_g1 == -1)
    display_score_g1 = 1.0 if is_g1_anomaly else float(abs(score_g1))
    log_event(username, f"Retrieve_{task_id}", "Gap 1: Behavior", display_score_g1, is_g1_anomaly)
    
    # GAP 2: Insider Threat (LSTM Autoencoder)
    # Features: ['session_count', 'off_hours', 'login_hour', 'device_connects', 'unique_pcs', 'failed_auth_attempts']
    session_count = user_stats[username]['session_count']
    if queries < 6:
        login_hour = float(current_hour) if 4 <= current_hour <= 23 else 14.0
        off_hours = 0
        device_connects = 0.0
        unique_pcs = 1
        failed_auth = 0
    else:
        login_hour = 3.0
        off_hours = 1
        device_connects = 25.0
        unique_pcs = 5
        failed_auth = 5
        
    g2_input = pd.DataFrame([[session_count, off_hours, login_hour, device_connects, unique_pcs, failed_auth]], columns=['session_count', 'off_hours', 'login_hour', 'device_connects', 'unique_pcs', 'failed_auth_attempts'])
    scaled_g2 = gap2_scaler.transform(g2_input).reshape(1, 1, 6)
    recon_g2 = gap2_model.predict(scaled_g2, verbose=0)
    mae_g2 = float(np.mean(np.abs(scaled_g2 - recon_g2)))
    is_g2_anomaly = bool(mae_g2 > 0.2)
    log_event(username, f"Retrieve_{task_id}", "Gap 2: Insider", mae_g2, is_g2_anomaly)
    
    # Fetch sensor reading from DB
    reading = readings_col.find_one({'task_id': task_id, 'stakeholder_id': username})
    if not reading:
        # Fallback generator if reading isn't completed yet
        val, unit = 34.2, "°C"
        if task['data_type'] == 'Air Quality': val, unit = 72, "AQI"
        elif task['data_type'] == 'Traffic Density': val, unit = 42, "vehicles/min"
        elif task['data_type'] == 'Noise Level': val, unit = 58, "dB"
        elif task['data_type'] == 'Humidity': val, unit = 65, "%"
        
        task_keyword = f"secret_key_{task_id}"
        encrypted_value = ske_encrypt(f"{val} {unit}", task_keyword)
        
        reading = {
            'task_id': task_id,
            'stakeholder_id': username,
            'device_id': task.get('assigned_device_id', 'D-999'),
            'data_type': task['data_type'],
            'value': encrypted_value,
            'unit': unit,
            'timestamp': datetime.datetime.now(),
            'quality_verdict': 'GOOD',
            'gap4_score': 0.0,
            'retrieved': False
        }
        readings_col.insert_one(reading)
        
    # Decrypt value since it's being retrieved (and mark as retrieved)
    task_keyword = f"secret_key_{task_id}"
    decrypted_value = ske_decrypt(reading['value'], task_keyword)
    
    # Mark reading as retrieved in the DB
    readings_col.update_one({'_id': reading['_id']}, {'$set': {'retrieved': True}})
    
    # Trigger PoM consensus on retrieval activity (audited by Gaps 1 & 2 federated detectors)
    if is_g1_anomaly:
        simulate_pom_consensus(task_id, "SKE_Retrieval", "Gap 1", score_g1)
    elif is_g2_anomaly:
        simulate_pom_consensus(task_id, "SKE_Retrieval", "Gap 2", mae_g2)
    else:
        simulate_pom_consensus(task_id, "SKE_Retrieval", "Gap 1", score_g1)
    
    return jsonify({
        "status": "success",
        "reading": {
            "task_id": reading['task_id'],
            "device_id": reading['device_id'],
            "data_type": reading['data_type'],
            "value": decrypted_value,
            "unit": reading['unit'],
            "timestamp": reading['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(reading['timestamp'], datetime.datetime) else reading['timestamp'],
            "quality_verdict": reading['quality_verdict']
        }
    })

# --- Retrieved Sensor Readings List API ---
@app.route('/api/sensor_readings', methods=['GET'])
def api_sensor_readings():
    username = session.get('user') or request.args.get('user')
    if not username:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Query readings directly scoped to this stakeholder
    readings = list(readings_col.find({'stakeholder_id': username}))
    for r in readings:
        r['_id'] = str(r['_id'])
        r['timestamp'] = r['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r['timestamp'], datetime.datetime) else r['timestamp']
        r['retrieved'] = r.get('retrieved', False)
        if r['retrieved']:
            task_keyword = f"secret_key_{r['task_id']}"
            r['value'] = ske_decrypt(r['value'], task_keyword)
    return jsonify(readings)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('admin.html')

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if username == "admin" and password == "password":
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))
        else:
            return render_template('admin_login.html', error="Unauthorized Access")
            
    return render_template('admin_login.html')

@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/api/admin_status')
def api_admin_status():
    if not session.get('admin_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    gaps_status = {}
    for gap_id in ["Gap 1: Behavior", "Gap 2: Insider", "Gap 3: Federated Insider Threat", "Gap 4: Data Quality", "Gap 5: Brute Force"]:
        last_event = events_col.find_one({"gap_triggered": gap_id}, sort=[("timestamp", -1)])
        if last_event:
            gaps_status[gap_id] = {
                "score": round(last_event['score'], 4),
                "is_anomaly": last_event['is_anomaly'],
                "timestamp": last_event['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            gaps_status[gap_id] = {
                "score": 0.0,
                "is_anomaly": False,
                "timestamp": "No events yet"
            }
            
    recent_events = list(events_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
    for e in recent_events:
        e['timestamp'] = e['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        
    return jsonify({
        "gaps": gaps_status,
        "events": recent_events
    })

@app.route('/api/consensus_logs')
def api_consensus_logs():
    if not session.get('admin_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    recent_logs = list(consensus_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
    for log in recent_logs:
        log['timestamp'] = log['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(recent_logs)

@app.route('/api/reset_demo', methods=['POST'])
def api_reset_demo():
    if not session.get('admin_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    events_col.delete_many({})
    tasks_col.delete_many({})
    readings_col.delete_many({})
    consensus_col.delete_many({})
    global user_stats
    user_stats = {
        'user': {'session_count': 0, 'queries': 0, 'failed_auth': 0}
    }
    # Clear stakeholder session variables, but keep admin logged in status
    admin_logged_in = session.get('admin_logged_in')
    session.clear()
    if admin_logged_in:
        session['admin_logged_in'] = True
    return jsonify({"status": "success", "message": "Demo system reset successfully."})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)

