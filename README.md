# Mobile Crowdsensing Secure Model (SP-CBAC)

This repository contains a full-featured demo application for the Secure Private Consortium Blockchain-Assisted Crowdsensing (SP-CBAC) model. The project is built with Flask and MongoDB. It simulates a mobile crowdsensing portal and displays real-time security threat detection using five pre-trained machine learning models (Gaps 1 to 5).

---

## What is in this Repository

- **app.py**: The main Flask server file. It initializes the database, loads the models, handles web pages and APIs, and runs the background device simulator.
- **models/**: A folder containing pre-trained models and preprocessing files:
  - `gap1_model_real.pkl`: Pre-trained Isolation Forest model for Gap 1.
  - `gap2_model.keras` and `gap2_scaler.pkl`: Pre-trained LSTM Autoencoder model and scaler for Gap 2.
  - `gap3_model.pkl` and `gap3_scaler.pkl`: Pre-trained Isolation Forest model and scaler for Gap 3.
  - `gap4_rf_model.pkl`, `gap4_scaler.pkl`, and `gap4_features.pkl`: Pre-trained Random Forest model and features list for Gap 4.
  - *Note: Gap 5 uses an online learning HalfSpaceTrees model that is built dynamically when Flask starts.*
- **templates/**: A folder containing HTML user interfaces:
  - `login.html`: The login page for crowdsensing stakeholders.
  - `admin_login.html`: The login page for security administrators.
  - `dashboard.html`: The 3-tab stakeholder portal (My Tasks, Post Task, and Retrieved Data).
  - `admin.html`: The Security SOC Panel that tracks anomaly detection models in real-time.
- **static/style.css**: The custom CSS stylesheet that implements a glassmorphic design.
- **requirements.txt**: Python package dependencies.
- **.gitignore**: Specifies files and folders that Git should ignore.

---

## Security Models (Gaps 1 to 5)

| Model / Gap | Focus Area | Algorithm | Demo Anomaly Trigger |
| :--- | :--- | :--- | :--- |
| **Gap 1: Behavior** | Stakeholder harvesting query rates | Isolation Forest | Post tasks or retrieve data 6 times in a single session |
| **Gap 2: Insider** | Sequence pattern threats | LSTM Autoencoder | Post tasks or retrieve data 6 times in a single session |
| **Gap 3: Federated** | Cross-node session flooding | Isolation Forest | Log in successfully 3 consecutive times in a row |
| **Gap 4: Data Quality** | Raw device telemetry verification | Random Forest | Runs automatically when background simulator uploads data |
| **Gap 5: Brute Force** | Failed login rate monitoring | HalfSpaceTrees | Enter incorrect passwords 3 times on the admin login screen |

---

## How to Set Up the Project

### Prerequisites
Make sure you have MongoDB installed and running on your local machine:
- MongoDB URI: `mongodb://localhost:27017/`

### Step 1: Install Dependencies
Open your terminal inside the project directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Server
Start the Flask application by running:
```bash
python app.py
```
The server will start on `http://127.0.0.1:5000/`.

---

## How to Run the Demo

### Authentication Portals
Stakeholder and administrator sessions are completely separate:
- **Stakeholder Portal** (`http://localhost:5000/login`): Log in as `user` with password `user123`.
- **Admin SOC Panel** (`http://localhost:5000/admin`): Automatically redirects to `/admin-login`. Log in as `admin` with password `password`.
- *Note: Navigation links at the bottom of each login card allow switching between portals.*

### Demo Steps
1. Open **Tab A** (Stakeholder Portal) and log in. You will see your seeded tasks. Click **Retrieve Data** on a completed task to watch the searchable encryption (SKE) trapdoor matching process.
2. Go to the **Post Task** tab and broadcast a new task. Wait 30 seconds. A simulated device will automatically complete it in the background.
3. Open **Tab B** (Admin SOC Panel), log in as administrator, and watch the live event logs populate. Clicking on any of the Gap cards will filter the log table.

### Resetting the Demo State
If you want to clear all tasks, sensor readings, and logs to show a fresh demo:
1. Go to Tab B (Admin SOC Panel).
2. Click the **Reset Demo** button in the top right navbar.
3. Confirm the prompt. The system will clear the database collections and log out active sessions immediately.
