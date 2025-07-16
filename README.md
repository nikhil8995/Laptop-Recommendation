# 🚀 LapRec – Modern AI-Powered Laptop Recommendation Web App

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Django](https://img.shields.io/badge/Django-5.2-green?logo=django)
![Mobile Ready](https://img.shields.io/badge/Mobile%20Ready-Yes-blueviolet?logo=android)
![Dark Mode](https://img.shields.io/badge/Dark%20Mode-Default-black?logo=moon)

> **LapRec** is a beautiful, mobile-friendly, dark-mode-first web app for laptop recommendations, powered by machine learning and a real laptop dataset. Get the best laptops for your needs, see similar alternatives, and enjoy a modern, seamless experience on any device.

---

## ✨ Features

- **Smart Recommendations:** Get the best laptops for your use-case, budget, and brand.
- **Similar Laptops:** See alternatives to the top pick, powered by Nearest Neighbors.
- **Dark/Light Mode:** Beautiful by default, with a toggle (remembers your choice).
- **Mobile Compatible:** Works great on phones and desktops.
- **Modern UI:** Clean, responsive, and easy to use.
- **Network Ready:** Access from any device on your WiFi.

---

## 📸 Screenshots

> _Add your own screenshots here!_

![LapRec Dark Mode](./screenshots/dark-mode.png)
![LapRec Light Mode](./screenshots/light-mode.png)

---

## 🚦 Quick Start

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Set up the environment
```bash
conda create -n lapenv python=3.11
conda activate lapenv
pip install -r requirements.txt
```

### 3. Prepare the model
If `lap_rec.joblib` is not present, generate it:
```bash
python lap_rec.py
```

### 4. Run migrations
```bash
python manage.py migrate
```

### 5. Run the server
```bash
python manage.py runserver 0.0.0.0:8000
```
- Access on your machine: [http://localhost:8000/](http://localhost:8000/)
- Access on your network: [http://<your-ip>:8000/](http://<your-ip>:8000/)

---

## 🖥️ Using the App
- Fill out the form, get recommendations, and see similar laptops.
- Toggle dark/light mode with the button in the top-right.
- Fully responsive and touch-friendly.

---

## 🗂️ Project Structure
```
manage.py
requirements.txt
lap_rec.py
laptops.csv
lap_rec.joblib
website/
    settings.py, urls.py, ...
frontend/
    views.py, urls.py, ...
    static/frontend/
        style.css, main.js
    templates/frontend/
        home.html
    migrations/
```

---

## ⚙️ Customization
- **Update the dataset:** Replace `laptops.csv` and rerun `lap_rec.py`.
- **Change branding:** Edit `frontend/static/frontend/style.css` and `frontend/templates/frontend/home.html`.
- **Deploy:** Use a production server (e.g., Gunicorn + Nginx) and set `DEBUG = False` in `website/settings.py`.
- **Allow all hosts (dev only):** Set `ALLOWED_HOSTS = ['*']` in `website/settings.py`.

---

## 📝 .gitignore Example
```gitignore
__pycache__/
*.pyc
db.sqlite3
staticfiles/
*.joblib
*.csv:Zone.Identifier
```

---

## 🙏 Credits
- **Dataset:** Your own or [Kaggle Laptop Dataset](https://www.kaggle.com/)
- **Frontend:** Custom HTML/CSS/JS
- **Backend:** Django, scikit-learn, TensorFlow
- **Icons:** [Twemoji](https://twemoji.twitter.com/)

---

## 📄 License
MIT (or your preferred license)

---

## 💡 Tips
- For best results, use a modern browser.
- To access from your phone, make sure your computer and phone are on the same WiFi, and use your computer’s IP address.
- For production, use HTTPS and a secure deployment setup.

---# Laptop-Recommendation
