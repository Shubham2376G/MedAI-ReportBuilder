import sqlite3
from datetime import datetime
import os

DB_PATH = "hospital.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            history TEXT,
            address TEXT,
            phone TEXT,
            notes TEXT,
            report_path TEXT,
            created_at TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dr_images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_path TEXT,
            note TEXT,
            uploaded_at TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')

    conn.commit()
    conn.close()

def insert_new_patient(name, age, gender, history, address, phone, notes, report_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO patients (name, age, gender, history, address, phone, notes, report_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, age, gender, history, address, phone, notes, report_path, datetime.now().isoformat()))
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id

def get_patient_by_id(patient_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return row

def insert_dr_image(patient_id, image_path, note):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO dr_images (patient_id, image_path, note, uploaded_at)
        VALUES (?, ?, ?, ?)
    ''', (patient_id, image_path, note, datetime.now().isoformat()))
    conn.commit()
    conn.close()
