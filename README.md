# MediReportAI – Artifact-Free Diabetic Retinopathy Report Generator

🧠 **AI in Healthcare & Diagnostics**

This project implements an **AI-powered pipeline for generating medical reports** from retinal fundus images and patient data.  
It focuses on **artifact-free diabetic retinopathy (DR) diagnosis** by combining U-Net–based artifact removal, DR severity classification, ICD coding, and automated PDF report generation.

---

## 🚀 Features
- **Artifact Removal**: U-Net image fusion to eliminate glare/flash artifacts from fundus images.  
- **DR Severity Classification**: Vision-Language Models (VLM) for reliable disease detection.  
- **Macular Edema Detection**: Additional retinal pathology screening.  
- **Explainability**: Grad-CAM visualizations for model transparency.  
- **ICD Code Prediction**: Automated mapping of diagnoses to ICD-10 codes.  
- **Report Generation**: End-to-end system that outputs standardized PDF medical reports.  
- **Database Support**: Patient data stored and retrieved via SQLite.  
- **Deployment Ready**: Dockerized setup for easy reproducibility.

---

## 🗂️ Project Structure
```
.
├── app.py              # Main entry point (report generation pipeline)
├── backup.py           # Backup utility
├── db.py               # SQLite database handling
├── data/               # Raw/processed data
├── hospital.db         # Sample SQLite database
├── models/             # Trained AI models (artifact remover, classifier)
├── outputs/            # Generated reports and results
├── sample_input/       # Example patient data and fundus images
├── template/           # Report templates (PDF/HTML)
├── utils/              # Helper functions
├── Demo.mp4            # Demo video of the system
├── Presentation.pdf    # Technical overview & slides
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker environment
└── README.md           # Documentation (this file)
```

---

## ⚙️ Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/MediReportAI.git
   cd MediReportAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Build with Docker:
   ```bash
   docker build -t medireportai .
   docker run -p 8501:8501 medireportai
   ```

---

## 🖥️ Usage
Run the app to process sample input data and generate reports:
```bash
python app.py
```

- Input: Patient demographics + retinal fundus images.  
- Output: PDF report with DR severity, ICD-10 codes, Grad-CAM heatmaps, and recommendations.  

Sample input is available in `sample_input/`, and generated reports will be stored in `outputs/`.

---

## 📊 Demo & Sample Output
🎥 Check out the [Demo Video](./Demo.mp4).  
📑 See the [Presentation](./Presentation.pdf) for the technical overview.  

### Sample Report (Preview)
![Sample Report](<ADD_LINK_HERE>)

---

## 🧩 Tech Stack
- **Python** (FastAPI/Streamlit for frontend, PyTorch for AI models)  
- **SQLite** for database integration  
- **U-Net with Residual Blocks** for artifact removal  
- **Grad-CAM** for explainability  
- **Docker** for deployment  

---

## 📌 Future Scope
- Extend to other retinal diseases (e.g., glaucoma, AMD).  
- Cloud-based deployment for large-scale screenings.  
- Integration with hospital EHR systems.  

---

## 📜 License
This project is licensed under the [MIT License](./LICENSE).  

