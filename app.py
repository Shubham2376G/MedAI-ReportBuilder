import streamlit as st
import os
from db import init_db, insert_new_patient, get_patient_by_id, insert_dr_image
from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName, PdfObject
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use only GPU 3
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import numpy as np
import tensorflow as tf
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tensorflow.image import ssim
# from utils.infer_qwen import qwen_model
from utils.infer_severity import severity_model
from utils.infer_macular import macular_model
from utils.icd_code import get_icd10_code_from_text, describe_icd10_code
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pdfrw import PdfReader, PdfWriter, PageMerge
import io
from utils.background_remove import *
import zipfile
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.infer_qwen_s import qwen_extract

from utils.dicom import *
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
# File paths
TEMPLATE_PATH = "template/temp2.pdf"
OUTPUT_PATH = "outputs/filled_temp.pdf"

# Dummy credentials for nurse login
NURSE_CREDENTIALS = {
    "nurse1": "pass123",
    "nurse2": "xyz456"
}

# Simple login check
def login():
    st.title("🧑‍⚕️ XYZ Hospital Nurse Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if NURSE_CREDENTIALS.get(username) == password:
            st.session_state["logged_in"] = True
            st.session_state["nurse_name"] = username
            st.success(f"Welcome, Nurse {username}!")
        else:
            st.error("Invalid credentials. Please try again.")

# New patient form
def new_patient_form():
    st.header("🆕 New Patient Registration")
    with st.form("new_patient_form"):
        name = st.text_input("Patient Name")
        age = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        history = st.text_area("Clinical History")
        address = st.text_input("Address")
        phone = st.text_input("Phone Number")
        notes = st.text_area("Additional Notes (Optional)")
        pdf_report = st.file_uploader("Upload Patient Report (PDF)", type=["pdf"])
        submitted = st.form_submit_button("Submit")

        if pdf_report:
            os.makedirs("data/uploads", exist_ok=True)
            pdf_path = f"data/uploads/{name.replace(' ', '_')}_{pdf_report.name}"
            with open(pdf_path, "wb") as f:
                f.write(pdf_report.read())
        else:
            pdf_path = None

        
        


        if submitted:
            report_extract="Dome"          
            if pdf_report is not None:
                # Read the uploaded file once
                pdf_bytes = pdf_report.read()

                if pdf_bytes:
                    # Save it
                    os.makedirs("data/uploads", exist_ok=True)
                    filename = f"{name.replace(' ', '_')}_{pdf_report.name}"
                    pdf_path = os.path.join("data/uploads", filename)

                    with open(pdf_path, "wb") as f:
                        f.write(pdf_bytes)


                report_extract = qwen_extract(pdf_path)


            patient_id = insert_new_patient(name, age, gender, history, address, phone, notes, pdf_path)
            st.success(f"New patient '{name}' registered successfully with ID: {patient_id}")

            st.text(report_extract)

# Existing patient - DR upload module
def existing_patient_module():
    st.header("📁 Existing Patient Management")
    patient_id = st.text_input("Enter Patient ID")
    if patient_id:
        st.subheader(f"Selected Patient: {patient_id}")
        option = st.selectbox("Choose Condition", ["-- Select --", "Diabetic Retinopathy", "Hypertension", "Other"])
        
        if option == "Diabetic Retinopathy":
            # Step 1: Fetch patient data
            patient = get_patient_by_id(patient_id)
            if not patient:
                st.error("Patient not found.")
                return

            # Step 2: Upload and preview images
            st.markdown("### 🖼️ Upload Retina Images")

            # Upload for Right Eye
            st.subheader("👁️‍🗨️ Right Eye")
            right_eye_images = st.file_uploader("Upload 2 Right Eye Images (PNG, JPG or DICOM)", type=["png", "dcm", "jpg"], accept_multiple_files=True, key="right_eye")

            # Upload for Left Eye
            st.subheader("👁️ Left Eye")
            left_eye_images = st.file_uploader("Upload 2 Left Eye Images (PNG, JPG or DICOM)", type=["png", "dcm", "jpg"], accept_multiple_files=True, key="left_eye")

            notes_r = st.text_area("Right Eye Clinical Observations")
            notes_l = st.text_area("Left Eye Clinical Observations")
            notes_right = f"This is a macula-centered fundus image of the eye. Clinical observations include: {notes_r}.\n\nDetermine the diabetic retinopathy grade (No DR, Mild, Moderate, Severe, or Proliferative)"
            notes_left = f"This is a macula-centered fundus image of the eye. Clinical observations include: {notes_l}.\n\nDetermine the diabetic retinopathy grade (No DR, Mild, Moderate, Severe, or Proliferative)"
            # Format selection
            format_choice = st.selectbox("Select image format to download:", ["PNG", "DICOM"])
            if st.button("Generate DR Report"):
                
            
                # Save and preview images
                os.makedirs(f"data/uploads/patient_{patient_id}", exist_ok=True)
                image_paths = []

                def save_images(images, label):
                    paths = []
                    for img in images:
                        img_path = f"data/uploads/patient_{patient_id}/{label}_{img.name}"
                        with open(img_path, "wb") as f:
                            f.write(img.read())
                        # st.image(img, caption=f"{label} - {img.name}", use_container_width=True)
                        insert_dr_image(patient_id, img_path, notes_left)
                        paths.append(img_path)
                    return paths

                right_eye_paths = save_images(right_eye_images, "right")
                left_eye_paths = save_images(left_eye_images, "left")

                if len(right_eye_images) == 1 and len(left_eye_images) == 1:
                    # result_right_eye= qwen_model(right_eye_paths[0],notes_right)
                    # result_left_eye= qwen_model(left_eye_paths[0],notes_left)
                    result_right_eye = severity_model(right_eye_paths[0],notes_right,f"data/uploads/patient_{patient_id}/gradcamm_severity_r.png")
                    result_left_eye= severity_model(left_eye_paths[0],notes_left,f"data/uploads/patient_{patient_id}/gradcamm_severity_l.png")

                    macular_right = macular_model(right_eye_paths[0],f"data/uploads/patient_{patient_id}/gradcamm_macular_r.png")
                    macular_left = macular_model(left_eye_paths[0], f"data/uploads/patient_{patient_id}/gradcamm_macular_l.png")
                    
                else:

                    def ssim_loss(y_true, y_pred):
                        return 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))


                    # Combined loss (SSIM + MSE)
                    def combined_loss(y_true, y_pred):
                        ssim_component = ssim_loss(y_true, y_pred)
                        mse_component = tf.reduce_mean(tf.square(y_true - y_pred))
                        return 0.7 * ssim_component + 0.3 * mse_component
                    def load_and_preprocess(image_path, target_size=(256, 256)):
                        img = cv2.imread(image_path)
                        img = cv2.resize(img, target_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0
                        return img

                    def fuse_images(img_path1, img_path2, model, target_size=(256, 256)):
                        img1 = load_and_preprocess(img_path1, target_size)
                        img2 = load_and_preprocess(img_path2, target_size)
                        
                        input_concat = np.concatenate([img1, img2], axis=-1)
                        input_batch = np.expand_dims(input_concat, axis=0)

                        pred = model.predict(input_batch)[0]
                        pred = np.clip(pred, 0, 1)

                        pred_uint8 = (pred * 255).astype(np.uint8)
                        return pred_uint8    

                    model = tf.keras.models.load_model("models/artifact_remove/improved_arts.keras", custom_objects={'ssim_loss': ssim_loss, 'combined_loss': combined_loss})

                    output_r = fuse_images( right_eye_paths[0],right_eye_paths[1],model)
                    output_l = fuse_images( left_eye_paths[0],left_eye_paths[1],model)
                    cv2.imwrite(f"data/uploads/patient_{patient_id}/infered_imageOD_r.jpg", cv2.cvtColor(output_r, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f"data/uploads/patient_{patient_id}/infered_imageOD_l.jpg", cv2.cvtColor(output_l, cv2.COLOR_RGB2BGR))                    


                    # result_right_eye= qwen_model(f"data/uploads/patient_{patient_id}/infered_imageOD_r.jpg",notes_right)
                    # result_left_eye= qwen_model(f"data/uploads/patient_{patient_id}/infered_imageOD_l.jpg",notes_left)

                    result_right_eye = severity_model(right_eye_paths[0],notes_right,f"data/uploads/patient_{patient_id}/gradcamm_severity_r.png")
                    result_left_eye= severity_model(left_eye_paths[0],notes_left,f"data/uploads/patient_{patient_id}/gradcamm_severity_l.png")

                    macular_right = macular_model(f"data/uploads/patient_{patient_id}/infered_imageOD_r.jpg",f"data/uploads/patient_{patient_id}/gradcamm_macular_r.png")
                    macular_left = macular_model(f"data/uploads/patient_{patient_id}/infered_imageOD_l.jpg",f"data/uploads/patient_{patient_id}/gradcamm_macular_l.png")





                if "NO" in str(result_right_eye) and "NO" in str(result_left_eye):
                    screening_result= "Negative for referable diabetic retinopathy."
                else:
                    screening_result= "Positive for vision threatening diabetic retinopathy."


                icd_code_right= get_icd10_code_from_text(result_right_eye, macular_right, "right")
                icd_code_left= get_icd10_code_from_text(result_left_eye, macular_left, "left")


                # Proceed with form filling and overlay (use first image from each for PDF preview)
                # Fill the PDF form
                from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName, PdfObject

                today_date = datetime.today().strftime('%d-%m-%Y')
                pdf = PdfReader(TEMPLATE_PATH)
                form_data = {
                    "Text1": patient_id,
                    "Text2": f"{result_right_eye}",
                    "Text5": f"{result_left_eye}",
                    "Text7": patient_id,
                    "Text9": screening_result,
                    "Text12": patient[1],
                    "Text13": patient[3],
                    "Text14": patient[2],
                    "Text15": today_date,
                    "Text16": icd_code_left,
                    "Text3": icd_code_right,
                    "Text17": describe_icd10_code(icd_code_left),
                    "Text4": describe_icd10_code(icd_code_right)
                    
                    }

                for page in pdf.pages:
                    if page.Annots:
                        for annot in page.Annots:
                            if annot.T:
                                key = annot.T[1:-1]
                                if key in form_data:
                                    annot.V = form_data[key]
                                    annot.AP = None

                if hasattr(pdf, 'Root') and hasattr(pdf.Root, 'AcroForm'):
                    pdf.Root.AcroForm.update(PdfDict(NeedAppearances=PdfObject('true')))

                PdfWriter().write(OUTPUT_PATH, pdf)

                # Overlay retina images (first of each eye)
                right_gradcamm_path= f"data/uploads/patient_{patient_id}/gradcamm_severity_r.png"
                left_gradcamm_path=f"data/uploads/patient_{patient_id}/gradcamm_severity_l.png"
                overlay_images(
                    base_pdf_path=OUTPUT_PATH,
                    image1_path=right_eye_paths[0],
                    image1_path2=right_gradcamm_path,
                    image2_path=left_eye_paths[0],
                    image2_path2=left_gradcamm_path,
                    output_path=f"outputs/{patient_id}_DR_Report.pdf"
                )

                # st.success("Diabetic Retinopathy report generated.")
                # st.download_button("📥 Download DR Report", open(f"outputs/{patient_id}_DR_Report.pdf", "rb").read(), file_name="DR_Report.pdf")

                # Sample structure: replace these with your actual image paths
                left_eye_images = [left_eye_paths[0], left_gradcamm_path]
                right_eye_images = [right_eye_paths[0], right_gradcamm_path]

                # Append inferred images only if they exist
                left_inferred_path = f"data/uploads/patient_{patient_id}/infered_imageOD_l.jpg"
                right_inferred_path = f"data/uploads/patient_{patient_id}/infered_imageOD_r.jpg"

                if os.path.exists(left_inferred_path):
                    left_eye_images.append(left_inferred_path)

                if os.path.exists(right_inferred_path):
                    right_eye_images.append(right_inferred_path)

                patient_info = {
                    "name": patient[1],
                    "id": patient_id,
                    "sex": patient[3]
                }

                for i, j in zip(left_eye_images, right_eye_images):
                    print(f"{i[:-3]}dcm")
                    image_to_dicom(i, f"{i[:-3]}dcm", patient_info)
                    image_to_dicom(j, f"{j[:-3]}dcm", patient_info)




                def create_zip_with_folders_and_report(left_images, right_images, format_choice, patient_id):
                    zip_buffer = io.BytesIO()
                    report_path = f"outputs/{patient_id}_DR_Report.pdf"

                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        
                        def add_images_to_zip(image_paths, eye_label):
                            for path in image_paths:
                                filename = os.path.basename(path).split('.')[0]  # Get base name without extension
                                try:
                                    if format_choice == "PNG":
                                        img = Image.open(path)
                                        img_byte_arr = io.BytesIO()
                                        img.save(img_byte_arr, format='PNG')
                                        zip_file.writestr(f"{eye_label}/{filename}.png", img_byte_arr.getvalue())
                                    elif format_choice == "DICOM":
                                        dicom_path = f"data/uploads/patient_{patient_id}/{filename}.dcm"
                                        if os.path.exists(dicom_path):
                                            with open(dicom_path, "rb") as f:
                                                zip_file.writestr(f"{eye_label}/{filename}.dcm", f.read())
                                        else:
                                            st.warning(f"DICOM file not found: {dicom_path}")
                                except Exception as e:
                                    st.error(f"Error processing {path}: {e}")

                        # Add images
                        add_images_to_zip(left_images, "left_eye")
                        add_images_to_zip(right_images, "right_eye")

                        # Add DR Report
                        if os.path.exists(report_path):
                            with open(report_path, "rb") as f:
                                zip_file.writestr("report/DR_Report.pdf", f.read())
                        else:
                            st.warning(f"DR Report PDF not found at: {report_path}")

                    zip_buffer.seek(0)
                    return zip_buffer

                zip_data = create_zip_with_folders_and_report(left_eye_images, right_eye_images, format_choice, patient_id)

                st.download_button(
                    label="📥 Download Images + DR Report ZIP",
                    data=zip_data,
                    file_name=f"DR_Package_{format_choice.lower()}.zip",
                    mime="application/zip"
                )



# Main app logic
def main():
    init_db()

    # if "logged_in" not in st.session_state:
    #     st.session_state["logged_in"] = False

    # if not st.session_state["logged_in"]:
    #     login()
    # else:
    st.sidebar.title(f"Nurse Panel")
    option = st.sidebar.radio("Choose Action", ["New Patient", "Existing Patient", "Logout"])

    if option == "New Patient":
        new_patient_form()
    elif option == "Existing Patient":
        existing_patient_module()
    elif option == "Logout":
        st.session_state["logged_in"] = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()