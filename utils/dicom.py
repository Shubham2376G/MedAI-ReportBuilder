import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage
from PIL import Image
import numpy as np
from datetime import datetime

def image_to_dicom(image_path, output_path, patient_info):
    # Load and convert image to RGB
    image = Image.open(image_path).convert("RGB")
    pixel_array = np.array(image)

    # Create File Meta Information
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    # Create DICOM dataset
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Add patient data
    ds.PatientName = patient_info.get("name", "Anonymous")
    ds.PatientID = patient_info.get("id", "000000")
    # ds.PatientBirthDate = patient_info.get("dob", "")
    ds.PatientSex = patient_info.get("sex", "")

    # Add study data
    dt = datetime.now()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')
    ds.Modality = 'OT'
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    # Add image data (RGB)
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.Rows, ds.Columns, _ = pixel_array.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0  # 0 = RGBRGBRGB..., 1 = RRR...GGG...BBB...
    ds.PixelData = pixel_array.tobytes()

    # Save the DICOM file
    ds.save_as(output_path)
    print(f"✅ DICOM file saved: {output_path}")
