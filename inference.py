from torch import det
from models.mtcnn import MTCNN
from PIL import Image
import os
from glob import glob

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN()
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

def detect(input_file_path, save_path=None):
    img_org=Image.open(input_file_path)
    if img_org.mode != "RGB":
        img = img_org.convert("RGB")
    else:
        img = img_org
    img_cropped = mtcnn(img, save_path=save_path)
    return img_org, img_cropped

def crop_faces(pil_img, img_cropped, margin=None):
    faces = []
    for i, quad in enumerate(img_cropped):
        left, top, right, bottom = quad
        width = right - left
        height = bottom - top
        max_edge = max(width, height)
        # 正方形に正規化
        left = left + (width - max_edge) / 2
        right = left + max_edge
        top = top + (height - max_edge) / 2
        bottom = top + max_edge
        if margin is not None:
            m_top, m_left, m_right, m_bottom = margin
            top = top - max_edge * m_top
            left = left - max_edge * m_left
            right = right + max_edge * m_right
            bottom = bottom + max_edge * m_bottom
        
        face = pil_img.crop((left, top, right, bottom))
        faces.append(face)
    return faces


if __name__ == "__main__":
    input_dir = "/home/radius5/Downloads/6/01_aspcom_alpha_matting_matting"
    save_dir = "/home/radius5/Downloads/6/02_aspcom_detect"
    os.makedirs(save_dir, exist_ok=True)
    input_file_path_list = glob(input_dir + "/*.png")
    for input_file in input_file_path_list:
        save_path = input_file.replace(input_dir, save_dir)
        pil_img, img_cropped = detect(input_file, save_path)
        if img_cropped is None:
            img_cropped = []

        pil_crop_faces = crop_faces(pil_img, img_cropped, margin=[1.0, 2.0, 2.0, 3.0])
        for i, pil_crop_face in enumerate(pil_crop_faces):
            pil_crop_face.save(save_path.replace(".png", "_" + str(i) + ".png"))
        print(img_cropped)
        print(img_cropped.shape)