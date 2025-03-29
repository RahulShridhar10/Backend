import cv2
from cv2 import dnn_superres
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys

def init_super_res(model_path: str):
    global sr, modelname, modelscale
    sr = dnn_superres.DnnSuperResImpl_create()
    
    modelname = model_path.split(os.path.sep)[-1].split("_")[0].lower()
    modelscale = model_path.split("_x")[-1]
    modelscale = int(modelscale[:modelscale.find(".")])
    
    print(f"[info] Loading super resolution model {modelname}...")
    print(f"[info] Model name: {modelname}")
    print(f"[info] Model scale: {modelscale}")
    
    sr.readModel(model_path)
    sr.setModel(modelname, modelscale)

def plot_results(image_list: list, title_list: list):
    _, axs = plt.subplots(1, len(image_list))
    axs = axs.flatten()
    for img, ax, title in zip(image_list, axs, title_list):
        ax.set_title(title)
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.tight_layout()
    # plt.show()

def bicubic_upsample(input_image: np.ndarray, out_path: str, out_image: str):
    print("[info] Upscaling using bicubic interpolation...")
    print(f"[info] w: {input_image.shape[1]}, h: {input_image.shape[0]}")
    
    start = time.time()
    upscaled_bicubic = cv2.resize(
        input_image, (input_image.shape[1]*modelscale, input_image.shape[0]*modelscale), interpolation=cv2.INTER_CUBIC)
    end = time.time()
    
    print(f"[info] Bicubic upscaling took {end - start:.6f} seconds")
    
    if len(upscaled_bicubic.shape) == 2:  # Grayscale image
        cv2.imwrite(os.path.join(out_path, "Bicubic_" + out_image), upscaled_bicubic)
    else:
        upscaled_bicubic = cv2.cvtColor(upscaled_bicubic, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_path, "Bicubic_" + out_image), upscaled_bicubic)
        upscaled_bicubic = cv2.cvtColor(upscaled_bicubic, cv2.COLOR_BGR2RGB)
    
    return upscaled_bicubic

def super_res(input_image: np.ndarray, out_path: str, out_image: str):
    print(f"[info] w: {input_image.shape[1]}, h: {input_image.shape[0]}")
    
    start = time.time()
    upscaled = sr.upsample(input_image)
    end = time.time()
    
    print(f"[info] Super resolution took {end - start:.6f} seconds")
    print(f"[info] w: {upscaled.shape[1]}, h: {upscaled.shape[0]}")
    
    if len(upscaled.shape) == 2:  # Grayscale image
        cv2.imwrite(os.path.join(out_path, out_image), upscaled)
    else:
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_path, out_image), upscaled)
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    
    return upscaled

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to super resolution model")
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    ap.add_argument("-o", "--output", required=True, help="Path to output directory")
    ap.add_argument("-B", "--Bicubic", action='store_true', help="Use Bicubic interpolation")
    args = vars(ap.parse_args())
    
    if not os.path.exists(args["model"]):
        print("[error] Model file does not exist!")
        sys.exit()
    if not os.path.exists(args["image"]):
        print("[error] Image file does not exist!")
        sys.exit()
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])
    
    image_name = os.path.basename(args["image"]).split(".")[0]
    init_super_res(args["model"])
    
    try:
        image = cv2.imread(args["image"])
        if image is None:
            print("[error] Failed to load image. Check the file format.")
            sys.exit()
        
        if len(image.shape) == 2:
            gray = True
        else:
            gray = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        out_images_list = []
        title_list = []
        
        if args["Bicubic"]:
            bicubic_image = bicubic_upsample(image, args["output"], image_name + "_Bicubic.jpg")
            out_images_list.append(bicubic_image)
            title_list.append("Bicubic_x" + str(modelscale))
        
        out_images_list.append(image)
        title_list.append("Original")
        
        upscaled_SR = super_res(image, args["output"], f"{image_name}_{modelname}_x{modelscale}.jpg")
        out_images_list.append(upscaled_SR)
        title_list.append(f"SR {modelname}_x{modelscale}")
        
        plot_results(out_images_list, title_list)
    except Exception as e:
        print(f"[error] {str(e)}")
