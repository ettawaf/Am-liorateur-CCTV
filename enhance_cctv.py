import cv2
import numpy as np
import os
import argparse

def enhance_opencv(image_path, output_path):
    # تحميل الصورة
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 2) تحسين التباين (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # 3) شحذ (Unsharp Mask)
    gaussian = cv2.GaussianBlur(enhanced, (9,9), 10.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    cv2.imwrite(output_path, unsharp)
    print(f"[+] OpenCV enhancement saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="CCTV Image Enhancer")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("-o", "--output", default="outputs", help="Output folder")
    parser.add_argument("--gfpgan", action="store_true", help="Use GFPGAN if installed")
    parser.add_argument("--realesrgan", action="store_true", help="Use Real-ESRGAN if installed")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # تحسين بالـ OpenCV (أساسي)
    opencv_out = os.path.join(args.output, "enhanced_opencv.png")
    enhance_opencv(args.input, opencv_out)

    # تحسين بالذكاء الاصطناعي (اختياري إذا مثبت)
    if args.gfpgan:
        print("[!] Trying GFPGAN (requires installation and pretrained model)...")
        os.system(f"python3 inference_gfpgan.py -i {args.input} -o {args.output} --version 1.3")

    if args.realesrgan:
        print("[!] Trying Real-ESRGAN (requires installation and pretrained model)...")
        os.system(f"python3 inference_realesrgan.py -n RealESRGAN_x4plus -i {args.input} -o {args.output}")

if __name__ == "__main__":
    main()
