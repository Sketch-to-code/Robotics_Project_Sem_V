import cv2
import numpy as np
import sys
import os


def detect_cubes(image_path, output_dir="detections", debug=True):
    img = cv2.imread(image_path)
    if img is None:
         raise FileNotFoundError(f"Could not read image: {image_path}")
    os.makedirs(output_dir, exist_ok = True)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(
         blurred,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5),np.uint8)
    clean = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations=2)
    clean = cv2.morphologyEx(clean,cv2.MORPH_OPEN,kernel,iterations=1)

    contours, _ = cv2.findContours(
         clean,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    MIN_AREA = 1500
    MAX_AREA = 200000

    cubes=[]
    for i, cnt in enumerate(contours):
         area = cv2.contourArea(cnt)
         if area < MIN_AREA or area > MAX_AREA:
              continue
         x,y,w,h = cv2.boundingRect(cnt)

         aspect_ratio = w/float(h)
         if aspect_ratio < 0.6 or aspect_ratio >1.6:
              continue
         cx,cy = x+w//2, y+h//2

         margin = int(0.15*min(w,h))
         digit_crop = img[y+margin:y+h-margin, x+margin:x+w-margin]

         cube_info={
              "id": i,
              "bbox":(x,y,w,h),
              "centroid_px": (cx,cy),
              "digit_crop": digit_crop,
         }
         cubes.append(cube_info)

         crop_path = os.path.join(output_dir,f"cube_{i}_digit.png")
         cv2.imwrite(crop_path,digit_crop)

         if debug:
              cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
              cv2.circle(img,(cx,cy),4,(0,0,255),-1)
              cv2.putText(
                   img,f"#{i}",(x,y-10),
                   cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2
              )

         if debug:
            debug_path = os.path.join(output_dir, "debug_annotated.png")
            cv2.imwrite(debug_path, img)
            thresh_path = os.path.join(output_dir, "debug_threshold.png")
            cv2.imwrite(thresh_path, clean)
            print(f"Saved annotated image to {debug_path}")
            print(f"Saved threshold mask to {thresh_path} (check this if detection looks off)")

    print(f"Detected {len(cubes)} cube(s).")
    for c in cubes:
        print(f"  cube #{c['id']}: bbox={c['bbox']}  centroid(px)={c['centroid_px']}")

    return cubes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 detect_cubes.py path/to/photo.jpg")
        sys.exit(1)

    detect_cubes(sys.argv[1])