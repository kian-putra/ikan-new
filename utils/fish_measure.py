from ultralytics import YOLO

import cv2
import numpy as np
import os


# =========================================================
# MAIN FUNCTION
# =========================================================

def measure_fish(image_path, model_path):

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = YOLO(model_path)

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    output = image.copy()

    # =====================================================
    # PREDICTION
    # =====================================================

    results = model.predict(
        source=image,
        conf=0.1,
        save=False,
        verbose=False
    )

    result = results[0]

    if result.masks is None:
        raise ValueError("No object detected")

    class_names = result.names

    # =====================================================
    # FIND SCALE
    # =====================================================

    REAL_SCALE_CM = 20.0

    scale_found = False
    cm_per_pixel = None

    for i, polygon in enumerate(result.masks.xy):

        cls_id = int(result.boxes.cls[i])
        cls_name = class_names[cls_id]

        if cls_name.lower() != "skala":
            continue

        scale_found = True

        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        pts = np.array(
            polygon,
            dtype=np.int32
        )

        cv2.fillPoly(
            mask,
            [pts],
            255
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        rect = cv2.minAreaRect(cnt)

        (center_x, center_y), (w, h), angle = rect

        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.drawContours(
            output,
            [box],
            0,
            (255, 0, 255),
            3
        )

        scale_pixel_length = max(w, h)

        cm_per_pixel = (
            REAL_SCALE_CM /
            scale_pixel_length
        )

        # ===============================================
        # DRAW SCALE LINE
        # ===============================================

        if w > h:

            dx = np.cos(
                np.radians(angle)
            )

            dy = np.sin(
                np.radians(angle)
            )

        else:

            dx = -np.sin(
                np.radians(angle)
            )

            dy = np.cos(
                np.radians(angle)
            )

        half_length = scale_pixel_length / 2

        x1 = int(center_x - dx * half_length)
        y1 = int(center_y - dy * half_length)

        x2 = int(center_x + dx * half_length)
        y2 = int(center_y + dy * half_length)

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            4
        )

        cv2.circle(
            output,
            (x1, y1),
            8,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            output,
            (x2, y2),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            "20 cm",
            (
                int(center_x),
                int(center_y)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        break

    if not scale_found:
        raise ValueError(
            "Class 'skala' not detected"
        )

    # =====================================================
    # FISH PROCESSING
    # =====================================================

    # menyimpan seluruh hasil sebelum filtering
    all_fish_data = []

    all_lengths = []

    fish_counter = 1

    for idx, polygon in enumerate(result.masks.xy):

        cls_id = int(result.boxes.cls[idx])

        cls_name = class_names[cls_id]

        if cls_name.lower() == "skala":
            continue

        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        pts = np.array(
            polygon,
            dtype=np.int32
        )

        cv2.fillPoly(
            mask,
            [pts],
            255
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            output,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        # ===============================================
        # PCA
        # ===============================================

        ys, xs = np.where(mask > 0)

        points = np.column_stack(
            (xs, ys)
        )

        if len(points) < 10:
            continue

        data_pts = np.float32(points)

        mean, eigenvectors, eigenvalues = cv2.PCACompute2(
            data_pts,
            mean=np.array([])
        )

        center = mean[0]

        direction = eigenvectors[0]

        projections = np.dot(
            points - center,
            direction
        )

        min_proj = projections.min()
        max_proj = projections.max()

        head = center + direction * min_proj
        tail = center + direction * max_proj

        head = head.astype(int)
        tail = tail.astype(int)

        pixel_length = np.linalg.norm(
            head - tail
        )

        fish_length_cm = (
            pixel_length *
            cm_per_pixel
        )

        # ===============================================
        # SAVE DATA
        # ===============================================

        fish_info = {

            "fish_id": fish_counter,

            "species": cls_name,

            "length_cm": round(
                float(fish_length_cm),
                2
            ),

            "head": head,

            "tail": tail,

            "center": center,

            "contour": cnt
        }

        all_fish_data.append(
            fish_info
        )

        all_lengths.append(
            float(fish_length_cm)
        )

        fish_counter += 1

        
    # =====================================================
# OUTLIER FILTER
# Median ±50%
# =====================================================

    fish_data = []
    fish_lengths = []

    if len(all_lengths) > 0:

        median_length = np.median(
            all_lengths
        )

        lower_limit = (
            median_length * 0.5
        )

        upper_limit = (
            median_length * 1.5
        )

        for fish in all_fish_data:

            length = fish["length_cm"]

            if (
                lower_limit
                <= length
                <= upper_limit
            ):

                fish_data.append(
                    fish
                )

                fish_lengths.append(
                    length
                )

        print("\n========== OUTLIER FILTER ==========")
        print(
            f"Median Length : {median_length:.2f} cm"
        )
        print(
            f"Lower Limit   : {lower_limit:.2f} cm"
        )
        print(
            f"Upper Limit   : {upper_limit:.2f} cm"
        )
        print(
            f"Before Filter : {len(all_fish_data)} fish"
        )
        print(
            f"After Filter  : {len(fish_data)} fish"
        )
    
    # =====================================================
    # DRAW FILTERED FISH ONLY
    # =====================================================

    for fish in fish_data:

        head = fish["head"]
        tail = fish["tail"]
        center = fish["center"]
        contour = fish["contour"]

        length_cm = fish["length_cm"]

        cv2.drawContours(
            output,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        cv2.line(
            output,
            tuple(head),
            tuple(tail),
            (0, 255, 255),
            4
        )

        cv2.circle(
            output,
            tuple(head),
            8,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            output,
            tuple(tail),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            f"{length_cm:.2f} cm",
            (
                int(center[0]),
                int(center[1])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    if len(fish_lengths) > 0:

        summary = {
            "total_fish": len(fish_lengths),
            "average_length": round(
                np.mean(fish_lengths),
                2
            ),
            "min_length": round(
                np.min(fish_lengths),
                2
            ),
            "max_length": round(
                np.max(fish_lengths),
                2
            )
        }

    else:

        summary = {
            "total_fish": 0,
            "average_length": 0,
            "min_length": 0,
            "max_length": 0
        }

    # =====================================================
    # CLEAN DATA FOR HTML
    # =====================================================

    fish_data_clean = []

    for fish in fish_data:

        fish_data_clean.append({

            "fish_id": fish["fish_id"],

            "species": fish["species"],

            "length_cm": fish["length_cm"]
        })
        # =====================================================
    # SAVE RESULT IMAGE
    # =====================================================

    # root project
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    # folder hasil
    results_dir = os.path.join(
        BASE_DIR,
        "static",
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    filename = os.path.basename(
        image_path
    )

    result_filename = (
        "result_" + filename
    )

    # path fisik untuk cv2.imwrite
    result_path = os.path.join(
        results_dir,
        result_filename
    )

    # simpan gambar
    saved = cv2.imwrite(
        result_path,
        output
    )

    if not saved:
        raise ValueError(
            f"Failed to save result image: {result_path}"
        )

    # path untuk browser Flask
    result_web_path = (
        "results/" +
        result_filename
    )

    # =====================================================
    # RETURN
    # =====================================================

    return (
        result_web_path,
        fish_data_clean,
        summary
    )


    # =====================================================
    # RETURN
    # =====================================================
