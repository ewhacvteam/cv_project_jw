import cv2
import mediapipe as mp
import numpy as np
import os
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 이미지 파일이 있는 폴더 경로
folder_path = r"C:\Users\user\Desktop\climb_dataset\climb_jiu"

# 폴더 내의 이미지 파일 목록 가져오기
image_files = [os.path.join(folder_path, f"climb_{i}.png") for i in range(2,4)]

# IMAGE_FILES에 추가
IMAGE_FILES = image_files

BG_COLOR = (192,192,192)
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    for idx, file in enumerate(IMAGE_FILES):
        image=cv2.imread(file)
        image = cv2.resize(image,(250,250))
        image_height, image_width, _ = image.shape

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue
        # 포즈가 감지된 부분은 그대로, 감지되지 않으면 회색 배경색으로
        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,)*3,axis=-1)>0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

        # 3개의 키포인트 간의 각도 계산 2D
        def calculateAngle(k1,k2,k3):
            x1 = k1.x * 250
            y1 = k1.y * 250
            x2 = k2.x * 250
            y2 = k2.y * 250
            x3 = k3.x * 250
            y3 = k3.y * 250

            result = math.degrees(
                math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2)
            )
            if result < 0:
                result+=360
            return result
        
        # 3개의 키포인트 간의 각도 계산 3D
        def calculate_3d(k1,k2,k3):
            vector_1_2=[(k2[i]-k1[i]) for i in range(3)]
            vector_2_3=[(k2[i]-k3[i]) for i in range(3)]

            dot_product = sum([vector_1_2[i]*vector_2_3[i] for i in range(3)])
            v_1_2_mag = math.sqrt(sum([coord**2 for coord in vector_1_2]))
            v_2_3_mag = math.sqrt(sum([coord**2 for coord in vector_2_3]))

            angle = math.degrees(math.acos(dot_product / (v_1_2_mag * v_2_3_mag)))

            return angle

        
        right_angle_2d=[]
        right_angle_3d=[]
        left_angle_2d=[]
        left_angle_3d=[]
        # 신체 우측 관절 각도
        for right in range(12,29,2):
            k1=results.pose_landmarks.landmark[right]
            k2=results.pose_landmarks.landmark[right+2]
            k3=results.pose_landmarks.landmark[right+4]

            result_2d=calculateAngle(k1,k2,k3)     
            result_3d=calculate_3d(k1,k2,k3)     
            right_angle_2d.append(result_2d)
            right_angle_3d.append(result_3d)
            print(f"right_angle[{right}]: {result_2d}")    #right_angle[시작 keypoint] ex.right_angle[12] -> 12-14-16 연결 각도
            print(f"right_angle[{right}]: {result_3d}")    #right_angle[시작 keypoint] ex.right_angle[12] -> 12-14-16 연결 각도


        # 신체 좌측 관절 각도
        for left in range(11,28,2):
            k1=results.pose_landmarks.landmark[left]
            k2=results.pose_landmarks.landmark[left+2]
            k3=results.pose_landmarks.landmark[left+4]

            result_2d=calculateAngle(k1,k2,k3)     
            result_3d=calculate_3d(k1,k2,k3)     
            left_angle_2d.append(result_2d)
            left_angle_3d.append(result_3d)
            print(f"left_angle[{left}]: {result_2d}")    #right_angle[시작 keypoint] ex.right_angle[12] -> 12-14-16 연결 각도
            print(f"left_angle[{left}]: {result_3d}")

        # 결과 시각화
        mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # 이미지에 포즈 랜드마크 그리기
        cv2.imshow(str(idx) + '.png',annotated_image)
        # 포즈 월드 랜드마크를 그리기
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
