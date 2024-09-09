import cv2
import os

# โหลดตัวตรวจจับใบหน้า Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# โหลดตัวจำแนกใบหน้าด้วย LBPHFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

# สร้างโฟลเดอร์เพื่อบันทึกภาพใบหน้า
output_folder = "faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ตัวแปรเพื่อเก็บข้อมูลใบหน้า
face_count = {}  # {face_key: count}
face_ids = {}  # {face_key: id}
face_id = 0
last_seen_faces = set()

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็นโทนสีเทา
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้าในภาพ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Set สำหรับเก็บใบหน้าที่ถูกตรวจจับในปัจจุบัน
    current_faces = set()
    for (x, y, w, h) in faces:
        # สร้างคีย์สำหรับใบหน้าในปัจจุบัน
        face_key = (x // 10, y // 10, w // 10, h // 10)  # ใช้ทูเพิลที่มีขนาดลดลงเพื่อการเปรียบเทียบ
        current_faces.add(face_key)

        # ตรวจสอบว่าใบหน้าเป็นใบหน้าใหม่หรือไม่ หรือใบหน้าเก่าที่ออกไปแล้วกลับมา
        if face_key not in last_seen_faces:
            if face_key not in face_ids:
                # กำหนด ID ใหม่ให้ใบหน้าใหม่
                face_id += 1
                face_ids[face_key] = face_id
                face_count[face_id] = 1
            else:
                # เพิ่มจำนวนครั้งสำหรับใบหน้าเก่าที่กลับมา
                face_count[face_ids[face_key]] += 1

            # ตัดเฉพาะส่วนใบหน้าจากภาพที่ตรวจจับได้
            face_image = gray[y:y + h, x:x + w]

            # บันทึกภาพใบหน้า
            cv2.imwrite(f"{output_folder}/face_{face_ids[face_key]}_count_{face_count[face_ids[face_key]]}.jpg", face_image)

        # วาดสี่เหลี่ยมรอบใบหน้าและแสดง ID และจำนวนครั้ง
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {face_ids[face_key]} Count: {face_count[face_ids[face_key]]}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # อัพเดตรายการใบหน้าที่เห็นล่าสุด
    last_seen_faces = current_faces

    # แสดงภาพที่กล้องจับได้
    cv2.imshow('Face Detection', frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างแสดงผล
cap.release()
cv2.destroyAllWindows()
