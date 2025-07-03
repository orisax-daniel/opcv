import cv2


face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_haar_cascade.empty():
    raise IOError("Cannot load haarcascade_frontalface_default.xml")

image = cv2.imread("mazi.jpg")
if image is None:
    raise IOError("Cannot load mazi.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY", gray)
faces = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

if len(faces) == 0:
    print("No faces detected.")
else:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

cv2.imshow('faces', image)
cv2.imwrite("faces_detected.jpg", image)
print(f"Saved result image as faces_detected.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()