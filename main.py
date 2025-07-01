import cv2
import time
import matplotlib.pyplot as plt
from simple_facerec import SimpleFacerec

simFacRec = SimpleFacerec()
simFacRec.load_encoding_images("CriminalDatabase/")

cap = cv2.VideoCapture(0)  # Loading the first available webcam

fps_values = []
start_time = time.time()

frame_count = 0
total_time = 0
execution_times = []


while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_timee = time.time()

    face_locations, face_names = simFacRec.detect_known_faces(frame)  # Detect Faces

    # Stop timer and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_timee

    # Update performance variables
    frame_count += 1
    total_time += execution_time
    execution_times.append(execution_time)

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]  # y1=top, x2=right, y2=bottom, x1=left
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    cv2.imshow("Criminal Recognition System", frame)
    fps_values.append(1 / (time.time() - start_time))
    start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot the performance graph
plt.plot(fps_values)
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.title("Face Recognition Performance")



plt.show()

# Calculate average execution time per frame
avg_execution_time = total_time / frame_count

# Generate performance graph
plt.plot(range(1, frame_count + 1), execution_times)
plt.xlabel('Frame')
plt.ylabel('Execution Time (s)')
plt.title('Face Recognition Performance')
plt.axhline(y=avg_execution_time, color='r', linestyle='--', label='Average Time: {:.4f}s'.format(avg_execution_time))
plt.legend()
plt.show()
