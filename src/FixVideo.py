import cv2

video_path = "annotated_video_output.mp4"
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# opening the writing to the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("annotated_video_output2.mp4", fourcc, fps, (width, height))

for i in range(frame_count):

    print(f"Processing frame {i}")

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    save_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out.write(save_bgr)

print(frame_count)

cap.release()
out.release()

