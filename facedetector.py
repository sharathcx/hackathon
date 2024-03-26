import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection


class FaceDetector:
    def __init__(self):
        pass

    def save_roi_videos(self, input_video_path, output_dir, rois):
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writers = {}
        for roi_id, (x, y, w, h) in rois.items():
            video_writers[roi_id] = cv2.VideoWriter(
                os.path.join(output_dir, f"roi_{roi_id}.mp4"), fourcc, 30, (w, h)
            )

        # Process input video and save ROIs
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for roi_id, (x, y, w, h) in rois.items():
                roi = frame[y:y + h, x:x + w]

                video_writers[roi_id].write(roi)

        # Release video writers
        for video_writer in video_writers.values():
            video_writer.release()

        cap.release()

    def detect_and_save_rois(self, input_video_path, output_dir):
        cap = cv2.VideoCapture(input_video_path)
        os.makedirs(output_dir, exist_ok=True)

        with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as facedetector:
            rois = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = facedetector.process(rgb_frame)

                if results.detections:
                    for idx, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(
                            bboxC.height * ih)
                        roi_scale_factor = 2
                        x -= int(w * (roi_scale_factor - 1) / 2)
                        y -= int(h * (roi_scale_factor - 1) / 2)
                        w *= roi_scale_factor
                        h *= roi_scale_factor
                        rois[idx] = (x, y, w, h)

            cap.release()
            self.save_roi_videos(input_video_path, output_dir, rois)


input_video_path = 'test2.mp4'
output_dir = 'face_videos'

detector = FaceDetector()
detector.detect_and_save_rois(input_video_path, output_dir)
