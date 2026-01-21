import cv2
import numpy as np
import mediapipe as mp

webcam_image = np.ndarray


class Detector:
    def __init__(self,
                 mode: bool = False,
                 num_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):

        self.mode = mode
        self.max_num_hands = num_hands
        self.complexity = model_complexity
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_num_hands,
                                         self.complexity,
                                         self.detection_confidence,
                                         self.tracking_confidence)

        self.mp_draw = mp.solutions.drawing_utils


    def findHands(self, img: webcam_image, draw_hands: bool = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw_hands:
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
        return img


    def raisedFingers(self, hand):
        fingers = []
        fingersHand = [4, 8, 12]

        if hand.landmark[fingersHand[0]].x > hand.landmark[fingersHand[0] - 1].x:
            fingers.append(False)

        else:
            fingers.append(True)

        for f in range(1, 3):
            if hand.landmark[fingersHand[f]].y < hand.landmark[fingersHand[f] - 2].y:
                fingers.append(True)

            else:
                fingers.append(False)

        return fingers


if __name__ == '__main__':
    detec = Detector()
    video = cv2.VideoCapture(0)
    blank = np.zeros((480, 640, 3), dtype='uint8')

    while True:
        success, frame = video.read()
        frame = cv2.flip(frame, 1)
        fingersRoi = []
        processFrame = detec.findHands(frame)

        if detec.results and detec.results.multi_hand_landmarks:
            hand_rigth_obj = detec.results.multi_hand_landmarks[0]
            fingersRoi = detec.raisedFingers(hand_rigth_obj)

            h, w, c = frame.shape

            ponta_polegar = hand_rigth_obj.landmark[4]
            ponta_indicador = hand_rigth_obj.landmark[8]
            ponta_meio = hand_rigth_obj.landmark[12]

            x1, y1 = int(ponta_polegar.x * w), int(ponta_polegar.y * h)
            x2, y2 = int(ponta_indicador.x * w), int(ponta_indicador.y * h)
            x3, y3 = int(ponta_meio.x * w), int(ponta_meio.y * h)

            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
            print(fingersRoi)

            if fingersRoi[1] and fingersRoi[0]:
                cv2.line(blank, (x2-1, y2-1), (x2, y2), (255, 0, 0), 8)

            if fingersRoi[2]:
                blank[:] = 0

            if not fingersRoi[0]:
                cv2.line(blank, (x2 - 1, y2 - 1), (x2, y2), (0, 0, 0), 11)

        cv2.imshow('window', frame)
        cv2.imshow('blank', blank)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
