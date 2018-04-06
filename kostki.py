import cv2
import numpy as np
from scipy.spatial import distance

kernel_3 = np.ones((3,3),np.uint8)
kernel_5 = np.ones((5,5),np.uint8)
kernel_circle = cv2.getStructuringElement(ksize = (3,3),shape = cv2.MORPH_ELLIPSE)
stosunek = 3.65

def szukaj_oczka(normal_frame,gray_frame,wykrywacz):

    bloby = wykrywacz.detect(gray_frame)
    # frame_with_blobs = cv2.drawKeypoints(gray_frame,bloby,np.array([]),(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('WITH_BLOBS', frame_with_blobs)
    grupy = grupuj_oczka(bloby)
    for grupa in grupy:
        x = [i.pt[0] for i in grupa]
        y = [i.pt[1] for i in grupa]
        srednia = (sum(x) / len(grupa), sum(y) / len(grupa))
        wlk = grupa[0].size
        wlk = wlk * stosunek
        point1 = (int(srednia[0] - wlk), int(srednia[1] - wlk))
        point2 = (int(srednia[0] + wlk), int(srednia[1] + wlk))
        cv2.putText(normal_frame, str(len(grupa)), point2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.rectangle(normal_frame, point1, point2, (0,0,255), 1)

    return normal_frame

def grupuj_oczka(bloby):
    grupy = []
    i = 0
    while len(bloby) != 0:
        grupy.append([bloby[0]])
        bloby.pop(0)
        j = 0
        while j < len(bloby):
            odl = distance.euclidean(bloby[j].pt, grupy[i][0].pt)
            if(odl <= grupy[i][0].size * stosunek):
                grupy[i].append(bloby[j])
                bloby.pop(j)
            else:
                j += 1
        i += 1
    return grupy

def main():

    cap = cv2.VideoCapture(0)
    parametry = cv2.SimpleBlobDetector_Params()

    parametry.filterByArea = True
    parametry.minArea = 33
    parametry.filterByCircularity = True
    parametry.minCircularity = 0.66
    parametry.maxCircularity = 1
    parametry.filterByInertia = True
    parametry.minInertiaRatio = 0.66
    parametry.maxInertiaRatio = 1

    wykrywacz_oczek = cv2.SimpleBlobDetector_create(parametry)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('szare', gray)
        gray_frame = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,53,28)
        cv2.imshow('thresh', gray_frame)
        frame = szukaj_oczka(frame,gray_frame,wykrywacz_oczek)
        cv2.imshow('result',frame)

        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()