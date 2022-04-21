import cv2
import mediapipe as mp
import numpy as np


# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)




# create capture object
cap = cv2.VideoCapture('10.mp4')  # 'video.mp4'
m = 0
framewidth = 1600
frameheight = 900


while cap.isOpened():
    # read frame from capture object
    _, frame = cap.read()

    try:
        # convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = pose.process(RGB)

        # Perform pose detection after converting the image into RGB format.
        #results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        thisxlist = []
        thisylist = []
        newthisylist=[]
        lefteye = 0
        righteye = 0
        leftelbow = 0
        rightelbow = 0
        leftwrist = 0
        rightwrist = 0
        leftshoulder = 0
        rightshoulder = 0
        leftear = 0
        rightear = 0

        # Check if any landmarks are found.
        if results.pose_landmarks:

            # Iterate two times as we only want to display first two landmarks.
            for i in range(33):
                # Display the found normalized         landmarks.
                if mp_pose.PoseLandmark(i).name=="LEFT_EYE":
                    lefteye  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="RIGHT_EYE":
                    righteye  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="LEFT_ELBOW":
                    leftelbow  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="RIGHT_ELBOW":
                    rightelbow  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="LEFT_WRIST":
                    leftwrist  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="RIGHT_WRIST":
                    rightwrist  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="LEFT_SHOULDER":
                    leftshoulder  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="RIGHT_SHOULDER":
                    rightshoulder  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="LEFT_EAR":
                    leftear  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y
                if mp_pose.PoseLandmark(i).name=="RIGHT_EAR":
                    rightear  = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y


                print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')


                thisxlist.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x)
                thisylist.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y)

            newthisylist = [lefteye, righteye, leftelbow, rightelbow, leftwrist, rightwrist, leftshoulder, rightshoulder, leftear, rightear]




                # thislist.append(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
                # thislist.append(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')


                # print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
                # print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')

        maxx = int(max(thisxlist) * framewidth)
        maxy = int(max(thisylist) * frameheight)
        minx = int(min(thisxlist) * framewidth)
        miny = int(min(thisylist) * frameheight)
        anothermaxy = int(min(newthisylist) * frameheight)

        # print(minx)
        # print(miny)
        # print(maxx)
        # print(maxy)

        frame = cv2.resize(frame, (1600, 900))
        #print(results.pose_landmarks)
        # draw detected skeleton on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec((255, 0, 0), 2, 2),
                                  mp_drawing.DrawingSpec((255, 0, 255), 2, 2)
                                  )
        # Display pose on original video/live stream : optional
        cv2.imshow("Pose Estimation", frame)

        # Extractraction
        # get shape of original frame
        h, w, c = frame.shape
        # create blank image with original frame size
        #frame = np.zeros([h, w, c])

        # set white background. put 0 if you want to make it black
        frame.fill(255)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec((255, 0, 0),5, 5),
                               mp_drawing.DrawingSpec((255, 0, 255), 5, 5)
                               )
        #print(mp_pose.PoseLandmark.name)
        # show the final output'


        cv2.imwrite('F:/poseestimation/10/Frame' + str(m) + '.jpg', frame)
        img = cv2.imread('F:/poseestimation/10/Frame' + str(m) + '.jpg')

        # if 900-maxy > 30:
        #     cropped_image = img[30:900, minx-40:maxx+40] # top: height, minx:maxx  0:900
        # else:
        #     cropped_image = img[0:maxy - miny, minx - 40:maxx + 40]  # top: height, minx:maxx  0:900

        # RIGHT_WRIST
        # print(anothermaxy)
        # print("\n")
        # print(900-anothermaxy)
        cropped_image = img[anothermaxy:maxy, minx - 40:maxx + 40]  # top: height, minx:maxx  0:900
        cv2.imwrite("F:/poseestimation/10/crop/CroppedImage"+ str(m) + ".jpg", cropped_image)
        m += 1
        cv2.imshow('Output', frame)
    except:
        break
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


