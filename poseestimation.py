# TechVidvan Human pose estimator
# import necessary packages

import cv2
import mediapipe as mp


# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)




# create capture object
cap = cv2.VideoCapture('pixel.mp4')  # 'video.mp4'
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
        # Check if any landmarks are found.
        if results.pose_landmarks:

            # Iterate two times as we only want to display first two landmarks.
            for i in range(33):
                # Display the found normalized         landmarks.
                print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')

                thisxlist.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x)
                thisylist.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y)

                # thislist.append(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
                # thislist.append(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')


                # print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
                # print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')

        maxx = int(max(thisxlist) * framewidth)
        maxy = int(max(thisylist) * frameheight)
        minx = int(min(thisxlist) * framewidth)
        miny = int(min(thisylist) * frameheight)
        # print(minx)
        # print(miny)
        # print(maxx)
        # print(maxy)



        #print(results.pose_landmarks)
        # draw detected skeleton on the frame

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #print(mp_pose.PoseLandmark.name)
        # show the final output'

        frame = cv2.resize(frame, (1600, 900))
        cv2.imwrite('F:/poseestimation/crop/Frame' + str(m) + '.jpg', frame)
        img = cv2.imread('F:/poseestimation/crop/Frame' + str(m) + '.jpg')

        if 900-maxy > 30:
            cropped_image = img[900-maxy+30:maxy+40, minx-40:maxx+40] # top: height, minx:maxx  0:900
        else:
            cropped_image = img[0:maxy - miny, minx - 40:maxx + 40]  # top: height, minx:maxx  0:900
        cv2.imwrite("F:/poseestimation/crop/crop/CroppedImage"+ str(m) + ".jpg", cropped_image)
        m += 1
        cv2.imshow('Output', frame)
    except:
        break
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
