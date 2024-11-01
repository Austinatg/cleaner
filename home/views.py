from django.shortcuts import render, redirect
import cv2
import numpy as np
from django.http import StreamingHttpResponse

# Create your views here.

todo = 100
final_height = -1

def gen_frames(request):
    global todo
    cap1 = cv2.VideoCapture(0)  
   
    while True:
        success, frame = cap1.read()
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(todo)

        _, thresholded = cv2.threshold(gray, todo, 255, cv2.THRESH_BINARY_INV)

        contours,_ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            cv2.drawContours(img, [i], -1, (0,255,0),2)

        if not success:
            cap1.release()
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def video_feed(request):
    return StreamingHttpResponse(gen_frames(request), content_type='multipart/x-mixed-replace; boundary=frame')


def home(request):
    global todo
    if request.method == "POST":
        todo = request.POST.get('todo')
        todo = int(todo)
        print(todo)
        return redirect('video-feed')
    
    return render(request, 'home.html')




##### height detection 


def detect_ground(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    height, width = edges.shape
    # bottom_part = edges[int(0.9 * height):, :]  

    #
    # ground_contours, _ = cv2.findContours(bottom_part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # if len(ground_contours) == 0:
    # #     return None  # No ground detected

    # ground_y = height - (bottom_part.shape[0] - np.max([pt[0][1] for contour in ground_contours for pt in contour]))

    return height

def detect_object_and_measure(image):
    global final_height

    ground_y = detect_ground(image)
    if ground_y is None:
        print("Ground level not detected in the image. Please provide an image that shows the ground surface.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        object_bottom_y = y + h
        distance_in_pixels = ground_y - object_bottom_y

        # print(f"Distance from object to ground in pixels: {distance_in_pixels}")
        cv2.putText(image, f"Distance from object to ground in pixels: {distance_in_pixels}", 
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        final_height = distance_in_pixels

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.line(image, (0, ground_y), (image.shape[1], ground_y), (255, 0, 0), 2)  
        # cv2.imshow('Object Detection', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("No object detected.")


# image = cv2.imread(r'C:\Users\purus\Downloads\img5.jpeg')


# detect_object_and_measure(image) 



def height(request):
    cap1 = cv2.VideoCapture(0)  
   
    while True:
        success, frame = cap1.read()
        image = frame
        detect_object_and_measure(image)

        if not success:
            cap1.release()
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def video_feed_2(request):
    return StreamingHttpResponse(height(request), content_type='multipart/x-mixed-replace; boundary=frame')


