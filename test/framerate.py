import cv2
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

video = None

# Start default camera
if args.get("video", None) is None:
    video = cv2.VideoCapture(0)
    time.sleep(0.25)
else:
    video = cv2.VideoCapture(args["video"])

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.

if int(major_ver) < 3:
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
else:
    fps = video.get(cv2.CAP_PROP_FPS)
    print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

# Number of frames to capture
num_frames = 120

print "Capturing {0} frames".format(num_frames)

# Start time
start = time.time()

# Grab a few frames
for i in xrange(0, num_frames):
    (grabbed, frame) = video.read()
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# End time
end = time.time()

# Time elapsed
seconds = end - start
print "Time taken : {0} seconds".format(seconds)

# Calculate frames per second
fps = num_frames / seconds;
print "Estimated frames per second : {0}".format(fps);

# Release video
video.release()

# close all windows opened by cv2
cv2.destroyAllWindows()
