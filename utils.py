import cv2 as cv


def extract_descriptors(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    harris_laplace_detector = \
        cv.xfeatures2d.HarrisLaplaceFeatureDetector_create()
    kp_hl = harris_laplace_detector.detect(gray, None)

    sift = cv.xfeatures2d.SIFT_create()
    _, descriptors = sift.compute(gray, kp_hl)

    return descriptors
