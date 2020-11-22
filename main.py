import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2


def warp(img, roi, size):
    pts1 = np.float32(roi)
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, size)


def bbox_point(bbox):
    points = []
    for box in bbox:
        points.append([box[0] + (box[2]-box[0])//2, box[3]])
    return points


def draw_points(img, points):
    for point in points:
        img = cv2.circle(img, tuple(point), 2, (0,0,255), 2)
    return img

def draw_distance_line(img, points):
    for point in points:
        distance_px = img.shape[0] - point[1]
        distance = 75 * distance_px/img.shape[0]
        img = cv2.line(img, tuple(point), (point[0], img.shape[0]), (0, 0, 255), 2)
        img = cv2.putText(img, str(distance) + ' Meter', tuple(point), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def point_distance(points, roi, size):
    pts1 = np.float32(roi)
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    warped_points = []

    for point in points:
        h_point = [point[0], point[1], 1]
        trans_h_point = matrix.dot(h_point)
        trans_h_point /= trans_h_point[2]
        warped_points.append(trans_h_point[:2].astype(int))

    return warped_points


if __name__ == '__main__':
    file_name = 'dashcam.mp4'
    fps = 30
    scale = 0.3
    cap = cv2.VideoCapture(file_name)

    width = cap.read()[1].shape[1]
    height = cap.read()[1].shape[0]
    new_width = int(scale*width)
    new_height = int(scale*height)

    roi = ((540, 510), (600, 510), (0, 630), (1020, 630))
    lane_box = (roi[0], roi[1], roi[3], roi[2])
    size = (300, 500)

    while cap.isOpened():
        res, frame = cap.read()

        # BW convert
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Warp convert
        warped = warp(bw, roi, size)
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        # YOLO Detection
        bbox, label, conf = cv.detect_common_objects(frame)
        boxed = draw_bbox(frame, bbox, label, conf)

        # Box lane
        boxed = cv2.polylines(boxed, [np.array(lane_box)], 1, (0, 255, 0), 2)

        # Point
        points = bbox_point(bbox)
        boxed = draw_points(boxed, points)

        # Distance
        warped_points = point_distance(points, roi, size)
        warped = draw_points(warped, warped_points)
        warped = draw_distance_line(warped, warped_points)

        # Show image
        cv2.imshow('real', boxed)
        cv2.imshow('warped', warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()