
import cv2
import numpy as np
import math

# HSV color ranges (adjust according to your lighting)
BLUE_LOWER = np.array([90, 80, 50])
BLUE_UPPER = np.array([135, 255, 255])

BALL_LOWER = np.array([20, 80, 80])
BALL_UPPER = np.array([45, 255, 255])

MIN_BLUE_AREA = 300
MIN_BALL_AREA = 150

selected_blue_center = None
previous_blue_center = None
blue_tracking_points = []
selected_blue_contour = None


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )
    return mask


def find_contours(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def mouse_callback(event, x, y, flags, param):
    global selected_blue_center, previous_blue_center
    global blue_tracking_points, selected_blue_contour

    if event == cv2.EVENT_LBUTTONDOWN:
        frame, contours = param

        # Select a blue contour at the clicked location.
        for contour in contours:
            if cv2.contourArea(contour) < MIN_BLUE_AREA:
                continue

            inside = cv2.pointPolygonTest(
                contour, (float(x), float(y)), False
            )

            if inside >= 0:
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                selected_blue_center = (cx, cy)
                previous_blue_center = (cx, cy)
                blue_tracking_points = [(cx, cy)]
                selected_blue_contour = contour
                break


def detect_tennis_ball(mask):
    contours = find_contours(mask)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_BALL_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        # The ball should be approximately circular.
        if circularity < 0.55:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        if radius < 5:
            continue

        candidates.append(
            (area, contour, (int(x), int(y)), int(radius))
        )

    if not candidates:
        return None

    # Choose the largest plausible tennis-ball candidate.
    return max(candidates, key=lambda item: item[0])


def main():
    global selected_blue_center, previous_blue_center
    global blue_tracking_points, selected_blue_contour

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window = "Blue Object and Tennis Ball Detection"
    cv2.namedWindow(window)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read camera frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_mask = clean_mask(
            cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        )
        ball_mask = clean_mask(
            cv2.inRange(hsv, BALL_LOWER, BALL_UPPER)
        )

        blue_contours = [
            c for c in find_contours(blue_mask)
            if cv2.contourArea(c) >= MIN_BLUE_AREA
        ]

        # Detect tennis ball independently of the blue object.
        ball = detect_tennis_ball(ball_mask)
        ball_center = None

        if ball is not None:
            ball_area, ball_contour, ball_center, radius = ball

            cv2.circle(
                frame, ball_center, radius, (0, 255, 255), 2
            )
            cv2.circle(
                frame, ball_center, 4, (0, 0, 255), -1
            )

            cv2.putText(
                frame,
                f"Ball center: {ball_center}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"Radius: {radius}px  Area: {ball_area:.0f}px^2",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )

        # Keep tracking the selected blue object.
        blue_contour = None

        if blue_contours:
            if previous_blue_center is None:
                # Before selection, show available blue objects.
                for c in blue_contours:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h),
                        (255, 150, 0), 1
                    )
            else:
                # Find the contour nearest to the previous center.
                def contour_center(c):
                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        return None
                    return (
                        int(M["m10"] / M["m00"]),
                        int(M["m01"] / M["m00"])
                    )

                candidates = []
                for c in blue_contours:
                    center = contour_center(c)
                    if center is not None:
                        distance = math.dist(
                            previous_blue_center, center
                        )
                        candidates.append((distance, c, center))

                if candidates:
                    distance, blue_contour, blue_center = min(
                        candidates, key=lambda item: item[0]
                    )

                    # Reject a candidate that jumps too far away.
                    if distance > 150:
                        blue_contour = None
                    else:
                        selected_blue_center = blue_center

        if blue_contour is not None:
            x, y, w, h = cv2.boundingRect(blue_contour)
            area = cv2.contourArea(blue_contour)

            M = cv2.moments(blue_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                selected_blue_center = (cx, cy)

                cv2.drawContours(
                    frame, [blue_contour], -1, (255, 0, 0), 2
                )
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (255, 0, 0), 2
                )
                cv2.circle(
                    frame, (cx, cy), 4, (255, 255, 255), -1
                )

                cv2.putText(
                    frame,
                    f"Blue area: {area:.0f}px^2",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 100, 0), 2
                )
                cv2.putText(
                    frame,
                    f"BBox: x={x}, y={y}, w={w}, h={h}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 100, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Blue center: ({cx}, {cy})",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 100, 0), 2
                )

                # Draw the blue object's center trajectory.
                if (
                    previous_blue_center is not None
                    and previous_blue_center != (cx, cy)
                ):
                    blue_tracking_points.append((cx, cy))

                for i in range(1, len(blue_tracking_points)):
                    cv2.line(
                        frame,
                        blue_tracking_points[i - 1],
                        blue_tracking_points[i],
                        (255, 0, 255), 2
                    )

                # Draw a separate line to the tennis ball.
                if ball_center is not None:
                    cv2.line(
                        frame, (cx, cy), ball_center,
                        (0, 255, 0), 2
                    )

                    dx = cx - ball_center[0]
                    dy = cy - ball_center[1]
                    distance = math.hypot(dx, dy)

                    cv2.putText(
                        frame,
                        f"Relative to ball: dx={dx}, dy={dy}",
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"Center distance: {distance:.1f}px",
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2
                    )

                previous_blue_center = (cx, cy)

        # Mouse selection uses the current frame and its contours.
        cv2.setMouseCallback(
            window, mouse_callback, (frame.copy(), blue_contours)
        )

        cv2.imshow(window, frame)
        cv2.imshow("Blue Mask", blue_mask)
        cv2.imshow("Tennis Ball Mask", ball_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        # Press r to reset blue-object selection.
        if key == ord("r"):
            selected_blue_center = None
            previous_blue_center = None
            blue_tracking_points = []
            selected_blue_contour = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
