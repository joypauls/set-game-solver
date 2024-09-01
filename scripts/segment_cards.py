import cv2
import numpy as np


MIN_CARD_AREA = 1000  # Adjust these values based on the expected size of the cards
# max_area = 5000


def display_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# order points to top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


image = cv2.imread("./images/test_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)[:, :, 0]

display_image(gray, "Grayscale Image")

# binary thresholding
_, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

display_image(binary_image, "Thresholded Image")

# countours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cards = []
for contour in contours:
    area = cv2.contourArea(contour)
    if MIN_CARD_AREA <= area:
        # approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # If the polygon has 4 points, it's likely a card
            cards.append(approx)

print(f"Detected {len(cards)} cards")

# draw contours on original image
image_copy = image.copy()
for card in cards:
    cv2.drawContours(image_copy, [card], -1, (0, 0, 255), 5)

display_image(image_copy, "Cards with Contours")

flat_cards = []
for i, card in enumerate(cards):
    pts = card.reshape(4, 2)
    rect = order_points(
        pts
    )  # order points to top-left, top-right, bottom-right, bottom-left

    # determine width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # prepare destination points for perspective transform
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    flat_cards.append(warped)

# display each card after perspective transform
for i, warped in enumerate(flat_cards):
    # cv2.imwrite(f"card_{i}.jpg", warped)
    display_image(warped, f"Card {i + 1}/{len(flat_cards)}")

for i, warped in enumerate(flat_cards):
    pass
