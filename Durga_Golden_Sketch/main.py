import os
import cv2
import pygame
import numpy as np
from PIL import Image

# =========================================
# MAA DURGA GOLDEN SKETCH
# Independent contour drawing + golden glow
# =========================================

IMAGE_PATH = "durga.png"
OUTPUT_PATH = "durga_golden_sketch.mp4"

WIDTH, HEIGHT = 720, 1280
FPS = 30

DRAW_SECONDS = 50
HOLD_SECONDS = 10
TOTAL_SECONDS = DRAW_SECONDS + HOLD_SECONDS

GOLD = (0, 215, 255)
BRIGHT_GOLD = (0, 235, 255)
WHITE_GOLD = (40, 255, 255)
# =========================================
# PYGAME SETUP
# =========================================

pygame.init()

screen = pygame.display.set_mode((540, 960))
pygame.display.set_caption("Maa Durga Golden Sketch")

clock = pygame.time.Clock()

# =========================================
# LOAD IMAGE
# =========================================

if not os.path.isfile(IMAGE_PATH):
    pygame.quit()
    raise FileNotFoundError(
        "durga.png ko main.py ke same folder mein rakho."
    )

img = Image.open(IMAGE_PATH).convert("RGB")

img.thumbnail(
    (WIDTH - 60, HEIGHT - 60),
    Image.Resampling.LANCZOS
)

canvas = Image.new(
    "RGB",
    (WIDTH, HEIGHT),
    (0, 0, 0)
)

offset_x = (WIDTH - img.width) // 2
offset_y = (HEIGHT - img.height) // 2

canvas.paste(img, (offset_x, offset_y))

rgb = np.asarray(canvas)

# =========================================
# EDGE DETECTION
# =========================================

gray = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2GRAY
)

gray = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

edges = cv2.Canny(
    gray,
    40,
    110
)

# Remove outer margin
margin = 12

edges[:margin, :] = 0
edges[-margin:, :] = 0
edges[:, :margin] = 0
edges[:, -margin:] = 0

# =========================================
# FIND INDEPENDENT CONTOURS
# =========================================

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE
)

# Remove tiny noise
contours = [
    c for c in contours
    if len(c) >= 12
]

# Simplify contours slightly to remove jagged noise.
# Every contour remains independent.
clean_contours = []

for contour in contours:

    simplified = cv2.approxPolyDP(
        contour,
        epsilon=0.7,
        closed=False
    )

    points = simplified.reshape(-1, 2)

    if len(points) >= 2:
        clean_contours.append(points)

# Draw long contours first
clean_contours.sort(
    key=lambda points: cv2.arcLength(
        points.reshape(-1, 1, 2),
        False
    ),
    reverse=True
)

if not clean_contours:
    pygame.quit()
    raise RuntimeError(
        "Koi outline detect nahi hui. Source image check karo."
    )

# =========================================
# PRE-CALCULATE CONTOUR LENGTHS
# =========================================

contour_lengths = []

for points in clean_contours:

    segments = np.diff(
        points.astype(np.float32),
        axis=0
    )

    length = np.linalg.norm(
        segments,
        axis=1
    ).sum()

    contour_lengths.append(
        max(float(length), 1.0)
    )

total_length = sum(contour_lengths)

print("Contours detected:", len(clean_contours))
print("Total outline length:", int(total_length))

# =========================================
# VIDEO SETUP
# =========================================

os.makedirs("output", exist_ok=True)

video_path = os.path.join(
    "output",
    OUTPUT_PATH
)

writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (WIDTH, HEIGHT)
)

if not writer.isOpened():
    pygame.quit()
    raise RuntimeError("Video writer initialize nahi hua.")

# Persistent drawing layer
drawing = np.zeros(
    (HEIGHT, WIDTH, 3),
    dtype=np.uint8
)

# =========================================
# HELPER: DRAW PART OF ONE CONTOUR
# =========================================

def draw_partial_contour(layer, points, fraction):

    if len(points) < 2 or fraction <= 0:
        return

    if fraction >= 1:
        visible = points
    else:
        visible_count = max(
            2,
            int(len(points) * fraction)
        )

        visible = points[:visible_count]

    # Only this contour is drawn.
    # Never connect it to another contour.
    cv2.polylines(
        layer,
        [visible.reshape(-1, 1, 2)],
        False,
        GOLD,
        1,
        cv2.LINE_AA
    )

# =========================================
# MAIN ANIMATION
# =========================================

total_frames = TOTAL_SECONDS * FPS

frame_number = 0
running = True

# Store how much of each contour has been drawn
drawn_fractions = np.zeros(
    len(clean_contours),
    dtype=np.float32
)

try:

    while running and frame_number < total_frames:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                running = False

        current_time = frame_number / FPS

        progress = min(
            current_time / DRAW_SECONDS,
            1.0
        )

        # Smooth progress
        eased = (
            progress * progress * (3 - 2 * progress)
        )

        distance_to_draw = total_length * eased

        remaining_distance = distance_to_draw

        # ---------------------------------
        # DRAW CONTOURS INDEPENDENTLY
        # ---------------------------------

        for i, points in enumerate(clean_contours):

            length = contour_lengths[i]

            fraction = np.clip(
                remaining_distance / length,
                0,
                1
            )

            fraction = float(fraction)

            # Only redraw when progress changes
            if fraction > drawn_fractions[i]:

                # Redraw the current contour on the
                # persistent layer. No line joins.
                # Clear the old version of this contour
                # by rebuilding the layer below instead.

                drawn_fractions[i] = fraction

            remaining_distance -= length

        # Rebuild drawing layer to prevent old partial
        # contours from leaving unwanted marks.
        drawing.fill(0)

        for i, points in enumerate(clean_contours):

            if drawn_fractions[i] > 0:

                draw_partial_contour(
                    drawing,
                    points,
                    drawn_fractions[i]
                )

        # ---------------------------------
        # GOLD GLOW
        # ---------------------------------

        gray_drawing = cv2.cvtColor(
            drawing,
            cv2.COLOR_BGR2GRAY
        )

        glow = cv2.GaussianBlur(
            gray_drawing,
            (0, 0),
            4
        )

        wide_glow = cv2.GaussianBlur(
            gray_drawing,
            (0, 0),
            10
        )

        frame = np.zeros(
            (HEIGHT, WIDTH, 3),
            dtype=np.uint8
        )

        # Pure golden glow, BGR channels
        frame[:, :, 0] = np.clip(
        wide_glow * 0.05,
        0,
        255
        ).astype(np.uint8)

        frame[:, :, 1] = np.clip(
        glow * 0.75,
        0,
        255
        ).astype(np.uint8)

        frame[:, :, 2] = np.clip(
        glow * 1.0,
        0,
        255
        ).astype(np.uint8)

        # Sharp outlines on top
        outline_mask = gray_drawing > 0

        frame[outline_mask] = (0, 215, 255)

        # ---------------------------------
        # FINAL SKETCH
        # ---------------------------------

        if progress >= 1.0:

            frame = cv2.addWeighted(
                frame,
                1.0,
                cv2.GaussianBlur(
                    frame,
                    (0, 0),
                    3
                ),
                0.3,
                0
            )

        # ---------------------------------
        # WRITE VIDEO
        # ---------------------------------

        writer.write(frame)

        preview = cv2.resize(
            frame,
            (540, 960),
            interpolation=cv2.INTER_AREA
        )

        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        surface = pygame.surfarray.make_surface(
            np.transpose(preview_rgb, (1, 0, 2))
        )

        screen.blit(surface, (0, 0))
        pygame.display.flip()

        frame_number += 1
        clock.tick(FPS)

        if frame_number % (FPS * 5) == 0:

            print(
                f"Rendering: {frame_number // FPS}"
                f"/{TOTAL_SECONDS} seconds"
            )

finally:

    writer.release()
    pygame.quit()

print("\nSketch video created!")
print("Saved at:", video_path)