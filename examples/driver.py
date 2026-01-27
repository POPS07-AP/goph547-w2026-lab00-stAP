import numpy as np

# 1. Create an array of ones with 3 rows and 5 columns
arr1 = np.ones((3, 5))
print("1. Array of ones (3x5):\n", arr1, "\n")

# 2. Produce an array of NaN with 6 rows and 3 columns
arr2 = np.full((6, 3), np.nan)
print("2. Array of NaN (6x3):\n", arr2, "\n")

# 3. Create a column vector of odd numbers between 44 and 75
# Odd numbers: 45, 47, 49, ..., 75
odd_col = np.arange(45, 76, 2).reshape(-1, 1)
print("3. Column vector of odd numbers between 44 and 75:\n", odd_col, "\n")

# 4. Find the sum of the vector produced in #3
odd_sum = np.sum(odd_col)
print("4. Sum of the odd number column vector:", odd_sum, "\n")

# 5. Produce the array A
A = np.array([[5, 7, 2],
              [1, -2, 3],
              [4, 4, 4]])
print("5. Array A:\n", A, "\n")

# 6. Using a single command, produce the array B (identity matrix)
B = np.eye(3, dtype=int)
print("6. Array B:\n", B, "\n")

# 7. Element-wise multiplication of A and B
elem_mult = A * B
print("7. Element-wise multiplication A * B:\n", elem_mult, "\n")

# 8. Dot product (matrix multiplication) of A and B
dot_prod = A @ B
print("8. Dot product (A @ B):\n", dot_prod, "\n")

# 9. Cross product of A and B
# Note: cross product is applied row-wise for 3D vectors
cross_prod = np.cross(A, B)
print("9. Cross product of A and B (row-wise):\n", cross_prod)





#5 Working with matplotlib for visualization

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------
# 10. Load the image
# -------------------------------------------------
img = np.asarray(Image.open("rock_canyon.jpg"))
print("10. Original image shape:", img.shape)

# -------------------------------------------------
# 11. Plot the image
# -------------------------------------------------
plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("rock_canyon.jpg")
plt.show()

# -------------------------------------------------
# 12. Re-open the image in grayscale
# -------------------------------------------------
gray_img = np.asarray(Image.open("rock_canyon.jpg").convert("L"))
print("12. Grayscale image shape:", gray_img.shape)

plt.figure()
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
plt.title("rock_canyon.jpg (Grayscale)")
plt.show()

# -------------------------------------------------
# 13. Crop a smaller grayscale image (pinnacle area)
#    Use fractions of the image size so it never
#    goes out of bounds.
# -------------------------------------------------
h, w = gray_img.shape
print("Image height:", h, "Image width:", w)

# Crop roughly middle-left region
row_start = int(0.25 * h)
row_end   = int(0.75 * h)
col_start = int(0.05 * w)
col_end   = int(0.35 * w)

small_gray_image = gray_img[row_start:row_end, col_start:col_end]
print("13. Small grayscale image shape:", small_gray_image.shape)

plt.figure()
plt.imshow(small_gray_image, cmap="gray")
plt.axis("off")
plt.title("Small Grayscale Image (Pinnacle Area)")
plt.show()

# -------------------------------------------------
# 14. RGB summaries
# -------------------------------------------------
# Separate channels
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

# Means along y-direction (collapse rows → function of x)
R_mean_x = np.mean(R, axis=0)
G_mean_x = np.mean(G, axis=0)
B_mean_x = np.mean(B, axis=0)
RGB_mean_x = np.mean(img, axis=(0, 2))

# Means along x-direction (collapse columns → function of y)
R_mean_y = np.mean(R, axis=1)
G_mean_y = np.mean(G, axis=1)
B_mean_y = np.mean(B, axis=1)
RGB_mean_y = np.mean(img, axis=(1, 2))

# -------------------------------------------------
# 15. Make subplots with labels, titles, legends
# -------------------------------------------------
plt.figure(figsize=(12, 6))

# (i) Mean colour vs x-coordinate
plt.subplot(1, 2, 1)
plt.plot(R_mean_x, color="red", label="Red")
plt.plot(G_mean_x, color="green", label="Green")
plt.plot(B_mean_x, color="blue", label="Blue")
plt.plot(RGB_mean_x, color="black", linewidth=2, label="Mean RGB")
plt.xlabel("X coordinate (pixels)")
plt.ylabel("Mean colour value")
plt.title("Mean Colour vs X Coordinate")
plt.legend()

# (ii) Mean colour vs y-coordinate
plt.subplot(1, 2, 2)
plt.plot(R_mean_y, color="red", label="Red")
plt.plot(G_mean_y, color="green", label="Green")
plt.plot(B_mean_y, color="blue", label="Blue")
plt.plot(RGB_mean_y, color="black", linewidth=2, label="Mean RGB")
plt.xlabel("Y coordinate (pixels)")
plt.ylabel("Mean colour value")
plt.title("Mean Colour vs Y Coordinate")
plt.legend()

plt.tight_layout()

# -------------------------------------------------
# 16. Save the figure
# -------------------------------------------------
plt.savefig("rock_canyon_RGB_summary.png")
print("16. Saved rock_canyon_RGB_summary.png")

plt.show()
