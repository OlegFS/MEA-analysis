
#%%
%matplotlib widget


# Developing code to coregister the stiched image with digitalized electrode position for further anatomical labelling
import numpy as np
import matplotlib.pyplot as plt
image_path = '/Users/ovinogradov/Documents/Projects/SCN1A/MEA-analysis/data/slice_images/28-11-24/slice1/28-11-24-slice1-fused.tif'
I = plt.imread(image_path)


t_path = '/Users/ovinogradov/Documents/Projects/SCN1A/MEA-analysis/data/templates/MEA256.tif'
template = plt.imread(t_path)
#%%
# %matplotlib inline
fig = plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Reference Image")
plt.imshow(template)

# coordinates1 = [730,480]
# plt.plot(coordinates1[0],coordinates1[1],'.r')
coords = []
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print (f'x = {ix}, y = {iy}')

    global coords
    coords.append((ix, iy))
    
    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
#%%

t_path = '/Users/ovinogradov/Documents/Projects/SCN1A/MEA-analysis/data/templates/MEA256.tif'
template = plt.imread(t_path)

# %matplotlib inline
fig = plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Reference Image")
plt.imshow(I)

# coordinates1 = [730,480]
# plt.plot(coordinates1[0],coordinates1[1],'.r')
coords2 = []
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print (f'x = {ix}, y = {iy}')

    global coords2
    coords2.append((ix, iy))
    
    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)

    return coords2
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

#%% 
na = np.array
coords = na(coords)
coords2 = na(coords2)
# %%
fig = plt.figure(figsize=(5, 5))
plt.subplot(1, 3, 1)
plt.title("Reference Image")
plt.plot(coords[:,0],coords[:,1],'r')
plt.imshow(template)
plt.subplot(1, 3, 2)
plt.title("Target Image")
plt.plot(coords2[:,0],coords2[:,1],'r')
plt.imshow(I)
# %%
import cv2
import numpy as np

# Load the images
image1 = cv2.imread(t_path)
image2 = cv2.imread(image_path)

# Manually select corresponding points in both images
# Format: (x, y) coordinates for image1 and image2
# points_image1 = np.float32([[100, 100], [200, 100], [100, 200]])  # Example points in image1
# points_image2 = np.float32([[150, 150], [250, 150], [150, 250]])  # Corresponding points in image2

# Compute the affine transformation matrix
matrix = cv2.getAffineTransform(na(coords,dtype=np.float32), na(coords2,dtype=np.float32))
# matrix = cv2.getAffineTransform(points_image1, points_image2)

# Apply the affine transformation to image1
aligned_image = cv2.warpAffine(image2, matrix, (image2.shape[1], image2.shape[0]))

# Show the result
# cv2.imshow('Aligned Image', aligned_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#%% 
# %matplotlib inline
plt.figure(figsize=(7, 5))
plt.subplot(1, 3, 1)
plt.title("Reference Image")
plt.imshow(template)

coordinates1 = [730,480]
plt.plot(coordinates1[0],coordinates1[1],'.r')
# plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Target Image")
plt.imshow(I)
plt.imshow(template,alpha=0.2)

# plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Aligned Image")
plt.imshow(aligned_image)
# plt.axis('off')
plt.tight_layout()
# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(t_path)  # Replace with your image path

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, binary_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY_INV)
# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to retain only circular dots
dots = []
structures =[]
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if area > 10:  # Filter very small noise
        # Circularity = 4π * Area / Perimeter² (ideal circle = 1.0)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if 0.7 < circularity <= 1.0:  # Adjust range for "roundness"
            dots.append(contour)

output_image = image.copy()

# cv2.drawContours(output_image, dots, -1, (0, 255, 0), 2)

# Extract dot centers for plotting or further processing
dot_centers = [cv2.moments(dot) for dot in dots]
dot_coordinates = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in dot_centers if m["m00"] != 0]

# Plot the result
plt.figure(figsize=(8, 8))
plt.imshow(output_image)
plt.scatter([c[0] for c in dot_coordinates], [c[1] for c in dot_coordinates], color='red', s=2, label='Detected Dots')
plt.legend()
plt.title("Filtered Dots")
plt.show()

contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lines = []
for contour in contours:
    if len(contour)>10:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
        aspect_ratio = max(w / h, h / w)  # Calculate aspect ratio (ensure it's >= 1)
        
        if aspect_ratio > 1.2:  # Filter based on aspect ratio (adjust threshold as needed)
            lines.append(contour)

# Draw filtered lines on the original image
# output_image = image.copy()
# cv2.drawContours(output_image, lines, -1, (0, 255, 0), 2)

#%%
# Plot the result
plt.figure(figsize=(8, 8))
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.imshow(template)
for line in lines:
    plt.plot(line[:,0,0],line[:,0,1],'k')
# plt.title("Filtered Line Contours")
plt.plot([c[0] for c in dot_coordinates], [c[1] for c in dot_coordinates], 'r.')
plt.legend()
# plt.title("Filtered Dots")
# plt.show()
plt.axis('off')
# plt.savefig('template_dig.pdf')


#%%


letters = ['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','R']
# Set the labels for each subplot
labels = []
exclude = [
'A1','R1','A16','R16'
]
for j in range(16):
    for i in range(16):
        label = f"{letters[i]}{j + 1}"
        if label not in exclude:
            labels.append(label)


plt.figure()
plt.imshow(template)
sorted_coordinates = na(sorted(dot_coordinates, key=lambda coord: (coord[1], coord[0])))
plt.plot([c[0] for c in sorted_coordinates], [c[1] for c in sorted_coordinates], 'r.')

[plt.text(x,y,s=labels[i],fontsize=5)  for i,(x,y) in enumerate(sorted_coordinates)]
#%%
import cv2
import numpy as np

# Define original points
# points = np.float32([[100, 100], [200, 100], [100, 200]])

# Define corresponding points in the new coordinate space
# new_points = np.float32([[150, 150], [250, 150], [150, 250]])
import pickle
with open('../../data/electrodes_template2.pickle', 'wb') as handle:
    pickle.dump({'pos':sorted_coordinates,'e_labels':labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)




def coregister_electrodes(points,tempate_points,template_e_position):
    matrix = cv2.getAffineTransform(na(tempate_points,dtype=np.float32), na(points,dtype=np.float32))
    e_position = na(template_e_position['pos'])
    mapped_e_position = np.dot(matrix, np.vstack((e_position.T, np.ones((1, e_position.shape[0])))))
    return mapped_e_position





# Compute the affine transformation matrix
matrix = cv2.getAffineTransform(na(coords,dtype=np.float32), na(coords2,dtype=np.float32))
dot_coordinates = na(dot_coordinates)

# Map the original points to the new coordinates
mapped_points = np.dot(matrix, np.vstack((dot_coordinates.T, np.ones((1, dot_coordinates.shape[0])))))

# Convert back to 2D points
mapped_points = mapped_points[:2].T
#%%
plt.figure()
print("Mapped Points:")
print(mapped_points)
plt.plot(mapped_points[:,0],mapped_points[:,1],'.r')
plt.imshow(I)
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# labels = [
#     [f"{letters[j]}{i + 1}" for j in range(16)] for i in range(16)
# ]
# remove extra labels 


plt.plot(sorted_coordinates[:,0],sorted_coordinates[:,1],'.')
for i,lab in enumerate(np.hstack(labels)):
    plt.text(sorted_coordinates[i,0],sorted_coordinates[i,1],lab,fontsize=6)

# Plot circles with text in each subplot
# electrode_dict = {}
#             circle = patches.Circle((0.5, 0.5), radius=0.4,facecolor=[1,1,1],edgecolor='skyblue')#, color='skyblue')
#             ax.add_patch(circle)
        # electrode_dict[labels[i][j]] =ax



# %%
#%%
# Plot the result
plt.figure(figsize=(8, 10))
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

plt.subplot(3,2,1)
plt.imshow(template,cmap='gray')
plt.title('reference')
plt.axis('off')

plt.subplot(3,2,2)
plt.imshow(template,cmap='gray')
plt.title('slice')
plt.axis('off')


plt.subplot(3,2,3)
plt.imshow(template,cmap='gray')
plt.title('reference')
plt.axis('off')
plt.plot(coords[:,0],coords[:,1],'xr')


# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.subplot(3,2,4)
plt.imshow(I,cmap='gray')
plt.plot(coords2[:,0],coords2[:,1],'xr')
plt.title('slice')
plt.axis('off')

plt.subplot(3,2,5)
plt.imshow(template,cmap='gray')
for line in lines:
    plt.plot(line[:,0,0],line[:,0,1],'r')
# plt.title("Filtered Line Contours")
plt.plot([c[0] for c in dot_coordinates], [c[1] for c in dot_coordinates], 'C0.')
plt.title('detected electrodes')
# plt.legend()
# plt.title("Filtered Dots")
# plt.show()
plt.axis('off')

plt.subplot(3,2,6)
plt.imshow(I,cmap='gray')
plt.plot(mapped_points[:,0],mapped_points[:,1],'.C0')
# plt.imshow(I)
plt.title('aligned electrodes')
plt.axis('off')




# %% Image stitching 
import json
path = '/Users/ovinogradov/Documents/Projects/SCN1A/MEA-analysis/data/slice_images/28-11-24/slice2/stack/stack_MMStack_5-Pos000_001_metadata.txt'
directory = '/Users/ovinogradov/Documents/Projects/SCN1A/MEA-analysis/data/slice_images/28-11-24/slice2/stack/'
# np.read(path)
d2 = json.load(open(path))

n_files = len(d2['Summary']['StagePositions'])

#compute extent
xs = []
ys = []
positions = []
for i in range(2):
    file_name  = 'stack_MMStack_'+d2['Summary']['StagePositions'][0]['Label']
    # x = d2['Summary']['StagePositions'][i]['GridRow'] #[0]['Position_um']
    # y = d2['Summary']['StagePositions'][i]['GridCol'] #[0]['Position_um']
    x,y = d2['Summary']['StagePositions'][i]['DevicePositions'][0]['Position_um']
    # y = d2['Summary']['StagePositions'][i]['GridCol'] #[0]['Position_um']
    xs.append(x)
    ys.append(y)
    positions.append([x,y])
    # print(x,y)
    # img = plt.imread(directory+file_name+'.ome.tif')

x_extent = np.max(np.abs(np.diff(xs)))
y_extent = np.max(np.abs(np.diff(ys)))

plt.figure()
file_names = []
for i in range(2):
    file_name  = directory+'stack_MMStack_'+d2['Summary']['StagePositions'][i]['Label']+'.ome.tif'
    file_names.append(file_name)

    # x,y = d2['Summary']['StagePositions'][i]['DevicePositions'][0]['Position_um']
    # img = plt.imread(directory+file_name+'.ome.tif')
    # plt.imshow(img,extent=[x,x+x_extent , y,y+y_extent])

#%%
position =na(positions) -np.min(na(positions),0)
position *=0.5

#%%
na = np.array
#%%
from PIL import Image

def stitch_images(image_paths, positions):
    """
    Stitches images together based on their (x, y) positions.

    :param image_paths: List of file paths to images.
    :param positions: List of (x, y) coordinates corresponding to image positions.
    :param output_path: Path to save the final stitched image.
    """
    # Load all images
    images = [Image.open(path) for path in image_paths]

    # Get number of rows and columns
    rows = max(r for r, _ in positions) + 1
    cols = max(c for _, c in positions) + 1

    # Get max image width and height
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Calculate canvas size
    canvas_width = cols * max_width
    canvas_height = rows * max_height
    print(images[0].width ,images[0].height)

    # Create a blank canvas
    canvas = Image.new("I", (canvas_width, canvas_height), (0))
    # canvas = Image.new(canvas_mode, (canvas_width, canvas_height), background_color)


    # Paste images onto the canvas at the specified positions
    # for img, (x, y) in zip(images, positions):
    #     canvas.paste(img, (x, y), mask=img if img.mode == 'L' else None)

    for img, (row, col) in zip(images, positions):
        x = col * max_width
        y = row * max_height
        canvas.paste(img, (x, y))

    # Save the final stitched image
    # canvas.save(output_path)
    # print(f"Stitched image saved to {output_path}")
    return canvas


# Example usage:
# image_paths = ["img1.png", "img2.png", "img3.png"]  # Replace with your image paths
# positions = [(0, 0), (100, 50), (200, 150)]  # Replace with your image positions
# output_path = "stitched_output.png"

canvas = stitch_images(file_names, np.array(positions,dtype=int))


# %%
from stitching import AffineStitcher

print(AffineStitcher.AFFINE_DEFAULTS)
settings = {# The whole plan should be considered
            "crop": False,
            # The matches confidences aren't that good
            "confidence_threshold": .005}    

stitcher = AffineStitcher(**settings)
panorama = stitcher.stitch(image_paths)
# %%
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

# input data (can be any numpy compatible array: numpy, dask, cupy, etc.)
tile_arrays = [np.random.randint(0, 100, (2, 10, 100, 100)) for _ in range(3)]

# indicate the tile offsets and spacing
tile_translations = [
    {"z": 2.5, "y": -10, "x": 30},
    {"z": 2.5, "y": 30, "x": 10},
    {"z": 2.5, "y": 30, "x": 50},
]
spacing = {"z": 2, "y": 0.5, "x": 0.5}

channels = ["DAPI", "GFP"]

# build input for stitching
msims = []
for tile_array, tile_translation in zip(tile_arrays, tile_translations):
    sim = si_utils.get_sim_from_array(
        tile_array,
        dims=["c", "z", "y", "x"],
        scale=spacing,
        translation=tile_translation,
        transform_key="stage_metadata",
        c_coords=channels,
    )
    msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
# %%
# plot the tile configuration
from multiview_stitcher import vis_utils
fig, ax = vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False)
# %%
