import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from skimage.draw import line as skimage_line
from projection import Camera, RadialPolyCamProjection, CylindricalProjection, read_cam_from_json, \
    create_img_projection_maps
from ultralytics import YOLO


def make_cylindrical_cam(cam: Camera):
    """generate a cylindrical camera with a centered horizon"""
    assert isinstance(cam.lens, RadialPolyCamProjection)
    # create a cylindrical projection
    lens = CylindricalProjection(cam.lens.coefficients[0])

    rot_zxz = SciRot.from_matrix(cam.rotation).as_euler('zxz')
    # adjust all angles to multiples of 90 degree
    rot_zxz = np.round(rot_zxz / (np.pi / 2)) * (np.pi / 2)
    # center horizon
    rot_zxz[1] = np.pi / 2

    return Camera(
        rotation=SciRot.from_euler(angles=rot_zxz, seq='zxz').as_matrix(),
        translation=cam.translation,
        lens=lens,
        size=cam.size,
        principle_point=(cam.cx_offset, cam.cy_offset),
        aspect_ratio=cam.aspect_ratio
    )


def calculate_straightness(points):
    x, y = points[:, 0], points[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    distances = np.abs(k*x - y + b) / np.sqrt(k**2 + 1)
    return np.mean(distances), np.var(distances)


def get_line_points(polyline, step=1):
    # Bresenham method
    points = []
    for i in range(len(polyline)-1):
        rr, cc = skimage_line(int(polyline[i][1]), int(polyline[i][0]),
                              int(polyline[i+1][1]), int(polyline[i+1][0]))
        points.extend(list(zip(cc, rr)))
    seen = set()
    out = []
    for pt in points[::step]:
        if pt not in seen:
            out.append(pt)
            seen.add(pt)
    return out


def get_normal(dx, dy):
    length = np.hypot(dx, dy)
    if length == 0:
        return np.array([0,0])
    return np.array([-dy/length, dx/length])


def mean_contrast_along_polyline(image, polyline, edge_half_width=1, side_width=5):
    points = get_line_points(polyline)
    contrasts = []
    if image.ndim == 3:
        h, w, _ = image.shape
    if image.ndim == 2:
        h, w = image.shape
    for idx, (x, y) in enumerate(points):
        if idx == 0:
            x1, y1 = points[idx]
            x2, y2 = points[idx+1]
        elif idx == len(points) - 1:
            x1, y1 = points[idx-1]
            x2, y2 = points[idx]
        else:
            x1, y1 = points[idx-1]
            x2, y2 = points[idx+1]
        dx, dy = x2 - x1, y2 - y1
        normal = get_normal(dx, dy)
        if np.all(normal == 0):
            continue

        edge_pixels = []
        for offset in range(-edge_half_width, edge_half_width+1):
            px = int(round(x + normal[0]*offset))
            py = int(round(y + normal[1]*offset))
            if 0 <= px < w and 0 <= py < h:
                edge_pixels.append(image[py, px])

        left_pixels = []
        for offset in range(-side_width-edge_half_width, -edge_half_width):
            px = int(round(x + normal[0]*offset))
            py = int(round(y + normal[1]*offset))
            if 0 <= px < w and 0 <= py < h:
                left_pixels.append(image[py, px])

        right_pixels = []
        for offset in range(edge_half_width+1, side_width+edge_half_width+1):
            px = int(round(x + normal[0]*offset))
            py = int(round(y + normal[1]*offset))
            if 0 <= px < w and 0 <= py < h:
                right_pixels.append(image[py, px])

        if edge_pixels and left_pixels and right_pixels:
            edge_mean = np.mean(edge_pixels)
            side_mean = (np.mean(left_pixels) + np.mean(right_pixels)) / 2
            contrasts.append(abs(edge_mean - side_mean))

    if len(contrasts) == 0:
        return 0
    return np.mean(contrasts)


def yolo_object_recognition(img, filename=None):
    model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
    results = model(img, stream=True)  # Run batched inference on a list of images and return a generator of Results objects

    detection_text = "Object recognition result:\n"
    for result_idx, result in enumerate(results):
        boxes = result.boxes
        vis_img = img.copy()

        valid_detections = 0
        for box_idx in range(len(boxes)):
            xyxy = boxes.xyxy[box_idx].cpu().numpy()
            cls_id = int(boxes.cls[box_idx])
            conf = float(boxes.conf[box_idx])

            if conf > 0.45:
                class_name = model.names[cls_id]
                detection_text += (
                    f"Object {valid_detections + 1}:\n"
                    f"  Class: {class_name}\n"
                    f"  Conf: {conf:.2f}\n"
                    f"  Boundary: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]\n"
                )

                x1, y1, x2, y2 = map(int, xyxy[:4])
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 255), 6)
                label = f"{class_name} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                text_x = x1 - 15
                text_y = y2 + 15
                if text_y - text_height < 0:
                    text_y = text_height + 5
                cv2.rectangle(vis_img,
                              (text_x, text_y - text_height - 5),
                              (text_x + text_width, text_y + 5),
                              (255, 255, 255),
                              -1)
                cv2.putText(vis_img,
                            label,
                            (text_x, text_y),
                            font,
                            font_scale,
                            (0, 0, 200),
                            thickness)

                valid_detections += 1

        if valid_detections > 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Object recognition')
            plt.axis('off')
            plt.savefig(filename)
            plt.show()
            print(detection_text)
        else:
            print(f'Object not found in {result_idx + 1}')


'''Distortion correction'''
# generate camera instances, load example image, and re-project it to a central cylindrical projection
fisheye_cam = read_cam_from_json('front.json')
original_ratio = fisheye_cam.size[0] / fisheye_cam.size[1]  # width/length
cylindrical_cam = make_cylindrical_cam(fisheye_cam)
fisheye_image = cv2.imread('rgb_00599_FV.png')
fisheye_image_copy = fisheye_image.copy()
fisheye_gray = cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2GRAY) # for edge detection

if os.path.exists('scale_tmp.txt'):
    os.remove('scale_tmp.txt')
map1, map2 = create_img_projection_maps(fisheye_cam, cylindrical_cam)
cylindrical_image = cv2.remap(fisheye_image, map1, map2, cv2.INTER_CUBIC)
cylindrical_image_copy = cylindrical_image.copy()
cyl_height, cyl_width = cylindrical_image.shape[:2]
os.rename('scale_tmp.txt', 'scale_values.txt')

# crop
pt1 = (197, 140)  # top left of cropped figure
pt2 = (1090, 140)  # top right of cropped figure
cropped = cylindrical_image[pt1[1]:(pt1[1] + int((pt2[0] - pt1[0]) / original_ratio)), pt1[0]:pt2[0]]
cropped_copy = cylindrical_image.copy()[pt1[1]:(pt1[1] + int((pt2[0] - pt1[0]) / original_ratio)), pt1[0]:pt2[0]]

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2RGB))
plt.title(f'Fisheye\n{fisheye_cam.size[0]}x{fisheye_cam.size[1]}')
plt.axis('off')
plt.imsave('Results/Fig5b.pdf', cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2RGB))

plt.subplot(132)
plt.imshow(cv2.cvtColor(cylindrical_image, cv2.COLOR_BGR2RGB))
plt.title(f'Distortion correction\n{cyl_width}x{cyl_height}')
plt.axis('off')
plt.imsave('Results/FigS18a.pdf', cv2.cvtColor(cylindrical_image, cv2.COLOR_BGR2RGB))

plt.subplot(133)
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.title(f'Distortion correction (cropped)\n{cropped.shape[1]}x{cropped.shape[0]}')
plt.axis('off')
plt.tight_layout()
plt.imsave('Results/FigS18b.pdf', cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.show()

'''Quantify straightness before and after distortion correction'''
# Fig.5c
cyl_contours = [
    np.array([[316, 428], [318, 350], [319, 281], [320, 212]]),
    np.array([[785, 363], [784, 309], [783, 255], [783, 189]]),
    np.array([[980, 249], [933, 208], [878, 167], [825, 130]]),
    np.array([[1052, 461], [1052, 389], [1052, 316], [1047, 240]]),
    np.array([[538, 366], [507, 370], [486, 374], [457, 379]])
]

MAE_before_list = []
MAE_after_list = []
for i, cyl_contour in enumerate(cyl_contours):
    world_points = cylindrical_cam.project_2d_to_3d(cyl_contour, norm=np.ones(cyl_contour.shape[0]))
    fisheye_contour = fisheye_cam.project_3d_to_2d(world_points)

    cv2.polylines(cylindrical_image, [cyl_contour.astype(int)], False, (58, 56, 216), 2)
    cv2.polylines(fisheye_image, [fisheye_contour.astype(int)], False, (102, 210, 243), 2)

    pre_error, _ = calculate_straightness(fisheye_contour)
    post_error, _ = calculate_straightness(cyl_contour)
    MAE_before_list.append(pre_error)
    MAE_after_list.append(post_error)

    padding = 25
    x1_cyl = max(0, np.min(cyl_contour[:, 0]) - padding)
    y1_cyl = max(0, np.min(cyl_contour[:, 1]) - padding)
    x2_cyl = min(cylindrical_image.shape[1], np.max(cyl_contour[:, 0]) + padding)
    y2_cyl = min(cylindrical_image.shape[0], np.max(cyl_contour[:, 1]) + padding)
    cropped_cyl = cylindrical_image[y1_cyl:y2_cyl, x1_cyl:x2_cyl]

    x1_fish = max(0, np.min(fisheye_contour[:, 0]) - padding)
    y1_fish = max(0, np.min(fisheye_contour[:, 1]) - padding)
    x2_fish = min(fisheye_image.shape[1], np.max(fisheye_contour[:, 0]) + padding)
    y2_fish = min(fisheye_image.shape[0], np.max(fisheye_contour[:, 1]) + padding)
    cropped_fish = fisheye_image[int(y1_fish):int(y2_fish), int(x1_fish):int(x2_fish)]

    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(cropped_fish, cv2.COLOR_BGR2RGB))
    plt.title(f'Fisheye (cropped) \n MAE:{pre_error:.2f}')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(cropped_cyl, cv2.COLOR_BGR2RGB))
    plt.title(f'Distortion correction (cropped) \n MAE:{post_error:.2f}')
    plt.axis('off')
    plt.savefig(f'Results/Fig5c_{i+1}.pdf')

plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2RGB))
plt.title('Fig.5b. Fisheye')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(cylindrical_image, cv2.COLOR_BGR2RGB))
plt.title('Distortion correction')
plt.axis('off')
plt.tight_layout()
plt.savefig(f'Results/Fig5b_1.pdf')
plt.show()

# Fig.5e
cyl_contours_2 = [
    np.array([[161,314], [160,354], [157,394], [156,419]]),
    np.array([[212,179], [205,251], [202,335], [198,417]]),
    np.array([[245,163], [241,254], [236,332], [234,424]]),
    np.array([[290,185], [289,269], [286,340], [282,423]]),
    np.array([[349,98], [349,163], [349,217], [348,289]]),
    np.array([[423,94], [422,203], [421,282], [420,329]]),
    np.array([[433,328], [432,361], [432,399], [432,432]]),
    np.array([[520,406], [520,428], [520,463], [521,499]]),
    np.array([[554,379], [553,413], [553,454], [553,489]]),
    np.array([[682,350], [683,387], [683,438], [683,467]]),
    np.array([[696,293], [697,325], [698,372], [699,412]]),
    np.array([[784,184], [785,245], [785,321], [786,365]]),
    np.array([[812,173], [814,239], [815,306], [817,355]]),
    np.array([[872,219], [872,278], [872,336], [873,408]]),
    np.array([[895,284], [895,317], [896,366], [896,409]]),
    np.array([[953,313], [954,343], [954,384], [955,420]]),
    np.array([[1010,160], [1013,261], [1016,372], [1018,456]]),
    np.array([[1047,241], [1052,320], [1052,386], [1054,464]]),
    np.array([[324,120], [348,139], [371,159], [419,199]]),
    np.array([[285,273], [329,293], [373,313], [418,333]]),
    np.array([[247,387], [353,403], [379,409], [418,417]]),
    np.array([[311,545], [384,537], [440,527], [474,522]]),
    np.array([[565,498], [625,498], [691,499], [762,501]]),
    np.array([[722,315], [735,281], [756,239], [777,189]]),
    np.array([[711,449], [727,439], [751,428], [779,412]]),
    np.array([[883,135], [911,159], [947,193], [972,217]]),
    np.array([[905,218], [942,248], [979,277], [1009,303]]),
    np.array([[889,229], [938,262], [973,290], [1003,314]]),
    np.array([[909,285], [939,304], [980,328], [1012,348]]),
    np.array([[899,331], [935,346], [972,362], [1008,380]]),
    np.array([[832,430], [892,438], [943,443], [986,449]]),
    np.array([[107,456], [172,457], [234,458], [293,463]]),
    np.array([[663,556], [746,551], [832,544], [888,538]]),
    np.array([[1017,336], [1064,329], [1105,325], [1141,321]]),
    np.array([[1024,471], [1063,469], [1112,469], [1149,468]]),
    np.array([[1046,531], [1105,533], [1160,534], [1200,535]]),
]

MAE_before_list_2 = []
MAE_after_list_2 = []
for i, cyl_contour in enumerate(cyl_contours_2):
    world_points = cylindrical_cam.project_2d_to_3d(cyl_contour, norm=np.ones(cyl_contour.shape[0]))
    fisheye_contour = fisheye_cam.project_3d_to_2d(world_points)

    cv2.polylines(cylindrical_image, [cyl_contour.astype(int)], False, (58, 56, 216), 2)
    cv2.polylines(fisheye_image, [fisheye_contour.astype(int)], False, (102, 210, 243), 2)

    pre_error, _ = calculate_straightness(fisheye_contour)
    post_error, _ = calculate_straightness(cyl_contour)
    MAE_before_list_2.append(pre_error)
    MAE_after_list_2.append(post_error)

    padding = 25
    x1_cyl = max(0, np.min(cyl_contour[:, 0]) - padding)
    y1_cyl = max(0, np.min(cyl_contour[:, 1]) - padding)
    x2_cyl = min(cylindrical_image.shape[1], np.max(cyl_contour[:, 0]) + padding)
    y2_cyl = min(cylindrical_image.shape[0], np.max(cyl_contour[:, 1]) + padding)
    cropped_cyl = cylindrical_image[y1_cyl:y2_cyl, x1_cyl:x2_cyl]

    x1_fish = max(0, np.min(fisheye_contour[:, 0]) - padding)
    y1_fish = max(0, np.min(fisheye_contour[:, 1]) - padding)
    x2_fish = min(fisheye_image.shape[1], np.max(fisheye_contour[:, 0]) + padding)
    y2_fish = min(fisheye_image.shape[0], np.max(fisheye_contour[:, 1]) + padding)
    cropped_fish = fisheye_image[int(y1_fish):int(y2_fish), int(x1_fish):int(x2_fish)]

    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(cropped_fish, cv2.COLOR_BGR2RGB))
    plt.title(f'Fisheye (cropped) \n MAE:{pre_error:.2f}')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(cropped_cyl, cv2.COLOR_BGR2RGB))
    plt.title(f'Distortion correction (cropped) \n MAE:{post_error:.2f}')
    plt.axis('off')
    plt.savefig(f'Results/Fig5g_{i+1}.pdf')

plt.figure(figsize=(4, 6))
sns.boxplot(data=[MAE_before_list_2, MAE_after_list_2], width=0.3, palette=['gray', (216/255, 56/255, 58/255)])
plt.xticks(ticks=[0, 1], labels=['Before', 'After'], fontsize=10)
for i, data in enumerate([MAE_before_list_2, MAE_after_list_2]):
    base_x = i + 0.3
    x_jitter = np.random.normal(base_x, 0.03, size=len(data))
    plt.scatter(x_jitter, data, color='gray', alpha=0.5, s=18, edgecolor=None, linewidth=0.5, zorder=5)
plt.xlim(-0.5, 1.5)
plt.yticks(np.linspace(0, 20, 5))
plt.ylim(-1, 21)
plt.ylabel('MAE')
plt.tight_layout()
plt.savefig('Results/Fig5g.pdf')

plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2RGB))
plt.title('Fig.S20a. Fisheye')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(cylindrical_image, cv2.COLOR_BGR2RGB))
plt.title('Fig.S20b. Distortion correction')
plt.axis('off')
plt.tight_layout()
plt.savefig('Results/FigS20ab.pdf')
plt.show()

# heatmap
plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
COLORS = [(r/255, g/255, b/255) for (r, g, b) in [(216, 56, 58), (253, 253, 253)]]
cmap = LinearSegmentedColormap.from_list('custom', COLORS)

ax0 = plt.subplot(gs[0])
im1 = ax0.imshow(np.array(MAE_before_list_2).reshape(6, 6), cmap=cmap, vmin=0, vmax=5)
ax0.set_title('Fig.5e')
ax0.set_xticks([])
ax0.set_yticks([])
for (i, j), val in np.ndenumerate(np.array(MAE_before_list_2).reshape(6, 6)):
    ax0.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val < 2.5 else 'black',
             fontsize=12, fontweight='bold')

ax1 = plt.subplot(gs[1])
im2 = ax1.imshow(np.array(MAE_after_list_2).reshape(6, 6), cmap=cmap, vmin=0, vmax=5)
ax1.set_title('Fig.5e')
ax1.set_xticks([])
ax1.set_yticks([])
for (i, j), val in np.ndenumerate(np.array(MAE_after_list_2).reshape(6, 6)):
    ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val < 2.5 else 'black',
             fontsize=12, fontweight='bold')

cbar_ax = plt.subplot(gs[2])
plt.colorbar(im1, cax=cbar_ax)
plt.tight_layout()
plt.savefig('Results/Fig5e.pdf')
plt.show()

# visualize scale values and matrice
scale_data = np.loadtxt('scale_values.txt')
assert scale_data.size == cylindrical_cam.width * cylindrical_cam.height
# reshape as data is stored in lines (1280 lines, each line 966 values)
scale_matrix = scale_data.reshape(cylindrical_cam.width, cylindrical_cam.height)
plt.figure()
plt.imshow(scale_matrix.T, cmap='viridis', aspect='auto')
plt.colorbar()  # represents scale values here, not matrix
plt.title('Fig.S17a. Scale values of the k-matrix')
plt.axis('off')
plt.savefig('Results/FigS17a.pdf')
plt.show()


'''Edge detection'''
# Sobel
sobelx = cv2.Sobel(fisheye_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(fisheye_gray, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
fisheye_edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
fisheye_edge_copy = fisheye_edge.copy()

sobelx = cv2.Sobel(cropped_copy, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(cropped_copy, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
final = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Fig.5d
cyl_contours = [
    np.array([[316, 428], [318, 350], [319, 281], [320, 212]]),
    np.array([[785, 363], [783, 309], [783, 255], [783, 189]]),
    np.array([[980, 249], [933, 208], [878, 167], [825, 130]]),
    np.array([[1052, 461], [1052, 389], [1052, 316], [1047, 240]]),
    np.array([[538, 366], [507, 370], [486, 374], [457, 379]])
]

fisheye_contours = []
for cyl_contour in cyl_contours:
    world_points = cylindrical_cam.project_2d_to_3d(cyl_contour, norm=np.ones(cyl_contour.shape[0]))
    fisheye_contour = fisheye_cam.project_3d_to_2d(world_points)
    fisheye_contours.append(fisheye_contour)

# contrast
mean_contrast_origin = []
mean_contrast_edge = []
cropped_ori_list = []
cropped_edge_list = []
mean_contrast_ori_list = []
mean_contrast_edge_list = []
padding = 25
edge_half_width = 1  # ±1 pixel at edge for averaging
side_width = 5       # 5 pixel at each side for averaging

for i, fisheye_contour in enumerate(fisheye_contours):
    x1 = int(max(0, np.min(fisheye_contour[:, 0]) - padding))
    y1 = int(max(0, np.min(fisheye_contour[:, 1]) - padding))
    x2 = int(min(fisheye_gray.shape[1], np.max(fisheye_contour[:, 0]) + padding))
    y2 = int(min(fisheye_gray.shape[0], np.max(fisheye_contour[:, 1]) + padding))

    cropped_ori = fisheye_image_copy[y1:y2, x1:x2]
    cropped_edge = fisheye_edge[y1:y2, x1:x2]
    cropped_ori_list.append(cropped_ori)
    cropped_edge_list.append(cropped_edge)

    local_contour = fisheye_contour - np.array([x1, y1])

    mean_contrast_ori = mean_contrast_along_polyline(cv2.cvtColor(cropped_ori, cv2.COLOR_BGR2GRAY), local_contour, edge_half_width, side_width)
    mean_contrast_edg = mean_contrast_along_polyline(cropped_edge, local_contour, edge_half_width, side_width)
    mean_contrast_origin.append(mean_contrast_ori)
    mean_contrast_edge.append(mean_contrast_edg)
    mean_contrast_ori_list.append(mean_contrast_ori)
    mean_contrast_edge_list.append(mean_contrast_edg)

fig_list = []
for i in range(len(fisheye_contours)):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig_list.append(fig)

    x_offset = int(max(0, np.min(fisheye_contours[i][:, 0]) - padding))
    y_offset = int(max(0, np.min(fisheye_contours[i][:, 1]) - padding))
    contour_draw = fisheye_contours[i].copy()
    contour_draw[:, 0] -= x_offset
    contour_draw[:, 1] -= y_offset
    contour_draw = np.round(contour_draw).astype(np.int32)

    ori = cropped_ori_list[i]
    if ori.ndim == 2:
        ori_vis = cv2.cvtColor(ori, cv2.COLOR_GRAY2BGR)
    else:
        ori_vis = ori.copy()
    cv2.polylines(ori_vis, [contour_draw], isClosed=False, color=(102, 210, 243), thickness=2)
    axes[0].imshow(cv2.cvtColor(ori_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Before Edge\nMean Contrast: {mean_contrast_ori_list[i]:.2f}')
    axes[0].axis('off')

    edge = cropped_edge_list[i]
    if edge.ndim == 2:
        edge_vis = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    else:
        edge_vis = edge.copy()
    cv2.polylines(edge_vis, [contour_draw], isClosed=False, color=(193, 127, 47), thickness=2)
    axes[1].imshow(cv2.cvtColor(edge_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'After Edge\nMean Contrast: {mean_contrast_edge_list[i]:.2f}')
    axes[1].axis('off')

plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(fisheye_image_copy, cv2.COLOR_BGR2RGB))
plt.title('Fig.5b. Fisheye')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(fisheye_edge, cv2.COLOR_BGR2RGB))
plt.title('Edge detection')
plt.axis('off')
plt.tight_layout()
plt.savefig('Results/Fig5b_2.pdf')
plt.show()

# Fig.5f
cyl_contours_2 = [
    np.array([[199, 419], [199, 381], [202, 361], [202, 328]]),
    np.array([[320, 211], [320, 232], [320, 254], [320, 281]]),
    np.array([[289, 384], [308, 387], [319, 389], [337, 390]]),
    np.array([[465, 390], [465, 405], [465, 421], [464, 436]]),
    np.array([[480, 307], [502, 328], [518, 343], [528, 354]]),
    np.array([[554, 390], [554, 421], [554, 439], [553, 471]]),
    np.array([[668, 412], [668, 428], [667, 443], [667, 461]]),
    np.array([[682, 366], [683, 397], [683, 427], [683, 456]]),
    np.array([[812, 174], [814, 215], [815, 296], [817, 359]]),
    np.array([[829, 196], [828, 255], [829, 320], [830, 402]]),
    np.array([[872, 218], [872, 276], [872, 332], [873, 409]]),
    np.array([[506, 362], [522, 361], [544, 358], [561, 376]]),
    np.array([[938, 307], [939, 333], [939, 366], [938, 409]]),
    np.array([[1006, 116], [1009, 153], [1010, 190], [1011, 244]]),
    np.array([[389, 472], [389, 491], [390, 508], [390, 522]]),
    np.array([[454, 438], [469, 447], [489, 463], [499, 473]]),
    np.array([[1046, 240], [1050, 273], [1051, 296], [1051, 321]]),
    np.array([[1148, 398], [1149, 425], [1149, 456], [1151, 496]]),
    np.array([[373, 214], [388, 225], [404, 239], [418, 248]]),
    np.array([[245, 295], [286, 304], [323, 316], [355, 329]]),
    np.array([[370, 335], [385, 340], [401, 346], [417, 353]]),
    np.array([[155, 483], [180, 483], [201, 484], [230, 484]]),
    np.array([[199, 577], [236, 574], [276, 570], [318, 564]]),
    np.array([[392, 507], [421, 502], [450, 501], [473, 499]]),
    np.array([[378, 537], [407, 531], [443, 526], [473, 522]]),
    np.array([[472, 388], [491, 385], [517, 382], [544, 379]]),
    np.array([[729, 414], [742, 403], [762, 388], [781, 371]]),
    np.array([[834, 174], [891, 210], [947, 250], [992, 287]]),
    np.array([[833, 349], [843, 352], [859, 356], [870, 359]]),
    np.array([[829, 430], [888, 436], [935, 442], [964, 445]]),
    np.array([[570, 498], [617, 498], [679, 500], [740, 500]]),
    np.array([[445, 548], [518, 552], [586, 557], [641, 556]]),
    np.array([[671, 555], [730, 552], [811, 545], [880, 538]]),
    np.array([[1054, 280], [1061, 278], [1075, 276], [1087, 273]]),
    np.array([[1056, 384], [1068, 381], [1082, 380], [1091, 380]]),
    np.array([[1115, 588], [1141, 589], [1171, 589], [1195, 589]]),
]

fisheye_contours_2 = []
for cyl_contour in cyl_contours_2:
    world_points = cylindrical_cam.project_2d_to_3d(cyl_contour, norm=np.ones(cyl_contour.shape[0]))
    fisheye_contour = fisheye_cam.project_3d_to_2d(world_points)
    fisheye_contours_2.append(fisheye_contour)

# contrast
mean_contrast_origin_2 = []
mean_contrast_edge_2 = []
cropped_ori_list_2 = []
cropped_edge_list_2 = []
mean_contrast_ori_list_2 = []
mean_contrast_edge_list_2 = []
padding = 25
edge_half_width = 1  # ±1 pixel at edge for averaging
side_width = 5       # 5 pixel at each side for averaging

for i, fisheye_contour in enumerate(fisheye_contours_2):
    x1 = int(max(0, np.min(fisheye_contour[:, 0]) - padding))
    y1 = int(max(0, np.min(fisheye_contour[:, 1]) - padding))
    x2 = int(min(fisheye_gray.shape[1], np.max(fisheye_contour[:, 0]) + padding))
    y2 = int(min(fisheye_gray.shape[0], np.max(fisheye_contour[:, 1]) + padding))

    cropped_ori = fisheye_image_copy[y1:y2, x1:x2]
    cropped_edge = fisheye_edge[y1:y2, x1:x2]
    cropped_ori_list_2.append(cropped_ori)
    cropped_edge_list_2.append(cropped_edge)

    local_contour = fisheye_contour - np.array([x1, y1])

    mean_contrast_ori = mean_contrast_along_polyline(cv2.cvtColor(cropped_ori, cv2.COLOR_BGR2GRAY), local_contour, edge_half_width, side_width)
    mean_contrast_edg = mean_contrast_along_polyline(cropped_edge, local_contour, edge_half_width, side_width)
    mean_contrast_origin_2.append(mean_contrast_ori)
    mean_contrast_edge_2.append(mean_contrast_edg)
    mean_contrast_ori_list_2.append(mean_contrast_ori)
    mean_contrast_edge_list_2.append(mean_contrast_edg)

fig_list_2 = []
for i in range(len(fisheye_contours_2)):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig_list_2.append(fig)

    x_offset = int(max(0, np.min(fisheye_contours_2[i][:, 0]) - padding))
    y_offset = int(max(0, np.min(fisheye_contours_2[i][:, 1]) - padding))
    contour_draw = fisheye_contours_2[i].copy()
    contour_draw[:, 0] -= x_offset
    contour_draw[:, 1] -= y_offset
    contour_draw = np.round(contour_draw).astype(np.int32)

    ori = cropped_ori_list_2[i]
    if ori.ndim == 2:
        ori_vis = cv2.cvtColor(ori, cv2.COLOR_GRAY2BGR)
    else:
        ori_vis = ori.copy()
    cv2.polylines(ori_vis, [contour_draw], isClosed=False, color=(102, 210, 243), thickness=2)
    axes[0].imshow(cv2.cvtColor(ori_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Before Edge\nMean Contrast: {mean_contrast_ori_list_2[i]:.2f}')
    axes[0].axis('off')

    edge = cropped_edge_list_2[i]
    if edge.ndim == 2:
        edge_vis = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    else:
        edge_vis = edge.copy()
    cv2.polylines(edge_vis, [contour_draw], isClosed=False, color=(193, 127, 47), thickness=2)
    axes[1].imshow(cv2.cvtColor(edge_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'After Edge\nMean Contrast: {mean_contrast_edge_list_2[i]:.2f}')
    axes[1].axis('off')
    plt.savefig(f'Results/Fig5e_{i+1}.pdf')

plt.figure(figsize=(8, 6))
plt.subplot(121)
fisheye_with_lines = fisheye_image_copy.copy()
for fisheye_contour in fisheye_contours_2:
    contour_int = fisheye_contour.astype(np.int32)
    cv2.polylines(fisheye_with_lines, [contour_int], isClosed=False, color=(102, 210, 243), thickness=2)
plt.imshow(cv2.cvtColor(fisheye_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Fig.S20c. Fisheye')
plt.axis('off')

plt.subplot(122)
for fisheye_contour in fisheye_contours_2:
    contour_int = fisheye_contour.astype(np.int32)
    cv2.polylines(fisheye_edge, [contour_int], isClosed=False, color=(193, 127, 47), thickness=2)
plt.imshow(cv2.cvtColor(fisheye_edge, cv2.COLOR_BGR2RGB))
plt.title('Fig.S20d. Edge detection')
plt.axis('off')
plt.tight_layout()
plt.savefig('Results/FigS20cd.pdf')
plt.show()

plt.figure(figsize=(4, 6))
sns.boxplot(data=[mean_contrast_origin_2, mean_contrast_edge_2], width=0.3, palette=['gray', (47/255, 127/255, 193/255)])
plt.xticks(ticks=[0, 1], labels=['Before', 'After'], fontsize=10)
for i, data in enumerate([mean_contrast_origin_2, mean_contrast_edge_2]):
    base_x = i + 0.3
    x_jitter = np.random.normal(base_x, 0.03, size=len(data))
    plt.scatter(x_jitter, data, color='gray', alpha=0.5, s=18, edgecolor=None, linewidth=0.5, zorder=5)
plt.xlim(-0.5, 1.5)
plt.title('Fig.5h')
plt.ylabel('Mean Contrast')
plt.tight_layout()
plt.savefig('Results/Fig5h.pdf')
plt.show()

# heatmap
plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
COLORS = [(r/255, g/255, b/255) for (r, g, b) in [(253, 253, 253), (47, 127, 193)]]
cmap = LinearSegmentedColormap.from_list('custom', COLORS)

ax0 = plt.subplot(gs[0])
im1 = ax0.imshow(np.array(mean_contrast_origin_2).reshape(6, 6), cmap=cmap, vmin=0, vmax=120)
ax0.set_title('Fig.5f')
ax0.set_xticks([])
ax0.set_yticks([])
for (i, j), val in np.ndenumerate(np.array(mean_contrast_origin_2).reshape(6, 6)):
    ax0.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val < 2.5 else 'black',
             fontsize=12, fontweight='bold')

ax1 = plt.subplot(gs[1])
im2 = ax1.imshow(np.array(mean_contrast_edge_2).reshape(6, 6), cmap=cmap, vmin=0, vmax=120)
ax1.set_title('Fig.5f')
ax1.set_xticks([])
ax1.set_yticks([])
for (i, j), val in np.ndenumerate(np.array(mean_contrast_edge_2).reshape(6, 6)):
    ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val < 2.5 else 'black',
             fontsize=12, fontweight='bold')

cbar_ax = plt.subplot(gs[2])
plt.colorbar(im1, cax=cbar_ax)
plt.tight_layout()
plt.savefig('Results/Fig5f.pdf')
plt.show()

'''Quantify multiplication-accumulation operations to perform distortion correction and edge detection'''
resolutions = {
    "480p": 854 * 480,
    "720p": 1280 * 720,
    "1080p": 1920 * 1080,
    "2K": 2560 * 1440,
    "4k": 3840 * 2160
}

distortion_ops = [4 * v / 1e6 for v in resolutions.values()]
edge_ops = [22 * v / 1e6 for v in resolutions.values()]

labels = list(resolutions.keys())
y_pos = np.arange(len(labels))
plt.figure(figsize=(12, 5))
plt.barh(y_pos, distortion_ops, height=0.4, color=(216/255, 56/255, 58/255), alpha=0.5, label='Distortion correction')
plt.barh(y_pos, edge_ops, height=0.4, left=distortion_ops, color=(47/255, 127/255, 193/255), alpha=0.5, label='Edge detection')
plt.yticks(y_pos, labels, fontsize=10)
plt.xlabel('MAC Operations (M ops)', fontsize=10)
plt.ylabel('Image Resolution', fontsize=10)
plt.title('Fig.5i. Multiplication-accumulation Operations to perform distortion correction and edge detection', fontsize=10)
plt.legend(loc='lower right', fontsize=10)
plt.xlim(0, max(edge_ops) + max(distortion_ops) + 20)
plt.tight_layout()
plt.savefig('Results/Fig5i.pdf')
plt.show()


'''Object recognition'''
# Front
fisheye_cam_FV = read_cam_from_json('00168_FV.json')
original_ratio_FV = fisheye_cam_FV.size[0] / fisheye_cam_FV.size[1]  # width/length
cylindrical_cam_FV = make_cylindrical_cam(fisheye_cam_FV)
fisheye_image_FV = cv2.imread('rgb_00168_FV.png')
fisheye_image_FV_copy = fisheye_image_FV.copy()
sobelx = cv2.Sobel(fisheye_image_FV_copy, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(fisheye_image_FV_copy, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
edge_FV = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
map1_FV, map2_FV = create_img_projection_maps(fisheye_cam_FV, cylindrical_cam_FV)
cylindrical_image_FV = cv2.remap(fisheye_image_FV, map1_FV, map2_FV, cv2.INTER_CUBIC)
cyl_height, cyl_width = cylindrical_image_FV.shape[:2]
pt1 = (197, 140)  # top left of cropped figure
pt2 = (1090, 140)  # top right of cropped figure
cropped_FV = cylindrical_image_FV[pt1[1]:(pt1[1] + int((pt2[0] - pt1[0]) / original_ratio_FV)), pt1[0]:pt2[0]]
yolo_object_recognition(cropped_FV, filename='Results/FigS22a.pdf')
sobelx = cv2.Sobel(cropped_FV, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(cropped_FV, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
final_FV = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Left
fisheye_cam_MVL = read_cam_from_json('00169_MVL.json')
original_ratio_MVL = fisheye_cam_MVL.size[0] / fisheye_cam_MVL.size[1]  # width/length
cylindrical_cam_MVL = make_cylindrical_cam(fisheye_cam_MVL)
fisheye_image_MVL = cv2.imread('rgb_00169_MVL.png')
fisheye_image_MVL_copy = fisheye_image_MVL.copy()
sobelx = cv2.Sobel(fisheye_image_MVL_copy, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(fisheye_image_MVL_copy, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
edge_MVL = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
map1_MVL, map2_MVL = create_img_projection_maps(fisheye_cam_MVL, cylindrical_cam_MVL)
cylindrical_image_MVL = cv2.remap(fisheye_image_MVL, map1_MVL, map2_MVL, cv2.INTER_CUBIC)
cyl_height, cyl_width = cylindrical_image_MVL.shape[:2]
pt1 = (197, 471)  # top left of cropped figure
pt2 = (1090, 471)  # top right of cropped figure
cropped_MVL = cylindrical_image_MVL[pt1[1]:(pt1[1] + int((pt2[0] - pt1[0]) / original_ratio_MVL)), pt1[0]:pt2[0]]
yolo_object_recognition(cropped_MVL, filename='Results/FigS22b.pdf')
sobelx = cv2.Sobel(cropped_MVL, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(cropped_MVL, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
final_MVL = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Rear
fisheye_cam_RV = read_cam_from_json('00171_RV.json')
original_ratio_RV = fisheye_cam_RV.size[0] / fisheye_cam_RV.size[1]  # width/length
cylindrical_cam_RV = make_cylindrical_cam(fisheye_cam_RV)
fisheye_image_RV = cv2.imread('rgb_00171_RV.png')
fisheye_image_RV_copy = fisheye_image_RV.copy()
sobelx = cv2.Sobel(fisheye_image_RV_copy, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(fisheye_image_RV_copy, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
edge_RV = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
map1_RV, map2_RV = create_img_projection_maps(fisheye_cam_RV, cylindrical_cam_RV)
cylindrical_image_RV = cv2.remap(fisheye_image_RV, map1_RV, map2_RV, cv2.INTER_CUBIC)
cyl_height, cyl_width = cylindrical_image_RV.shape[:2]
pt1 = (197, 210)  # top left of cropped figure
pt2 = (1090, 210)  # top right of cropped figure
cropped_RV = cylindrical_image_RV[pt1[1]:(pt1[1] + int((pt2[0] - pt1[0]) / original_ratio_RV)), pt1[0]:pt2[0]]
yolo_object_recognition(cropped_RV, filename='Results/FigS22c.pdf')
sobelx = cv2.Sobel(cropped_RV, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(cropped_RV, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
final_RV = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Right
fisheye_cam_MVR = read_cam_from_json('00170_MVR.json')
original_ratio_MVR = fisheye_cam_MVR.size[0] / fisheye_cam_MVR.size[1]  # width/length
cylindrical_cam_MVR = make_cylindrical_cam(fisheye_cam_MVR)
fisheye_image_MVR = cv2.imread('rgb_00170_MVR.png')
fisheye_image_MVR_copy = fisheye_image_MVR.copy()
sobelx = cv2.Sobel(fisheye_image_MVR_copy, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(fisheye_image_MVR_copy, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
edge_MVR = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
map1_MVR, map2_MVR = create_img_projection_maps(fisheye_cam_MVR, cylindrical_cam_MVR)
cylindrical_image_MVR = cv2.remap(fisheye_image_MVR, map1_MVR, map2_MVR, cv2.INTER_CUBIC)
cyl_height, cyl_width = cylindrical_image_MVR.shape[:2]
pt1 = (197, 471)  # top left of cropped figure
pt2 = (1090, 471)  # top right of cropped figure
cropped_MVR = cylindrical_image_MVR[pt1[1]:(pt1[1] + int((pt2[0] - pt1[0]) / original_ratio_MVR)), pt1[0]:pt2[0]]
yolo_object_recognition(cropped_MVR, filename='Results/FigS22d.pdf')
sobelx = cv2.Sobel(cropped_MVR, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(cropped_MVR, cv2.CV_64F, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)
final_MVR = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

plt.figure(figsize=(8, 8))
plt.subplot(4,4,1)
plt.imshow(cv2.cvtColor(fisheye_image_FV, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,5)
plt.imshow(cv2.cvtColor(cropped_FV, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,9)
plt.imshow(edge_FV)
plt.axis('off')
plt.subplot(4,4,13)
plt.imshow(final_FV)
plt.axis('off')

plt.subplot(4,4,2)
plt.imshow(cv2.cvtColor(fisheye_image_MVL, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,6)
plt.imshow(cv2.cvtColor(cropped_MVL, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,10)
plt.imshow(edge_MVL)
plt.axis('off')
plt.subplot(4,4,14)
plt.imshow(final_MVL)
plt.axis('off')

plt.subplot(4,4,3)
plt.imshow(cv2.cvtColor(fisheye_image_RV, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,7)
plt.imshow(cv2.cvtColor(cropped_RV, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,11)
plt.imshow(edge_RV)
plt.axis('off')
plt.subplot(4,4,15)
plt.imshow(final_RV)
plt.axis('off')

plt.subplot(4,4,4)
plt.imshow(cv2.cvtColor(fisheye_image_MVR, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,8)
plt.imshow(cv2.cvtColor(cropped_MVR, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(4,4,12)
plt.imshow(edge_MVR)
plt.axis('off')
plt.subplot(4,4,16)
plt.imshow(final_MVR)
plt.axis('off')

plt.tight_layout()
plt.savefig('Results/FigS21.pdf')
plt.show()


'''Overall performance'''
plt.figure(figsize=(12, 8))  # 调整比例以适应三列平均布局
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])

ax1 = plt.subplot(gs[0:2, 0])
ax1.imshow(cv2.cvtColor(fisheye_image_copy, cv2.COLOR_BGR2RGB))
ax1.set_title('Fisheye')
ax1.axis('off')

ax2 = plt.subplot(gs[0, 1])
ax2.imshow(cv2.cvtColor(cylindrical_image_copy, cv2.COLOR_BGR2RGB))
ax2.set_title('Distortion correction')
ax2.axis('off')

ax3 = plt.subplot(gs[1, 1])
ax3.imshow(cv2.cvtColor(fisheye_edge_copy, cv2.COLOR_BGR2RGB))
ax3.set_title('Edge detection')
ax3.axis('off')

ax4 = plt.subplot(gs[0:2, 2])
ax4.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
ax4.set_title('Combining')
ax4.axis('off')

plt.tight_layout()
plt.savefig('Results/FigS18c.pdf')
plt.show()