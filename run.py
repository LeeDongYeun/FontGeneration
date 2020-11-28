import cv2
import numpy as np
import sys
from pathlib import Path

adjust_size = 500
rho_threshold = 10
theta_threshold = 30*np.pi/180
group_threshold = 15*np.pi/180
missing_threshold = 1.2

def angle_diff(theta1, theta2):
    return min(abs(theta1 - theta2), abs(theta1+np.pi - theta2), abs(theta1-np.pi - theta2))

def dist_diff(rho1, rho2):
    return abs(abs(rho1) - abs(rho2))

# 이미지 읽어오기
file_path = sys.argv[1]
orig_img = cv2.imread(file_path)

# 이미지 Resize
height, width, _ = orig_img.shape
img = cv2.resize(orig_img, dsize=(adjust_size, adjust_size), interpolation=cv2.INTER_AREA)

# 이미지 흑백 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지 프로세싱
edges = cv2.Canny(gray, 45, 150, apertureSize=3)

kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

kernel = np.ones((5, 5), np.uint8)
edges = cv2.erode(edges, kernel, iterations=1)

# 직선 추출
lines = cv2.HoughLines(edges, 1, 0.5*np.pi/180, 150)

if not lines.any():
    print('No lines were found')
    exit()

# 유사한 직선들 하나로 통합
filtered_lines = []

# how many lines are similar to a given one
similar_lines = {i : [] for i in range(len(lines))}
for i in range(len(lines)):
    for j in range(len(lines)):
        if i == j:
            continue
        rho_i, theta_i = lines[i][0]
        rho_j, theta_j = lines[j][0]
        if dist_diff(rho_i, rho_j) < rho_threshold and angle_diff(theta_i, theta_j) < theta_threshold:
            similar_lines[i].append(j)

# ordering the INDECES of the lines by how many are similar to them
indices = [i for i in range(len(lines))]
indices.sort(key=lambda x : len(similar_lines[x]))

# line flags is the base for the filtering
line_flags = len(lines)*[True]
for i in range(len(lines) - 1):
    if not line_flags[indices[i]]:
        continue

    for j in range(i + 1, len(lines)):
        if not line_flags[indices[j]]:
            continue

        rho_i,theta_i = lines[indices[i]][0]
        rho_j,theta_j = lines[indices[j]][0]
        if dist_diff(rho_i, rho_j) < rho_threshold and angle_diff(theta_i, theta_j) < theta_threshold:
            line_flags[indices[j]] = False

# filtering
for i in range(len(lines)):
    if line_flags[i]:
        filtered_lines.append(lines[i])

# 가로/세로 직선 그룹핑
similar_lines = {i : set() for i in range(len(filtered_lines))}
for i in range(len(filtered_lines)):
    for j in range(len(filtered_lines)):
        if i == j:
            continue
        rho_i, theta_i = filtered_lines[i][0]
        rho_j, theta_j = filtered_lines[j][0]
        if angle_diff(theta_i, theta_j) < group_threshold:
            similar_lines[i].add(j)
groupped_lines = set()
groups = []
for i, line in enumerate(filtered_lines):
    if i in groupped_lines:
        group_idx = [j for j, g in enumerate(groups) if i in g][0]
    else:
        groups.append(set())
        group_idx = len(groups) - 1
    groupped_lines.add(i)
    groups[group_idx] |= similar_lines[i]
    groups[group_idx].add(i)
    groupped_lines |= groups[group_idx]
groups.sort(key=len)
height_groups, width_groups = groups[-2:]

width_lines = [filtered_lines[i][0] for i in width_groups]
height_lines = [filtered_lines[i][0] for i in height_groups]
if np.abs(width_lines[0][1]) < np.abs(height_lines[0][1]):
    width_lines, height_lines = height_lines, width_lines

# 직선 거리 순 정렬 및 거리차 계산
width_lines = sorted(width_lines, key=lambda l: abs(l[0]))
height_lines = sorted(height_lines, key=lambda l: abs(l[0]))

width_adj_dists = [dist_diff(width_lines[i][0], width_lines[i-1][0]) for i in range(1, len(width_lines))]
height_adj_dists = [dist_diff(height_lines[i][0], height_lines[i-1][0]) for i in range(1, len(height_lines))]

width_adj_dists = sorted(width_adj_dists)
height_adj_dists = sorted(height_adj_dists)

line_gap = width_adj_dists[len(width_adj_dists)//6]
_width_adj_dists_filtered = [e for e in width_adj_dists if e > 1.5*line_gap]
line_height = _width_adj_dists_filtered[len(_width_adj_dists_filtered)//6]
letter_width = height_adj_dists[len(height_lines)//2]

# 빠진 세로줄 찾기
missing_height_lines = []
for i in range(1, len(height_lines)):
    dist = dist_diff(height_lines[i][0], height_lines[i-1][0])
    angle_dist = angle_diff(height_lines[i][1], height_lines[i-1][1])
    if letter_width * (1-missing_threshold) <= dist <= letter_width * missing_threshold:
        continue
    missing_count = np.round(dist/letter_width)-1
    for j in range(1, 1+int(missing_count)):
        rho = height_lines[i-1][0] + dist*j/(1+missing_count)
        theta = height_lines[i-1][1] + angle_dist*j/(1+missing_count)
        missing_height_lines.append(np.array([rho, theta]))
height_lines += missing_height_lines
height_lines = sorted(height_lines, key=lambda l: abs(l[0]))

# 빠진 가로줄 찾기
missing_width_lines = []
is_letter_area = True
removal_idx = []
i = 0
while i < len(width_lines)-1:
    i += 1
    dist = dist_diff(width_lines[i][0], width_lines[i-1][0])
    angle_dist = angle_diff(width_lines[i][1], width_lines[i-1][1])
    expected_line_gap = np.abs(line_height - dist) > np.abs(line_gap - dist)
    dist_combination = line_height + line_gap
    is_combination = np.abs(line_height - dist) >  np.abs(dist_combination - dist)
    if i == 1 and expected_line_gap:
        removal_idx.append(0)
        continue
    if is_letter_area:
        if expected_line_gap:
            removal_idx.append(i-1)
        elif is_combination:
            rho = width_lines[i-1][0] + line_height
            theta = (width_lines[i-1][1] + width_lines[i][1])/2
            missing_width_lines.append(width_lines[i-1])
            width_lines[i-1] = np.array([rho, theta])
            is_letter_area = False
            i -= 1
        else:
            is_letter_area = False
    else:
        if is_combination:
            rho = width_lines[i-1][0] + line_gap
            theta = (width_lines[i-1][1] + width_lines[i][1])/2
            missing_width_lines.append(width_lines[i-1])
            width_lines[i-1] = np.array([rho, theta])
            i -= 1
        elif not expected_line_gap:
            missing_width_lines.append(width_lines[i-1])
            i -= 1
        is_letter_area = True
width_lines = [l for i, l in enumerate(width_lines) if i not in removal_idx]
width_lines += missing_width_lines
width_lines = sorted(width_lines, key=lambda l: abs(l[0]))

# Cartesian으로 변환
cartesian_width_lines = []
for line in width_lines:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = (x0-b) * width / adjust_size
    x2 = (x0+b) * width / adjust_size
    y1 = (y0+a) * height / adjust_size
    y2 = (y0-a) * height / adjust_size
    cartesian_width_lines.append(np.array([[x1, y1], [x2, y2]]))

cartesian_height_lines = []
for line in height_lines:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = (x0-b) * width / adjust_size
    x2 = (x0+b) * width / adjust_size
    y1 = (y0+a) * height / adjust_size
    y2 = (y0-a) * height / adjust_size
    cartesian_height_lines.append(np.array([[x1, y1], [x2, y2]]))

# 박스 구하기
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def perp(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1, a2, b1, b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1

boxes = []

for i_width in range(0, len(cartesian_width_lines)-1, 2):
    for i_height in range(len(cartesian_height_lines)-1):
        a1, a2 = cartesian_width_lines[i_width]
        b1, b2 = cartesian_height_lines[i_height]
        p1 = seg_intersect(a1, a2, b1, b2)

        a1, a2 = cartesian_width_lines[i_width+1]
        b1, b2 = cartesian_height_lines[i_height]
        p2 = seg_intersect(a1, a2, b1, b2)

        a1, a2 = cartesian_width_lines[i_width]
        b1, b2 = cartesian_height_lines[i_height+1]
        p3 = seg_intersect(a1, a2, b1, b2)

        a1, a2 = cartesian_width_lines[i_width+1]
        b1, b2 = cartesian_height_lines[i_height+1]
        p4 = seg_intersect(a1, a2, b1, b2)

        rect = order_points(np.array([p1, p2, p3, p4]))
        tl, _, br, _ = rect
        boxes.append((tl, br))
print(boxes)
'''
        boxes.append(four_point_transform(orig_img, np.array([p1, p2, p3, p4])))

# 박스 이미지 프로세싱 및 저장
Path("output").mkdir(exist_ok=True)
box_num = 1
for box in boxes:
    box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    _, box = cv2.threshold(box, 200, 255, cv2.THRESH_BINARY)
    h, w = box.shape
    box = box[h//16:-h//16, w//10:-w//10]
    h, w = box.shape
    inner_box = box[h//4:-h//4, w//4:-w//4]
    if np.sum(inner_box < 200) > 0:
        n = str(box_num).zfill(1+np.log10(len(boxes)).astype(int))
        cv2.imwrite(f'output/box-{n}.jpg', box)
        box_num += 1

# 원본 사진에 직선 그리기 (Optional)
def draw(line, color):
    rho,theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int((x0 + 1000*(-b)) * width / adjust_size)
    y1 = int((y0 + 1000*(a)) * height / adjust_size)
    x2 = int((x0 - 1000*(-b)) * width / adjust_size)
    y2 = int((y0 - 1000*(a)) * height / adjust_size)
    cv2.line(orig_img, (x1,y1) , (x2,y2), color, 2)

for line in width_lines:
    draw(line, (0, 0, 150))
for line in height_lines:
    draw(line, (150, 0, 0))

cv2.imwrite('converted.jpg', orig_img)
'''
