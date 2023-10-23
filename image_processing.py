import pandas as pd
import numpy as np
import skimage
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import statistics

def perp(a):
    """ Makes a perpindicular vector """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2):
    """ Calculates the intersection point between two line segments """
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def convert_rad_to_degrees(angle):
    """ Converts an angle in radians to degrees """
    return (angle * 180.0) / math.pi

def get_regionprops(image):
    """ Extracts regionprops for an image. Returns a list. """
    
    label_img = skimage.measure.label(image)    
    regions = skimage.measure.regionprops(label_img)
    
    
    return regions

def process_rprops(rprops):
    data = []
    
    num_regions = len(rprops)
    avg_area = 0
    for props in rprops:
        avg_area += props.area / num_regions
    
    for props in rprops:
        if props.area < avg_area * 0.45:
            print(f"Small anomalous region; excluding. Area={props.area}, Average Area={avg_area}")
            continue
        screw_props = {}
        screw_props["cent_y"], screw_props["cent_x"] = props.centroid
        screw_props["orientation"] = props.orientation
        screw_props["perimeter"] = props.perimeter
        screw_props["area"] = props.area
        y0, x0 = props.centroid
        orientation = props.orientation
        screw_props["edge_x"] = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        screw_props["edge_y"] = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
        data.append(screw_props)

    data_sorted = sorted(data, key=lambda d: d['cent_y']) 
    
    return data_sorted

def process_keypoints(keypoints, orientations, verbose=False):
    """ Calculates intersections and angles. Returns point for plotting """
    lines = []
    angles = []
    results = []
    both_flag = None
    for i, _ in enumerate(keypoints):
        res = {}
        try:
            s1 = keypoints[i]
            s1_orient = orientations[i]
            s2 = keypoints[i+1]
            s2_orient = orientations[i+1]
           
        except IndexError as e:
            break
        
        if s1_orient == "both" or s2_orient == "both":
            both_flag = True
            pass
        if s1_orient != s2_orient and both_flag is None:   
            try:
                s2 = keypoints[i+2]
                s2_orient = orientations[i+2]                        
            except IndexError as e:
                continue        
        if s1_orient != s2_orient and both_flag is None:
            continue

        linecolor = get_linecolor(i)
        # Calculate Intersection Point
        p11 = np.array( [s1["edge_x"], s1["edge_y"]] )
        p12 = np.array( [s1["cent_x"], s1["cent_y"]] )
        p21 = np.array( [s2["edge_x"], s2["edge_y"]] )
        p22 = np.array( [s2["cent_x"], s2["cent_y"]] )
        intersection = seg_intersect(p11, p12, p21, p22)
        line1 = {
            "x0": s1["cent_x"],
            "x1": intersection[0],
            "y0": s1["cent_y"],
            "y1": intersection[1],
            "color": linecolor,
            "width": 0.2
        }
        line2 = {
            "x0": s2["cent_x"],
            "x1": intersection[0],
            "y0": s2["cent_y"],
            "y1": intersection[1],
            "color": linecolor,
            "width": 0.2            
        }
        lines.append(line1)
        lines.append(line2)
         
        # Calculate Vector Angle
        v1 = (intersection[0] - s1["cent_x"], intersection[1] - s1["cent_y"])
        v2 = (intersection[0] - s2["cent_x"], intersection[1] - s2["cent_y"])
        
        angle = angle_between(
            v1=v1,
            v2=v2 
        )
        angle_degrees = convert_rad_to_degrees(angle)
        if angle_degrees > 90:
            angle_degrees = 180 - angle_degrees
            
        angles.append(angle_degrees)
        
        if verbose:  
            print(f"Screw 1 orientation: {s1_orient}\tScrew 2 orientation: {s2_orient}")
            print(f"Angle Between Screw{i:2d} and Screw{i+1:2d}")
            print(f"Intersection point: (X={intersection[0]:6.1f},  Y={intersection[1]:7.1f})")
            print(f"Angle in degrees: {angle_degrees:4.1f}\n")
        
        res["s1_orient"] = s1_orient
        res["s2_orient"] = s2_orient
        res["angle"] = angle_degrees
        res["line1"] = line1
        res["line2"] = line2
        results.append(res)
    return results

def plot_tether_angles(base_img, masks=[], lines=[], points=[], save_path=None):
    """ Plots the base image, and any segments, lines, and points specified. """
    plt.style.use('dark_background')
    
    colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
    colors = [(0,0,1,c) for c in np.linspace(0,1,100)]
    cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
    fig, ax = plt.subplots()
    ax.axis('off')
    plt.xlim([0, 1024])
    plt.ylim([1024, 0])
    ax.imshow(
        base_img,
        cmap='bone'
    )   

    for mask in masks:
        ax.imshow(
            mask,
            cmap=cmapred,
            alpha=0.6
        )
    for line in lines:
        ax.plot(
            (line["x0"], line["x1"]),
            (line["y0"], line["y1"]),
            line["color"],
            line["width"],
        )
    for point in points:
        ax.plot(
            point["x"], point["y"],
            point["color"],
            point["size"]
        )
    
    if save_path:    
        plt.savefig(save_path)
    plt.close('all')
    return fig

def load_true_angles(path):
    df = pd.read_excel(
        io=path,
        header=1,           
    )
    return df

def calc_metrics(pred, true):
    pred_num = len(pred)
    true_num = len(true)
    
    if pred_num != true_num:
        print("The number of predicted angles does not equal the number of true angles. Check mask")
        return []
    
    diff = np.subtract(pred, true)
    
    results = {}
    results["diff_list"] = diff
    results["mean_abs_error"] = np.mean(np.absolute(diff))
    
    return results

def get_screw_orientations(mask, keypoints, plot=False):
    results = []
    
    ## Generate contours, reversing list as contours list bottom to top
    cnts, _ = cv2.findContours(mask.astype(np.uint8), 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE)
    cnts = list(reversed(cnts))
    
    keypoints_area = [x["area"] for x in keypoints]
    avg_area = statistics.fmean(keypoints_area)
    stdev_area = statistics.pstdev(keypoints_area)
    
    ## Detect tiny anomalous contours
    avg_length = 0
    num_cnts = len(cnts)
    for cnt in cnts:
        avg_length += len(cnt) / num_cnts
    for i, cnt in enumerate(cnts):
        if len(cnt) < avg_length * 0.2:
            del cnts[i]

    ## Set up plot
    if plot:
        canvas = np.zeros_like(mask)
        color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        _, ax = plt.subplots()

    ## Loop through contours and keypoints
    for pair in zip(cnts, keypoints):
        contour, props = pair
        
        ## Get the bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        ## Check for short edge of bounding box and define bbox corners
        dist1 = math.dist(box[0], box[1])
        dist2 = math.dist(box[1], box[2])        
        if dist1 > dist2:
            top_left_ind = 0
            bot_left_ind = 3
            top_right_ind = 1
            bot_right_ind = 2
        else:
            top_left_ind = 1
            bot_left_ind = 0
            top_right_ind = 2
            bot_right_ind = 3
            
        ## Calculate the midpoints of the short edges of the bbox
        midpoint_l_x = (box[top_left_ind][0] + box[bot_left_ind][0]) / 2
        midpoint_l_y = (box[top_left_ind][1] + box[bot_left_ind][1]) / 2
        midpoint_r_x = (box[top_right_ind][0] + box[bot_right_ind][0]) / 2
        midpoint_r_y = (box[top_right_ind][1] + box[bot_right_ind][1]) / 2        
        midpoint_l = [midpoint_l_x, midpoint_l_y]
        midpoint_r = [midpoint_r_x, midpoint_r_y]
        
        ## Calculate distance from centroid to bbox midpoints
        centroid = [props["cent_x"], props["cent_y"]]
        dis_l = math.dist(centroid, midpoint_l)
        dis_r = math.dist(centroid, midpoint_r)
        
        ## Compare distances and append results
        if props["area"] > avg_area + 2 * stdev_area:
            #print("Screw orientation: Both")
            results.append("both")
        elif dis_l < dis_r:
            #print("Screw orientation: Right")
            results.append("right")
        else:
            #print("Screw orientation: Left")
            results.append("left")
        
        if plot:
            cv2.drawContours(color, [box], -1, (0, 255, 0), 3)
            ax.scatter(midpoint_l_x, midpoint_l_y, color="y", s=2)
            ax.scatter(midpoint_r_x, midpoint_r_y, color="r", s=2)
            ax.scatter(centroid[0], centroid[1], color="b", s=2)
    if plot:
        ax.imshow(mask)
        ax.imshow(color, alpha=0.5)
        plt.show()
        
    return results

def get_linecolor(index):
    linecolors = [
        "#fde725",
        "#d0e11c",
        "#a0da39",
        "#73d056",
        "#4ac16d",
        "#2db27d",
        "#1fa187",
        "#21918c",
        "#277f8e",
        "#2e6e8e",
        "#365c8d",
        "#3f4788",
        "#46327e",
        "#481b6d",
        "#440154"
    ]
    return linecolors[index]