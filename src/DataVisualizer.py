import cv2

class DataVisualizer:

    # --------------------------------------------
    #       Define visualization details
    # --------------------------------------------

    def __init__(self):

        self.connections = [(0, 1), (0, 2), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

        self.hold_class_colours = [(31, 119, 180), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
                              (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
                              (196, 156, 148), (247, 182, 210), (199, 199, 199),( 219, 219, 141), (158, 218, 229), (99, 99, 99)]

        self.classes = ['jug', 'crimp', 'pinch', 'pocket', 'sloper', 'edge', 'jib', 'volume',
                   'slopey_jug', 'slopey_crimp', 'slopey_pinch', 'slopey_pocket', 'slopey_edge',
                   'slopey_jib', 'slopey_volume', 'crimpy_jug', 'crimpy_pinch', 'crimpy_pocket',
                   'crimpy_edge', 'crimpy_jib']


    #------------------------------------------
    #         Draw Hold Annotations
    #------------------------------------------

    def visualize(self, hold_data, pose_data, img_rgb):

        for hold in hold_data:
            color = self.hold_class_colours[hold["class"]]

            x1, y1, x2, y2 = hold["bbox"]
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color=color, thickness=2)

            # Put text just below the bbox
            text = self.classes[hold["class"]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Calculate text size to center it under the bbox
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = int(x2 - text_width)
            text_y = int(y2 + text_height / 2 + 10)

            roi_y1 = max(text_y - text_height - 2, 0)
            roi_y2 = max(text_y + baseline + 2, roi_y1 + 1)  # at least 1 pixel
            roi_x1 = max(text_x - 2, 0)
            roi_x2 = max(text_x + text_width + 2, roi_x1 + 1)

            h, w = img_rgb.shape[:2]
            roi_y1 = max(0, min(roi_y1, h - 1))
            roi_y2 = max(0, min(roi_y2, h))
            roi_x1 = max(0, min(roi_x1, w - 1))
            roi_x2 = max(0, min(roi_x2, w))

            roi = img_rgb[roi_y1:roi_y2, roi_x1:roi_x2]

            # Create overlay for ROI
            overlay = roi.copy()
            alpha = 0.4
            cv2.rectangle(overlay, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), color, -1)

            # Blend overlay with ROI
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

            text_color = tuple([min(255, channel * 2) for channel in color])
            cv2.putText(img_rgb, text, (text_x, text_y), font, font_scale, text_color, thickness)


        #------------------------------------
        #         Draw connections
        #------------------------------------

        for (start, end) in self.connections:
            pt1 = (int(pose_data["keypoints"][start][0]), int(pose_data["keypoints"][start][1]))
            pt2 = (int(pose_data["keypoints"][end][0]), int(pose_data["keypoints"][end][1]))
            cv2.line(img_rgb, pt1, pt2, color=(255, 0, 0), thickness=4)  # red lines


        #-----------------------------------
        #       Draw keypoints
        #-----------------------------------

        for x, y in pose_data["keypoints"]:
            cv2.circle(img_rgb, (int(x), int(y)), radius=6, color=(0, 255, 0), thickness=-1)

        return img_rgb