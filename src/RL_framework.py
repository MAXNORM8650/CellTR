import gym
import numpy as np
import cv2
import os
from gym import spaces

class CellTrackingEnv(gym.Env):
    def __init__(self, dataset_path):
        super(CellTrackingEnv, self).__init__()
        
        self.dataset_path = dataset_path
        self.frame_idx = 0
        self.frames = self.load_frames()
        self.ground_truths = self.load_ground_truths()
        self.current_bbox = None
        self.reset()

        # Define action space: [move_x, move_y, resize_w, resize_h]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Define observation space: Image and bounding box coordinates
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(self.frames[0].shape[0], self.frames[0].shape[1], 3),
                                            dtype=np.uint8)

    def load_frames(self):
        # Load all frames from the dataset (simple implementation)
        frames = []
        for file_name in sorted(os.listdir(os.path.join(self.dataset_path, "01"))):
            if file_name.endswith(".tif"):
                img = cv2.imread(os.path.join(self.dataset_path, "01", file_name))
                frames.append(img)
        return frames

    def load_ground_truths(self):
        # Load ground truth bounding boxes (dummy implementation for now)
        ground_truths = []
        for file_name in sorted(os.listdir(os.path.join(self.dataset_path, "01_GT", "TRA"))):
            if file_name.endswith(".tif"):
                gt = cv2.imread(os.path.join(self.dataset_path, "01_GT", "TRA", file_name), cv2.IMREAD_GRAYSCALE)
                x, y, w, h = cv2.boundingRect(gt)
                ground_truths.append([x, y, w, h])
        return ground_truths

    def reset(self):
        self.frame_idx = 0
        self.current_bbox = self.ground_truths[self.frame_idx]
        return self.get_observation()

    def step(self, action):
        # Apply action to bounding box
        x, y, w, h = self.current_bbox
        move_x, move_y, resize_w, resize_h = action
        x += int(move_x * w)
        y += int(move_y * h)
        w = max(1, w + int(resize_w * w))
        h = max(1, h + int(resize_h * h))
        self.current_bbox = [x, y, w, h]

        # Calculate reward
        reward = self.calculate_reward()

        # Move to next frame
        self.frame_idx += 1
        done = self.frame_idx >= len(self.frames)

        return self.get_observation(), reward, done, {}

    def calculate_reward(self):
        # Dummy reward calculation based on IoU with ground taruth
        gt_bbox = self.ground_truths[self.frame_idx]
        iou = self.calculate_iou(self.current_bbox, gt_bbox)
        return iou

    def calculate_iou(self, bbox1, bbox2):
        # Intersection over Union calculation
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area

    def get_observation(self):
        # Return the current frame with the bounding box
        frame = self.frames[self.frame_idx].copy()
        x, y, w, h = self.current_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def render(self, mode='human'):
        # Render the environment
        if mode == 'human':
            cv2.imshow('Tracking', self.get_observation())
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

# Example usage:
env = CellTrackingEnv('/home/komal.kumar/Documents/Cell/datasets/celltrack/2D/BF-C2DL-HSC')
obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        break

env.close()
