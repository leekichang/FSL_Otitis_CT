import os
import cv2
import numpy as np

class Subject:
    def __init__(self, is_left, subject_):
        self.subject_id    = subject_id
        self.original_imgs = []
        self.data_path     = f'../dataset/{self.subject_id}/'
        files = [file for file in os.listdir(self.data_path) if file.endswith('.tif')]
        for file in files:
            self.original_imgs.append(cv2.imread(self.data_path+file, cv2.IMREAD_GRAYSCALE))
        self.original_imgs = np.array(self.original_imgs)
        self.right_imgs    = self.original_imgs[:,:,:256]   # CT image is reversed left and right
        self.left_imgs     = self.original_imgs[:,:,256:]   # CT image is reversed left and right
        for idx, left_img in enumerate(self.left_imgs):
            self.left_imgs[idx] = np.fliplr(left_img)
        self.H, self.W = self.left_imgs[0].shape
        
    def get_best_frame(self):
        self.left_max      = -1
        self.left_max_idx  = 0
        self.right_max     = -1
        self.right_max_idx = 0
        for idx, left_img in enumerate(self.left_imgs):
            kp = sift.detect(left_img, None)
            if len(kp) > self.left_max:
                self.left_max = len(kp)
                self.left_max_idx = idx
            kp = sift.detect(self.right_imgs[idx], None)
            if len(kp) > self.right_max:
                self.right_max = len(kp)
                self.right_max_idx = idx
                
    def get_best_window(self, window_size=112, stride=8):
        assert window_size < self.W and window_size < self.H
        self.window_size = window_size
        self.stride = stride
        col_step = (self.W-window_size)//stride
        row_step = (self.H-window_size)//stride
        kp = sift.detect(self.left_imgs[self.left_max_idx])
        self.left_points = np.array(kp[0].pt)
        for point in kp[1:]:
            self.left_points = np.vstack((self.left_points, np.array(point.pt)))

        self.h_idx    = 0
        self.w_idx    = 0
        self.sift_max = 0
        for row in range(row_step):
            for col in range(col_step):
                count = sum(np.where(np.logical_and(np.logical_and(self.left_points[:, 1] >= row*self.stride, self.left_points[:, 1] <= row*self.stride+self.window_size),\
                                                    np.logical_and(self.left_points[:, 0] >= col*self.stride, self.left_points[:, 0] <= col*self.stride+self.window_size)), 1, 0))
                if count > self.sift_max:
                    self.sift_max = count
                    self.h_idx = row
                    self.w_idx = col
        self.crop_left = cv2.resize(self.left_imgs[self.left_max_idx, self.h_idx*self.stride:self.window_size+self.h_idx*self.stride, self.w_idx*self.stride:self.window_size+self.w_idx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4)
                
        kp = sift.detect(self.right_imgs[self.right_max_idx])
        self.right_points = np.array(kp[0].pt)
        for point in kp[1:]:
            self.right_points = np.vstack((self.right_points, np.array(point.pt)))
        self.h_jdx    = 0
        self.w_jdx    = 0
        self.sift_max = 0
        for row in range(row_step):
            for col in range(col_step):
                count = sum(np.where(np.logical_and(np.logical_and(self.right_points[:, 1] >= row*self.stride, self.right_points[:, 1] <= row*self.stride+self.window_size),\
                                                    np.logical_and(self.right_points[:, 0] >= col*self.stride, self.right_points[:, 0] <= col*self.stride+self.window_size)), 1, 0))
                if count > self.sift_max:
                    self.sift_max = count
                    self.h_jdx = row
                    self.w_jdx = col                    
        
        self.crop_right = cv2.resize(self.right_imgs[self.right_max_idx, self.h_jdx*self.stride:self.window_size+self.h_jdx*self.stride, self.w_jdx*self.stride:self.window_size+self.w_jdx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    def get_crop_volume(self, n_sample=25):
        self.get_best_frame()
        self.get_best_window(window_size=112)
        for i in range(n_sample):
            cv2.imwrite(f'./sift_base_crop_{self.window_size}/{self.subject_type}/{self.subject_id}/left/{self.subject_id}_left_{self.left_max_idx-i}.png', \
                cv2.resize(self.left_imgs[self.left_max_idx-i, self.h_idx*self.stride:self.window_size+self.h_idx*self.stride, self.w_idx*self.stride:self.window_size+self.w_idx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))
            cv2.imwrite(f'./sift_base_crop_{self.window_size}/{self.subject_type}/{self.subject_id}/left/{self.subject_id}_left_{self.left_max_idx+i}.png', \
                cv2.resize(self.left_imgs[self.left_max_idx+i, self.h_idx*self.stride:self.window_size+self.h_idx*self.stride, self.w_idx*self.stride:self.window_size+self.w_idx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))
            cv2.imwrite(f'./sift_base_crop_{self.window_size}/{self.subject_type}/{self.subject_id}/right/{self.subject_id}_right_{self.right_max_idx-i}.png', \
                cv2.resize(self.right_imgs[self.right_max_idx-i, self.h_jdx*self.stride:self.window_size+self.h_jdx*self.stride, self.w_jdx*self.stride:self.window_size+self.w_jdx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))
            cv2.imwrite(f'./sift_base_crop_{self.window_size}/{self.subject_type}/{self.subject_id}/right/{self.subject_id}_right_{self.right_max_idx+i}.png', \
                cv2.resize(self.right_imgs[self.right_max_idx+i, self.h_jdx*self.stride:self.window_size+self.h_jdx*self.stride, self.w_jdx*self.stride:self.window_size+self.w_jdx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))