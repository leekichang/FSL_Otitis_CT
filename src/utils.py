import os
import cv2
import numpy as np

sift = cv2.SIFT_create()
datapath = './datasets/dataset_unlabeled'
savepath = './datasets/dataset_unlabeled_crop'
# datapath = './dataset_original'
# savepath = './dataset_crop'
class Subject:
    def __init__(self, data_path, subject_id):
        self.subject_id    = subject_id
        self.original_imgs = []
        self.data_path     = f'{data_path}/{self.subject_id}/'
        files = [file for file in os.listdir(self.data_path) if file.endswith('.jpg') or file.endswith('.tif')]
        for file in files:
            img = cv2.imread(self.data_path+file, cv2.IMREAD_GRAYSCALE)
            if img.shape[1] > 512:
                img = cv2.resize(img, dsize = (512, 512), interpolation=cv2.INTER_AREA)
            elif img.shape[1] < 512:
                img = cv2.resize(img, dsize = (512, 512), interpolation=cv2.INTER_LANCZOS4)
            self.original_imgs.append(img)
        self.original_imgs = np.array(self.original_imgs)
        
        self.right_imgs    = self.original_imgs[:, 256:, :256]   # CT image is reversed left and right
        self.left_imgs     = self.original_imgs[:, 256:, 256:]   # CT image is reversed left and right
        for idx, left_img in enumerate(self.left_imgs):
            self.left_imgs[idx] = np.fliplr(left_img)
        self.H, self.W = self.left_imgs[0].shape
        
    def get_best_frame(self, n_sample):
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
        
        if self.left_max_idx + n_sample >= len(self.left_imgs):
            print("FUCK")
            self.left_max_idx = len(self.left_imgs) - n_sample
        if self.right_max_idx + n_sample >= len(self.right_imgs):
            print("FUCK")
            self.right_max_idx = len(self.right_imgs) - n_sample
                
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
        print(self.subject_id, self.left_max_idx, self.right_max_idx)
        plot_left  = cv2.cvtColor(self.left_imgs[self.left_max_idx], cv2.COLOR_GRAY2BGR)
        plot_right = cv2.cvtColor(self.right_imgs[self.right_max_idx], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(plot_left, (self.w_idx*stride, self.h_idx*stride), (window_size+self.w_idx*stride, window_size+self.h_idx*stride), (255, 0, 0), 2)
        cv2.rectangle(plot_right, (self.w_jdx*stride, self.h_jdx*stride), (window_size+self.w_jdx*stride, window_size+self.h_jdx*stride), (0, 255, 0), 2)
        cv2.imwrite(f'./{savepath}_{self.window_size}/{self.subject_id}/{self.subject_id}_left.png', plot_left)
        cv2.imwrite(f'./{savepath}_{self.window_size}/{self.subject_id}/{self.subject_id}_right.png', plot_right)
        cv2.imwrite(f'./{savepath}_{self.window_size}/{self.subject_id}/{self.subject_id}_left_crop.png', self.crop_left)
        cv2.imwrite(f'./{savepath}_{self.window_size}/{self.subject_id}/{self.subject_id}_right_crop.png', self.crop_right)
        
    
    def get_crop_volume(self, save_path, window_size = 112, n_sample=25):
        self.get_best_frame(n_sample)
        self.get_best_window(window_size=window_size)
        for file in os.listdir(f'./{save_path}_{self.window_size}/{self.subject_id}/left'):
            os.remove(f'./{save_path}_{self.window_size}/{self.subject_id}/left/'+file)
        for file in os.listdir(f'./{save_path}_{self.window_size}/{self.subject_id}/right'):
            os.remove(f'./{save_path}_{self.window_size}/{self.subject_id}/right/'+file)
        for i in range(n_sample):
            cv2.imwrite(f'./{save_path}_{self.window_size}/{self.subject_id}/left/{self.subject_id}_left_{self.left_max_idx-i}.png', \
                cv2.resize(self.left_imgs[self.left_max_idx-i, self.h_idx*self.stride:self.window_size+self.h_idx*self.stride, self.w_idx*self.stride:self.window_size+self.w_idx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))
            cv2.imwrite(f'./{save_path}_{self.window_size}/{self.subject_id}/left/{self.subject_id}_left_{self.left_max_idx+i}.png', \
                cv2.resize(self.left_imgs[self.left_max_idx+i, self.h_idx*self.stride:self.window_size+self.h_idx*self.stride, self.w_idx*self.stride:self.window_size+self.w_idx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))
            cv2.imwrite(f'./{save_path}_{self.window_size}/{self.subject_id}/right/{self.subject_id}_right_{self.right_max_idx-i}.png', \
                cv2.resize(self.right_imgs[self.right_max_idx-i, self.h_jdx*self.stride:self.window_size+self.h_jdx*self.stride, self.w_jdx*self.stride:self.window_size+self.w_jdx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))
            cv2.imwrite(f'./{save_path}_{self.window_size}/{self.subject_id}/right/{self.subject_id}_right_{self.right_max_idx+i}.png', \
                cv2.resize(self.right_imgs[self.right_max_idx+i, self.h_jdx*self.stride:self.window_size+self.h_jdx*self.stride, self.w_jdx*self.stride:self.window_size+self.w_jdx*self.stride], (224, 224), interpolation=cv2.INTER_LANCZOS4))

if __name__ == '__main__':
    folders = [folder for folder in os.listdir(datapath) if os.path.isdir(f'{datapath}/{folder}')]
    for folder in folders:
        subject = Subject(datapath, folder)
        subject.get_crop_volume(savepath, window_size=144)
    
    # subject = Subject(datapath, '2758306')
    # subject.get_crop_volume(savepath, window_size=144)