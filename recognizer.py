import cv2
import numpy as np
import tensorflow as tf

class Recognizer:
    """The recognizer for siamese model.
    """
    def __init__(self, BACKBONE):
        self.load_model(BACKBONE)
#   ===========getter=========
    @property
    def img(self):
        return self.__img
    
    @property
    def img_full(self):
        return self.__img_full

    @property
    def gt(self):
        return self.__gt.astype(np.uint8)
    
    @property
    def pred(self):
        return self.__pred #.astype(np.uint8)
    
    @property
    def model(self):
        return self.__model.summary()

    def set_inputs(self, img_path: str, gt_path: str, resolution: float) -> None:
        """To set the inputs of model.
        Args:
            img_path: The original satellite image.
            gt_path: The ground truth of original image.
            resolution: The resolution of satellite image.
                The resolution of model is 2.5x2.5m.
        """
        image = cv2.imread(img_path)
        self.__img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.__img_full = cv2.resize(self.__img, (256,256), interpolation=cv2.INTER_CUBIC) # cv2.imread(img_full_path)
        
        if gt_path == '':
            self.__gt = np.zeros(image.shape)[:,:,0]
        else:
            self.__gt = cv2.imread(gt_path)
        
#         check image and ground truth are the same size
        if self.__img[:,:,0].shape != self.__gt.shape:
            raise BaseException("Image size and GT size are not matching.")
        else:
            h_tmp, w_tmp, _ = self.img.shape
            scale = lambda x: int(x*resolution/2.5)
            self.__img = cv2.resize(self.__img, (scale(w_tmp), scale(h_tmp)), interpolation=cv2.INTER_CUBIC)
            self.__gt = cv2.resize(self.__gt, (scale(w_tmp), scale(h_tmp)), interpolation=cv2.INTER_CUBIC)
            self.__w, self.__h, self.__c = self.img.shape

    
    def set_predict(self):
        """To set prediction image.
        """
        pred_tmp = np.zeros((self.__w, self.__h))
        w_iter_all = int(self.__w/256)-1
        h_iter_all = int(self.__h/256)-1
        i=0
        for w_ind in range(w_iter_all):
            w_s = w_ind*256
            w_e = (w_ind+1)*256
            for h_ind in range(h_iter_all):
                i+=1
                print(f"{i*100/(w_iter_all*h_iter_all):.2f}%", end='\r')
                h_s = h_ind*256
                h_e = (h_ind+1)*256
                img_slice = self.img[w_s:w_e, h_s:h_e, :]
                pred_tmp[w_s:w_e, h_s:h_e] = self.__predict(img_slice)
        self.__pred = pred_tmp
            
    def __predict(self, img):
        """To predict cropped image.
        Args:
            img: The cropped image.

        Returns:
            The predicted image.
        """
        x = [np.array([img], dtype=np.float32), np.array([self.__img_full], dtype=np.float32)]
        pr_mask = self.__model.predict(x).round()

        return pr_mask[..., 0]
    
    def predict(self,img_path: str, gt_path: str, resolution: float):
        """The prediction function.
        Args:
            img_path: The original satellite image.
            gt_path: The ground truth of original image.
            resolution: The resolution of satellite image.
                The resolution of model is 2.5x2.5m.
        """
        self.set_inputs(img_path, gt_path, resolution)
        self.set_predict()
        
    def load_model(self, BACKBONE: str):
        """To load pre-trained model.
        """
        if BACKBONE=='siameseUnet':
            self.__model = tf.keras.models.load_model('your/model/path/model.h5', 
            compile=False,
            custom_objects={'leaky_relu': tf.nn.leaky_relu})
        else:
            raise BaseException('')


if __name__ == '__main__':
    recog = Recognizer('siameseUnet')
    res = 2.5
    img_path = 'image.jpg'
    gt_path = 'gt.png'
    recog.predict(img_path, gt_path, res_li[num])
    cv2.imwrite('res.png', recog.pred*255)
