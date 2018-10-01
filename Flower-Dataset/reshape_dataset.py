import glob
import cv2

flowers = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


for each_flower in flowers:
    i = 0
    for each_flower_image in glob.glob('/Users/roshni/Documents/CODE/flowers_dataset/'+ each_flower+"/*.jpg"):
        print each_flower_image
        reshaped_image = cv2.resize(cv2.imread(each_flower_image), (256, 256))
        cv2.imwrite('/Users/roshni/Documents/CODE/flowers_dataset/reshaped/'+ each_flower+'_'+ str(i)+".jpg", reshaped_image)
        i+=1