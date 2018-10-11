import cv2 as cv

# image = cv.imread("~/Fami/fami1.jpg")
# print(image)

with open("positives.txt","w") as file:
    for count in range (1,102):
        file.write("/Users/yamamotomasaomi/Fami/fami"+str(count)+".jpg"+" "+ "1 1 1 250 250" +"\n")
        count += 1
