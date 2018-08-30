import cv2 as cv

image = cv.imread("~/Fami/fami1.jpg")
print(image)

# with open("path_and_label.txt","w") as file:
#     for count in range (0,103):
#         file.write("/Users/yamamotomasaomi/Familymart/fami"+str(0)+".JPG"+" "+ "1" +"¥n")
#         count += 1
#     for count in range (103,113):
#         file.write("/Users/yamamotomasaomi/Familymart/fami"+str(0)+".JPG"+" "+ "0" +"¥n")
#         count += 1