import cv2
import numpy as np
import bisect
import time



def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img



    dst = src.copy()
    tol = max(0, min(50, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst

def convert_grayscale(image):
  # Get size
  width, height,_ = np.shape(image)




  # Transform to grayscale
  for i in range(width):
      for j in range(height):


          pixel = image[i][j]

          #print(pixel)

          red = pixel[2]
          blue = pixel[0]
          green = pixel[1]
          #total = red+blue+green
          # total = 1
          # if (total) == 0:
          #     greenupdated = 0#green/100
          #     blueupdated = 0#blue/100
          #     redupdated = 0#red/100
          #
          # else:
          #     greenupdated = green/total
          # #greenupdated = green
          #     redupdated = max(0,min((red-blue),(red-green))/total)
          #     blueupdated = max(0,((blue-red))/total)
              #print('1', greenupdated)
              #redupdated = np.max(0,(np.min(blue-red))/red+green+blue)
              #print(red-green)

          #gray = 0*greenupdated+blueupdated+0*redupdated
          #gray = 0.1*greenupdated+0.2*blueupdated+0.2*redupdated
          #gray = (0.4*redupdated + 0*greenupdated + 0.3*blueupdated)
          gray = (1*red + 0*green+0*blue)
          #gray = greenupdated+redupdated+blueupdated


          #print(new[i][j])
          #image[i][j] = [0.7*int(blueupdated),0*int(greenupdated),0.2*int(redupdated)]
          #image[i][j] = [int(gray),int(gray),int(gray)]
          image[i][j] = [(gray), (gray), (gray)]
          #image[i][j] = [0, 0, 1*int(red)]
  cv2.imshow('',image)
  cv2.waitKey()
  return image




  # cv2.imshow('',image)
  # cv2.waitKey(0)
  #cv2.imwrite('',new.jpg)


  # for i in range(width):
  #   for j in range(height):
  #     # Get Pixel
  #         pixel = image[i][j]
  #         print('pixel',pixel)
  #         #print('pixel',pixel)
  #
  #         # Get R, G, B values (This are int from 0 to 255)
  #         red =   pixel[0]
  #         print('red',red)
  #         green = pixel[1]
  #         blue =  pixel[2]
  #
  #         # Transform to grayscale
  #         gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)
  #
  #
  #         new[i,j] = ((255,255,255))#((gray),(gray),(gray))
  #         print(i)
  #      # Set Pixel in new image
  #
  #   # Return new image

def hsv(img,OG):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # lower_blue = np.array([95,110,20])
    # upper_blue = np.array([130,255,255])
    #
    # lower_red = np.array([0,100,60])
    # upper_red = np.array([20,255,255])

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([110, 255, 255])

    # lower_blue = np.array([38, 86, 0])
    # upper_blue = np.array([121, 255, 255])

    lower_red = np.array([170, 70, 20])
    upper_red = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv,lower_blue,upper_blue)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2

    # contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # img = cv2.drawContours(OG,contours,-1,(0,255,0),3)




    return img,mask

def grayscale(c1,c2,c3,img):
    width, height= np.shape(c1)
    k = 0
    for i in range(width):
        for j in range(height):



            blue = c1[i][j]
            green = c2[i][j]
            red = c3[i][j]

            # if blue+red+green==0:
            #     print(k)
            #     k = k+1

            #blue = max(0,min((blue-red),(blue-green)))
            red = max(0,min((red-blue),(red-green)))

            # hsv = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

            # lower_red = np.array([10, 100, 60])
            # upper_red = np.array([20, 255, 255])
            #
            # mask2 = cv2.inRange(red, lower_red, upper_red)
            #
            #
            # img = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(img, 127, 255, 0)
            # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

            green = green

            c1[i][j] = blue
            c2[i][j] = green
            c3[i][j] = red

    cv2.imshow("c3",c3)

    ret, thresh = cv2.threshold(c3, 20, 100, 0)

    #cv2.imshow("thresh", thresh)

    image = cv2.merge((c1,c2,c3))



    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)





    lower_red = np.array([0,10,15])
    upper_red = np.array([40,95,115])
    mask2 = cv2.inRange(image, lower_red, upper_red)



    FinImg = cv2.bitwise_and(img,img,mask=mask2)

    FinImg2 = cv2.addWeighted(FinImg,3,FinImg,1,1)
    FinImg2 = cv2.add(FinImg2,FinImg2)
    FinImg2 = cv2.add(FinImg2, FinImg2)


    GrayScale_Img = cv2.cvtColor(FinImg2, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("Gray Image", GrayScale_Img)
    #cv2.imshow("masked", FinImg2)
    #
    #cv2.waitKey(0)
    """
    #img = cv2.cvtColor(mask2, cv2.COLOR_BAYER_RG2GRAY)
    #cv2.imshow("gray",img)
    ret, thresh = cv2.threshold(mask2, 5, 255, 0)
    _,contours= cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img2 = cv2.drawContours(mask2, contours, -1, (0, 255,255), 5)
    cv2.imshow("contour",img2)


    #cv2.imshow(" ", image)
    cv2.waitKey(0)
    """
    return GrayScale_Img

for i in range(2032,2052):
#for i in range(785,795):

    start_time = time.time()

    #img= cv2.imread('Data/TSR/input/image.0{}.jpg'.format(i))
    img = cv2.imread('denoised_input/denoised_input/image{}.jpg'.format(i))
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    OG = img
    #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10,7, 21)
    # img = cv2.GaussianBlur(img,(5,5),0)
    # img = cv2.bilateralFilter(img,9,75,75)
    b, g, r = cv2.split(img)

    b1 = imadjust(b)


    g1 = imadjust(g)

    r1 = imadjust(r)

    image = cv2.merge((b1,g1,r1))
    cv2.imshow("seaf", image)
    #dst = np.zeros(shape=(5,2))

    #b=cv2.normalize(image,dst,0,255,cv2.NORM_L1)

    #cv2.imshow("normalized and merged",b)


    image2 = grayscale(b1,g1,r1, img)
    cv2.imshow('mergedfin',image2)
    #cv2.waitKey(0)
    #

    #cv2.imwrite('Dataset/img{}.jpg'.format(i),img)
    print(time.time() - start_time)




    #img ,mask= hsv(img,OG)


    #img = convert_grayscale(img)

    #cv2.imshow('',mask)
    #cv2.waitKey()
    # cv2.imshow('',img)
    # cv2.waitKey()
    # break
    #break



    #mser = cv2.MSER_create(_delta=5, _min_area=500, _max_area=5000)
    mser = cv2.MSER_create()
    regions = mser.detectRegions(image2)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    img = cv2.polylines(OG, hulls, 1, (0, 255, 0))
    cv2.imshow('Detection',img)
    cv2.waitKey(1)



    # mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    #
    # for contour in hulls:
    #     cv2.drawContours(img, [contour], -1, (255, 255, 255), -1)
    #
    # # # this is used to find only text regions, remaining are ignored
    # text_only = cv2.bitwise_and(img, img, mask=mask)
    # #
    # cv2.imshow("text only", text_only)





    # img = imadjust(img)


    #


    # img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img = cv2.equalizeHist(img)
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #img = imadjust(img)
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.hconcat([img,img1])
    # cv2.imshow('',img)
    #cv2.imwrite('image{}.jpg'.format(i),img)
    # cv2.waitKey(0)
    # #print(i)
    # break







#im1 = cv2.imread('Data/TSR/Training/00000/01153_00000.ppm')
# im1 = cv2.imread('Data/TSR/Testing/00000/00017_00000.ppm')
# #im1 = cv2.resize(im1,(720,1080))
# cv2.imshow('',im1)
# cv2.waitKey()