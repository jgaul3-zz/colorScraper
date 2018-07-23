'''
This script iterates through a given comic on tapas.io, downloads each page
from each chapter, performs k-means clustering on that page, and saves the
resulting colors in an image file.

This variant uses the multiprocessing library to do some of the tasks in
parallel, resulting in about 50% speedup.

Author: Jonathan Gaul
Date: July 13, 2018
'''
import urllib2, re, cv2, requests, time
import numpy as np
from PIL import Image
from collections import Counter
from bs4 import BeautifulSoup
from StringIO import StringIO

from multiprocessing.dummy import Pool as ThreadPool

# User parameters
# Comic's first page ID
initialId = "143458"
# Number of dominant colors to get
numDomColors = 3
# Width of color column in pixels
colWidth = 200
# Thickness of color bar in pixels
barThickness = 3
# Number of threads (If multithreading)
threads = 2
# Removing pure black/pure white/color of speech bubbles can lead to better results
colorBlacklist = [[0, 0, 0], [201, 201, 201], [255, 255, 255]]
# Set to 0 to get less vibrant color output
tweak = 2

outputWidth = colWidth * numDomColors
prev = time.time()


def main():
    pool = ThreadPool(threads)

    print "# Extracting chapters."
    soup = getSoup(initialId)
    script = soup.find("script", text=lambda text: text and "episodeList" in text)
    episodeIdRegex = re.compile(r'\"id\":(\d*)')
    idArray = episodeIdRegex.findall(script.text)
    printElapsedTime()

    print "# Downloading image urls"
    imgSrcs = pool.map(getPageSrcs, idArray)
    imgSrcs = [item for sublist in imgSrcs for item in sublist]
    printElapsedTime()

    print "# Downloading images"
    imgs = pool.map(downloadImg, imgSrcs)
    imgs = filter(lambda x: np.any(x), imgs)
    printElapsedTime()

    print "# Extracting dominant colors"
    dominantColors = pool.map(imgToDominant, imgs)
    printElapsedTime()

    print "# Generating output image"
    colorBars = pool.map(generateColorBar, dominantColors)
    printElapsedTime()

    pixels = [item for sublist in colorBars for item in sublist]
    print "# Analysis finished!"

    pool.close()
    pool.join()

    pixels = np.asarray(pixels).astype('uint8').reshape(-1, outputWidth, 3)
    colorMap = Image.fromarray(pixels, 'RGB')
    colorMap.save('sorted.png')
    colorMap.show()


def printElapsedTime():
    global prev
    print "# Elapsed time: " + str(int(time.time() - prev)) + " sec"
    prev = time.time()


def getSoup(chapter):
    req = urllib2.Request("http://tapas.io/episode/" + chapter,
        headers={'User-Agent' : "Magic Browser"})
    html = urllib2.urlopen(req)
    return BeautifulSoup(html, 'html.parser')


def getPageSrcs(chapter):
    soup = getSoup(chapter)
    return soup.findAll("img", {"class":"art-image"})


def downloadImg(imgSrc):
    response = requests.get(imgSrc.get("src"))
    img = np.array(Image.open(StringIO(response.content)).getdata(), dtype=np.uint8)
    if len(img.shape) < 2:
        # removes black and white images if any exist
        return []
    elif img.shape[1] is 4:
        # delete alpha channel if it exists
        img = np.delete(img, 3, axis=1)
    return img


def imgToDominant(img):
    for color in colorBlacklist:
        mask = np.argwhere(np.all(img == color, axis=1))
        img = np.delete(img, mask, 0)
    toProcess = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(toProcess, numDomColors + tweak, None,
        criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return sortByBrightness(center, label)


def sortByBrightness(center, label):
    center = np.uint8(center)
    counts = Counter(label.flatten()).most_common(numDomColors)
    colorArray = [center[counts[x][0]] for x in range(numDomColors)]
    brightness = np.average(colorArray, axis=1)
    return [colorArray[x] for x in np.flip(np.argsort(brightness), axis=0)]


def generateColorBar(colorArray):
    numPixels = outputWidth * barThickness
    colorBlock = [[0, 0, 0] for i in range(numPixels)]
    for i in range(numPixels):
        colorBlock[i] = colorArray[(i % outputWidth) // colWidth]
    return colorBlock


if __name__ == "__main__":
    main()
