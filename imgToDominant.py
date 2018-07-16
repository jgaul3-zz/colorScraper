'''
This script iterates through a given comic on tapas.io, downloads each page
from each chapter, performs k-means clustering on that page, and saves the
resulting colors in an image file.
Author: Jonathan Gaul
Date: July 13, 2018
'''
import urllib2, re, cv2, requests
import numpy as np
from PIL import Image
from collections import Counter
from bs4 import BeautifulSoup
from StringIO import StringIO

def main():
    # User parameters
    initialId = "143458"
    dominantColors = 3
    colWidth = 200
    barThickness = 3

    # Removing pure black/pure white/color of speech bubbles can lead to better results
    colorBlacklist = [[0, 0, 0], [201, 201, 201], [255, 255, 255]]

    print "Preparing to process comic."
    soup = getSoup(initialId)
    script = soup.find("script", text=lambda text: text and "episodeList" in text)
    episodeIdRegex = re.compile(r'\"id\":(\d*)')
    idArray = episodeIdRegex.findall(script.text)

    index = 0
    localPixels = []
    outputWidth = colWidth * dominantColors

    for chapter in idArray:
        soup = getSoup(chapter)
        imgSrcs = soup.findAll("img", {"class":"art-image"})
        for imgSrc in imgSrcs:
            while True:
                try:
                    print "# Downloading page " + str(index) + " at " + imgSrc.get("src")
                    response = requests.get(imgSrc.get("src"))
                    image = Image.open(StringIO(response.content))
                    img = np.array(image.getdata(), dtype=np.uint8)
                    if len(img.shape) < 2:
                        # removes black and white images if any exist
                        break
                    elif img.shape[1] is 4:
                        # delete alpha channel if it exists
                        img = np.delete(img, 3, axis=1)

                    print "# Processing..."
                    colorArray = imgToDominant(img, dominantColors, colorBlacklist)
                    colorBar = generateColorBar(colorArray, outputWidth, colWidth, barThickness)
                    localPixels.extend(colorBar)
                    index += 1
                except Exception, e:
                    print str(e)
                    response = input("Encountered error! Retry/Skip Page/Quit [0/1/2]")
                    while response not in range(3):
                        response = input("Please input valid number [0/1/2]")
                    if response is 0:
                        continue
                    elif response is 1:
                        break
                    else:
                        return
                break
    print "Analysis finished!"

    localPixels = np.asarray(localPixels).astype('uint8').reshape(-1, outputWidth, 3)
    colorMap = Image.fromarray(localPixels, 'RGB')
    colorMap.save('sorted.png')
    colorMap.show()


def getSoup(chapter):
    req = urllib2.Request("http://tapas.io/episode/" + chapter,
        headers={'User-Agent' : "Magic Browser"})
    html = urllib2.urlopen(req)
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def imgToDominant(img, dominantColors, colorBlacklist):
    for color in colorBlacklist:
        mask = np.argwhere(np.all(img == color, axis=1))
        img = np.delete(img, mask, 0)

    toProcess = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(toProcess, dominantColors + 2, None,
        criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return sortByBrightness(center, label, dominantColors)


def sortByBrightness(center, label, dominantColors):
    brightness = np.average(center, axis=1)
    center = np.uint8(center)
    brightness = np.uint8(brightness)
    counts = Counter(label.flatten()).most_common(dominantColors)
    colorArray = [center[counts[x][0]] for x in range(dominantColors)]
    brightnessArray = [brightness[counts[x][0]] for x in range(dominantColors)]
    colorArray = [colorArray[x] for x in np.flip(np.argsort(brightnessArray), axis=0)]
    return colorArray


def generateColorBar(colorArray, outputWidth, colWidth, barThickness):
    numPixels = outputWidth * barThickness
    colorBlock = [[0, 0, 0] for i in range(numPixels)]
    for i in range(numPixels):
        colorBlock[i] = colorArray[(i % outputWidth) // colWidth]
    return colorBlock


if __name__ == "__main__":
    main()
