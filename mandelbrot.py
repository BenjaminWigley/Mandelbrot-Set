import numba
from numba import cuda
import numpy as np
import time
from PIL import Image



def visibleArea(pixelsPerUnit, resX, resY, centreX, centreY):
    xMin = centreX - (resX/2) / pixelsPerUnit
    xMax = centreX + (resX/2) / pixelsPerUnit
    yMin = centreY - (resY/2) / pixelsPerUnit
    yMax = centreY + (resY/2) / pixelsPerUnit
    return xMin, xMax, yMin, yMax

def zeroFill(inputNumber, requiredLength):
    number = str(inputNumber)
    numberLength = len(number)
    while numberLength < requiredLength:
        number = "0" + number
        numberLength = len(number)

    return str(number)

@cuda.jit(device=True)
def mandelbrot(x, y, max_iters):
    startingPoint = complex(x, y)
    z = startingPoint
    for iteration in range(max_iters):
        z = (z*z) + startingPoint
        if (z.real ** 2 + z.imag ** 2) >= 4:
            return iteration
    return -1

@cuda.jit
def fillArray(start_x, start_y, pixel_x, pixel_y, width, height, max_iters, array):
  x, y = cuda.grid(2)
  if x < width and y < height:
    real = start_x + x * pixel_x
    imag = start_y + ((height-y-1) * pixel_y)
    #height-y-1 flips around the y value from 0 being top of image to 0 being the bottom

    real2 = start_x + (x+0.5) * pixel_x
    imag2 = start_y  + ((height-y-1)+0.5) * pixel_y

    color = mandelbrot(real, imag, max_iters)
    color2 = mandelbrot(real2, imag2, max_iters)

    averageColor = (color + color2)/2
    #Second pass

    array[y, x] = averageColor
    
@cuda.jit
def colorPixels(d_pixels, d_data, width, height, colorPoints, theColours, gradientRepeats):
    x, y = cuda.grid(2)
    if x < width and y < height:
        positionAlongGradient = (int(d_data[y, x]) % gradientRepeats) / gradientRepeats

        i = 0
        while colorPoints[i] <= positionAlongGradient:
            i += 1

        topColor = theColours[i]
        bottomColor = theColours[i-1]
        
        posBetweenColors = (positionAlongGradient - colorPoints[i-1]) / (colorPoints[i] - colorPoints[i-1])

        r = int(bottomColor[0] + posBetweenColors / 1 * (topColor[0]-bottomColor[0]))
        g = int(bottomColor[1] + posBetweenColors / 1 * (topColor[1]-bottomColor[1]))
        b = int(bottomColor[2] + posBetweenColors / 1 * (topColor[2]-bottomColor[2]))

        d_pixels[y, x] = (r, g, b)
        

def calculateMandelbrot(width, height, max_iters, min_x, max_x, min_y, max_y, gradientRepeats):
  
  threads = 16
    
  blocksX = width // threads + 1
  blocksY = height // threads + 1

  pixel_x = (max_x - min_x) / width
  pixel_y = (max_y - min_y) / height

  data = np.zeros((height, width))
  
  startTime = time.perf_counter()
  
  d_data = cuda.to_device(data)
  
  fillArray[(blocksX, blocksY), (threads, threads)](min_x, min_y, pixel_x, pixel_y, width, height, max_iters, d_data)
  
  pixels = Image.new('RGB', (width, height))
  pixels = np.asarray(pixels) #Gets the array in the right format
  
  d_pixels = cuda.to_device(pixels)
  
  colorPoints = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
  theColours = np.array([[255, 255, 255], [255, 204, 0], [135, 30, 20], [0, 0, 153], [0, 0, 153], [255, 255, 255]])
  difference = np.zeros(3)
  
  d_colorPoints = cuda.to_device(colorPoints)
  d_theColours = cuda.to_device(theColours)
  
  colorPixels[(blocksX, blocksY), (threads, threads)](d_pixels, d_data, width, height, d_colorPoints, d_theColours, gradientRepeats)
  
  colorPoints = d_colorPoints.copy_to_host()
  theColours = d_theColours.copy_to_host()
  
  data = d_data.copy_to_host()
  pixels = d_pixels.copy_to_host()
  
  endTime = time.perf_counter()
  
  print(f"Time to Calculate: {(endTime - startTime)}" )
  
  im = Image.fromarray(pixels, mode="RGB")

  return im, data

#Resolution of output image
xRes = 800
yRes = 600

maxIterations = 15000
gradientRepeats = 100

requiredLength = 4 #for name of file output

xMin = -1.574590111060041378352
xMax = -1.574589863820997445596
yMin = 0.000273836775278080587
yMax = 0.000274022091563295267

image, data = calculateMandelbrot(xRes, yRes, maxIterations, xMin, xMax, yMin, yMax, gradientRepeats)
fileName = zeroFill(1, 4)
image.save(fileName+".png")


