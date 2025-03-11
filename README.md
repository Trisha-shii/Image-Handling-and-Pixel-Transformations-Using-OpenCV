# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** TRISHA PRIYADARSHNI PARIDA
- **Register Number:** 212224230293

  ### Ex. No. 01
  
### IMPORT LIBRARIES COMMON FOR ALL CODING SNIPPETS BELOW :-
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```
image_gray = cv2.imread(r"C:\Users\Trisha Priyadarshni\Downloads\Eagle_in_Flight.jpg", cv2.IMREAD_GRAYSCALE)
print(image_gray)
```

#### 2. Print the image width, height & Channel.
```
height, width = image_gray.shape
print(f'Width: {width}, Height: {height}')
```

#### 3. Display the image using matplotlib imshow().
```
plt.imshow(image_gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```
cv2.imwrite('Eagle_in_Flight_gray.png', image_gray)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```
image_color = cv2.imread(r"C:\Users\Trisha Priyadarshni\Downloads\Eagle_in_Flight.jpg",cv2.IMREAD_COLOR)
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```
plt.imshow(image_color)
plt.title("Color Image")
plt.axis('off')
plt.show()
print(f'Width: {image_color.shape[1]}, Height: {image_color.shape[0]}, Channels: {image_color.shape[2]}')
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```
cropped_image = image_color[100:400, 150:450]
plt.imshow(cropped_image, cmap='gray')
plt.title("Cropped Image")
plt.axis('off')
plt.show()
```

#### 8. Resize the image up by a factor of 2x.
```
resized_image = cv2.resize(image_color, (width*2, height*2))
plt.imshow(resized_image, cmap='gray')
plt.title("Resized Image")
plt.axis('off')
plt.show()
```

#### 9. Flip the cropped/resized image horizontally.
```
flipped_image = cv2.flip(resized_image, 1)
plt.imshow(flipped_image, cmap='gray')
plt.title("Flipped Image")
plt.axis('off')
plt.show()
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```
apollo_img = cv2.imread(r"C:\Users\Trisha Priyadarshni\Downloads\Apollo-11-launch.jpg")
apollo_img = cv2.cvtColor(apollo_img, cv2.COLOR_BGR2RGB)
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image): text = 'Apollo 11 Saturn V Launch, July 16, 1969'font_face = cv2.FONT_HERSHEY_PLAIN
```
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
cv2.putText(apollo_img, text, (50, apollo_img.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```
cv2.rectangle(apollo_img, (100, 100), (400, 700), (255, 0, 255), 2)
```

#### 13. Display the final annotated image.
```
plt.imshow(cv2.cvtColor(apollo_img, cv2.COLOR_BGR2RGB))
plt.title("Annotated Apollo 11 Image")
plt.axis('off')
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```
boy_img = cv2.imread(r"C:\Users\Trisha Priyadarshni\Downloads\boy.jpg")
boy_img = cv2.cvtColor(boy_img, cv2.COLOR_BGR2RGB)
```

#### 15. Adjust the brightness of the image.
# Create a matrix of ones (with data type float64)
# matrix_ones = 
```
matrix_ones = np.ones(boy_img.shape, dtype='uint8') * 50
```

#### 16. Create brighter and darker images.
```
img_brighter = cv2.add(boy_img, matrix_ones)
img_darker = cv2.subtract(boy_img, matrix_ones)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(boy_img, cv2.COLOR_BGR2RGB)); plt.title("Original Image")
plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(img_darker, cv2.COLOR_BGR2RGB)); plt.title("Darker Image")
plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(img_brighter, cv2.COLOR_BGR2RGB)); plt.title("Brighter Image")
plt.show()
```

#### 18. Modify the image contrast.
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
matrix1 = 
matrix2 = 
# img_higher1 = 
# img_higher2 = 
```
matrix1 = np.ones(boy_img.shape, dtype='uint8') * 30
matrix2 = np.ones(boy_img.shape, dtype='uint8') * 60
img_higher1 = cv2.addWeighted(boy_img, 1.1, matrix1, 0, 0)
img_higher2 = cv2.addWeighted(boy_img, 1.2, matrix2, 0, 0)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(boy_img, cv2.COLOR_BGR2RGB)); plt.title("Original Image")
plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(img_higher1, cv2.COLOR_BGR2RGB)); plt.title("Higher Contrast 1")
plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(img_higher2, cv2.COLOR_BGR2RGB)); plt.title("Higher Contrast 2")
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```
b, g, r = cv2.split(boy_img)
```

#### 21. Merged the R, G, B , displays along with the original image
```
hsv_img = cv2.cvtColor(boy_img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(b, cmap='gray'); plt.title("Blue Channel")
plt.subplot(1, 3, 2); plt.imshow(g, cmap='gray'); plt.title("Green Channel")
plt.subplot(1, 3, 3); plt.imshow(r, cmap='gray'); plt.title("Red Channel")
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```
hsv_img = cv2.cvtColor(boy_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(h, cmap='gray'); plt.title("Hue Channel")
plt.subplot(1, 3, 2); plt.imshow(s, cmap='gray'); plt.title("Saturation Channel")
plt.subplot(1, 3, 3); plt.imshow(v, cmap='gray'); plt.title("Value Channel")
plt.show()

```
#### 23. Merged the H, S, V, displays along with original image.
```
merged_hsv = cv2.merge([h, s, v])
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(boy_img, cv2.COLOR_BGR2RGB)); plt.title("Original Image")
plt.subplot(1, 4, 2); plt.imshow(h, cmap='gray'); plt.title("Hue")
plt.subplot(1, 4, 3); plt.imshow(s, cmap='gray'); plt.title("Saturation")
plt.subplot(1, 4, 4); plt.imshow(v, cmap='gray'); plt.title("Value")
plt.show()


```

## Output:
- **i)** Read and Display an Image.
- ![Screenshot 2025-03-11 223915](https://github.com/user-attachments/assets/b1e3769e-4903-4e92-a5f1-41a10e34491e)
![Screenshot 2025-03-11 223922](https://github.com/user-attachments/assets/134bfb94-cb67-4e59-819e-73948a53a97d)
![Screenshot 2025-03-11 223929](https://github.com/user-attachments/assets/4647a2ea-d9b4-4cb9-baae-301754a87f84)
![Screenshot 2025-03-11 223935](https://github.com/user-attachments/assets/e290daa1-6723-44b0-b689-dc5914f9234c)
![Screenshot 2025-03-11 223942](https://github.com/user-attachments/assets/abd6d91f-2cb5-40ce-a58f-a23cf5821bc7)
![Screenshot 2025-03-11 223955](https://github.com/user-attachments/assets/e8f37375-36af-4cd8-ae5e-b843540dbd59)





-  
- **ii)** Adjust Image Brightness.
- ![Screenshot 2025-03-11 224011](https://github.com/user-attachments/assets/8d67d1f0-89f9-4039-85d9-e83cf579af75)

- 
- **iii)** Modify Image Contrast.
- ![Screenshot 2025-03-11 224032](https://github.com/user-attachments/assets/663eede7-bd8d-4ac4-9dd0-8d05239f7919)

- 
- **iv)** Generate Third Image Using Bitwise Operations.
- ![Screenshot 2025-03-11 224041](https://github.com/user-attachments/assets/e5d87678-329a-4cbc-9a1d-87c446f4d914)
- ![Screenshot 2025-03-11 224051](https://github.com/user-attachments/assets/22c312c9-ebc2-497f-bfa8-ff286f2093b2)
![Screenshot 2025-03-11 224103](https://github.com/user-attachments/assets/fdb267a0-fbf0-419f-8c5e-87d54bab4fdb)


- 

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

