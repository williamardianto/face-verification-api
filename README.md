# Face Verification API with SpringBoot and DL4J

This repo contain a simple implementation of face verification API service using Springboot and DL4J. Model used in this project is Keras VGGFace taken from this repo (https://github.com/rcmalli/keras-vggface)

Model weights can be download here: https://drive.google.com/drive/folders/1DdD7u951AiFwlyccIBfqcZl6MgIG13ld?usp=sharing

## Request

#### URI

```http
POST /verify
```

#### Request Body

Sample request (Method 1: Use a Base64-encoded image.)

```json
{
  "image1_base64":"/9j/4AAQSkZJRgABAgEASABIAAD",
  "image2_base64":"/9j/4AAQSkZJRgABAgEASABIAAD"
}
```

Sample request (Method 2: Use an image file.)

```
image1_file: File (image file)
image2_file: File (image file)
```

## Response

```json
{
    "distance": 0.5859787368774414
}
```
