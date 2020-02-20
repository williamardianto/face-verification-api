# Face Verification API with SpringBoot, DL4J, and Keras Weights

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

# Response

```json
{
    "distance": 0.5859787368774414
}
```