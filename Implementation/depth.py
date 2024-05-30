import cv2
import torch

# Load Model
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas.to('cuda')
midas.eval()

# Transformational Pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()

    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda')

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        # print(prediction)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()
        # print(output)
    
    output = (output - output.min()) / (output.max() - output.min()) * 255
    output = output.astype('uint8')
    
    cv2.imshow('Depth Map', output)
    cv2.imshow('CV2Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()