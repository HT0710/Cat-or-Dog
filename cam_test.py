import torch
import cv2
from model import VGG13
from torchvision import transforms as T


# Define
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['cat', 'dog']
TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# Camera setting
VIDEO_SIZE = (960, 540)
FULLSCREEN = False


# Define model
model = VGG13().to(DEVICE).eval()
model.load_state_dict(torch.load('Models/E12_L0.0873_A0.97.pth'))

# Video capture
vid = cv2.VideoCapture(0)

while(True):   
    ret, frame = vid.read()

    # Prediction
    input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input = TRANSFORM(input).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        outputs = model(input)
        _, pred = torch.max(outputs, 1)

    # Video size
    frame = cv2.resize(frame, VIDEO_SIZE)

    # FPS and label
    cv2.putText(frame, str(vid.get(cv2.CAP_PROP_FPS)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, CLASSES[pred], (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)

    if FULLSCREEN:
        cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
