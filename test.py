import torch
from model import *
from config import *
import torchvision
from torch.utils.data import DataLoader
from data_set import *
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    device = torch.device(DEVICE)
    if DEVICE == "cpu":
        class_weights=torch.FloatTensor(class_weight).cpu()
    else:
        class_weights=torch.FloatTensor(class_weight).cuda()

    img_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) # convert image into pytorch tensor
    test_dataset = DataLoader(tvtDatasetList(file_path=TEST_PATH, transforms=img_to_tensor),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    model = STRNN().to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    trained_weights = torch.load(TRAINED_WEIGHTS_PATH)
    # model.load_state_dict(trained_weights)
    model_dict = model.state_dict()
    trained_weights_dict = {k: v for k, v in trained_weights.items() if (k in model_dict)}
    model_dict.update(trained_weights_dict)
    model.load_state_dict(model_dict)

    # Output Result on test set
    model.eval()
    
    count = 0
    
    with torch.no_grad():
        for mini_batch in test_dataset:
            count += 1
            images = mini_batch['data'].to(device)
            truth = mini_batch['label'].type(torch.LongTensor).to(device)
            output = model(images)
            # print(output)
            pred = output.max(1, keepdim=True)[1]
            # print(pred.size())
            pred_ = torch.unbind(pred, dim=0)
            images_ = torch.unbind(images, dim=0)
            img_ = []
            lab_ = []
            for j in range(len(pred_)):
                pred_img = torch.squeeze(pred_[j]).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
                pred_img = Image.fromarray(pred_img.astype(np.uint8))
                truth_image = torch.squeeze(images_[j]).cpu().numpy()
                truth_image = np.transpose(truth_image[-1], [1, 2, 0]) * 255
                truth_image = Image.fromarray(truth_image.astype(np.uint8))
                rows = pred_img.size[0]
                cols = pred_img.size[1]
                for i in range(0, rows):
                    for j in range(0, cols):
                        pred_img2 = (pred_img.getpixel((i, j)))
                        if (pred_img2[0] > 200 or pred_img2[1] > 200 or pred_img2[2] > 200):
                            truth_image.putpixel((i, j), (234, 53, 57, 255))
                truth_image = truth_image.convert("RGB")
                truth_image.save(SAVE_PATH_TEST_1 + "%s_data.jpg" % (BATCH_SIZE*count+j))#red line on the original image
                pred_img.save(SAVE_PATH_TEST_2 + "%s_pred.jpg" % (BATCH_SIZE*count+j))#prediction result
                count += 1


    # Model Evaluation
    loss = 0
    pixels = 0
    precision = 0.0
    recall = 0.0
    error = 0
    with torch.no_grad():
        for mini_batch in test_dataset:
            count += 1
            images = mini_batch['data'].to(device)
            truth = mini_batch['label'].type(torch.LongTensor).to(device)
            output = model(images)
            pred = output.max(1, keepdim=True)[1]
            pred_ = torch.unbind(pred, dim=0)
            truth_ = torch.unbind(truth, dim=0)
            img_ = []
            lab_ = []
            kernel = np.uint8(np.ones((3, 3)))
            for j in range(BATCH_SIZE):
                img = torch.squeeze(pred[j]).cpu().numpy()*255
                lab = torch.squeeze(truth[j]).cpu().numpy()*255
                img = img.astype(np.uint8)
                lab = lab.astype(np.uint8)
                # img = torch.unsqueeze(img, dim=0)
                # lab = torch.unsqueeze(lab, dim=0)
                label_precision = cv2.dilate(lab, kernel)
                pred_recall = cv2.dilate(img, kernel)
                img = img.astype(np.int32)
                lab = lab.astype(np.int32)
                label_precision = label_precision.astype(np.int32)
                pred_recall = pred_recall.astype(np.int32)
                a = len(np.nonzero(img*label_precision)[1])
                b = len(np.nonzero(img)[1])
                if b==0:
                    error=error+1
                    continue
                else:
                    precision += float(a/b)
                c = len(np.nonzero(pred_recall*lab)[1])
                d = len(np.nonzero(lab)[1])
                if d==0:
                    error = error + 1
                    continue
                else:
                    recall += float(c / d)
                F1_measure=(2*precision*recall)/(precision+recall)   
                

            #accuracy
            loss += loss_function(output, truth).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            pixels += pred.eq(truth.view_as(pred)).sum().item()

            #precision,recall,f1
            # img = torch.cat(img_)
            # lab = torch.cat(lab_)
            # label_precision = cv2.dilate(lab, kernel)
            # pred_recall = cv2.dilate(img, kernel)
            # img = img.astype(np.int32)
            # lab = lab.astype(np.int32)
            # label_precision = label_precision.astype(np.int32)
            # pred_recall = pred_recall.astype(np.int32)
            # a = len(np.nonzero(img*label_precision)[1])
            # b = len(np.nonzero(img)[1])
            # if b==0:
            #     error=error+1
            #     continue
            # else:
            #     precision += float(a/b)
            # c = len(np.nonzero(pred_recall*lab)[1])
            # d = len(np.nonzero(lab)[1])
            # if d==0:
            #     error = error + 1
            #     continue
            # else:
            #     recall += float(c / d)
            # F1_measure=(2*precision*recall)/(precision+recall)    
    
    
    loss /= count
    accuracy = 100. * int(pixels) / (count * BATCH_SIZE * 128 * 256)
    print("Loss = {}".format(loss))
    print("Accuracy = {}".format(accuracy))
    

    precision = precision / (count * BATCH_SIZE - error)
    recall = recall / (count * BATCH_SIZE - error)
    F1_measure = F1_measure / (count * BATCH_SIZE - error)

    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1_measure = {}".format(F1_measure))