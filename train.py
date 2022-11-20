import torch
from model import *
from config import *
import torchvision
from torch.utils.data import DataLoader
from data_set import *
import numpy as np
import cv2

def evaluate_model(mini_batch_list):
    print("###########Evaluation##################")
    count = 0
    loss = 0
    pixels = 0
    precision = 0.0
    recall = 0.0
    error = 0
    with torch.no_grad():
        for mini_batch in mini_batch_list:
            count += 1
            images = mini_batch['data'].to(device)
            truth = mini_batch['label'].type(torch.LongTensor).to(device)
            output = model(images)
            pred = output.max(1, keepdim=True)[1]
            pred_ = torch.unbind(pred, dim=0)
            truth_ = torch.unbind(truth, dim=0)
            kernel = np.uint8(np.ones((3, 3)))
            for j in range(len(pred_)):
                img = torch.squeeze(pred_[j]).cpu().numpy() * 255
                lab = torch.squeeze(truth_[j]).cpu().numpy() * 255
                img = img.astype(np.uint8)
                lab = lab.astype(np.uint8)
                label_precision = cv2.dilate(lab, kernel)
                pred_recall = cv2.dilate(img, kernel)
                img = img.astype(np.int32)
                lab = lab.astype(np.int32)
                label_precision = label_precision.astype(np.int32)
                pred_recall = pred_recall.astype(np.int32)
                a = len(np.nonzero(img * label_precision)[1])
                b = len(np.nonzero(img)[1])
                if b == 0:
                    error = error + 1
                    continue
                else:
                    precision += float(a / b)
                c = len(np.nonzero(pred_recall * lab)[1])
                d = len(np.nonzero(lab)[1])
                if d == 0:
                    error = error + 1
                    continue
                else:
                    recall += float(c / d)
                F1_measure = (2 * precision * recall) / (precision + recall)

                # accuracy
            loss += loss_function(output, truth).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            pixels += pred.eq(truth.view_as(pred)).sum().item()

    loss /= count
    accuracy = 100. * int(pixels) / (count * TEST_BATCH_SIZE * 128 * 256)
    print("Loss = {}".format(loss))
    print("Accuracy = {}".format(accuracy))
    if (not(count * TEST_BATCH_SIZE - error == 0)):
        precision = precision / (count * TEST_BATCH_SIZE - error)
        recall = recall / (count * TEST_BATCH_SIZE - error)
        F1_measure = F1_measure / (count * TEST_BATCH_SIZE - error)
        print("Precision = {}".format(precision))
        print("Recall = {}".format(recall))
        print("F1_measure = {}".format(F1_measure))
        print("######################################")
    else:
        print("Gradient Exploding")




if __name__ == '__main__':
    
    device = torch.device(DEVICE)
    if DEVICE == "cpu":
        class_weights=torch.FloatTensor(class_weight).cpu()
    else:
        class_weights=torch.FloatTensor(class_weight).cuda()

    img_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) # convert image into pytorch tensor
    train_dataset = DataLoader(tvtDatasetList(file_path=TRAIN_PATH, transforms=img_to_tensor),batch_size=TRAIN_BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    validation_dataset = DataLoader(tvtDatasetList(file_path=VALIDATION_PATH, transforms=img_to_tensor), batch_size=TRAIN_BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    model = STRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)   
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    for epoch in range(EPOCH):
        print("#####################TRAINING#####################")
        print("--------------------------------------------------")

        ## Training
        model.train()
        mini_batch_list = []
        for i, mini_batch in enumerate(train_dataset):
            mini_batch_list.append(mini_batch)
            torch.autograd.set_detect_anomaly(True)
            images = mini_batch['data'].to(device)
            truth = mini_batch['label'].type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, truth)
            pred = output.max(1, keepdim=True)[1]
            # print(pred.size())
            pred = torch.unbind(pred, dim=0)
            for j in range(TRAIN_BATCH_SIZE):
                img = torch.squeeze(pred[j]).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
                img = Image.fromarray(img.astype(np.uint8))
                img.save(SAVE_PATH + "%s_pred.jpg" % (i*TRAIN_BATCH_SIZE+j))
            loss.backward()
            optimizer.step()
            print("Train Epoch = {}  Batch Index = {}".format(epoch, i))
            print("Loss = {}".format(loss))
            print("------------------------------------------")
            if (i%TRAIN_EVAL_BATCH == 0):
                evaluate_model(mini_batch_list)
                mini_batch_list.clear()

        print("######################VALIDATION######################")
        ## Validation
        model.eval()
        loss = 0
        count = 0
        pixels = 0
        with torch.no_grad():
            for mini_batch in validation_dataset:
                count += 1
                images = mini_batch['data'].to(device)
                truth = mini_batch['label'].type(torch.LongTensor).to(device)
                output = model(images)
                pred = output.max(1, keepdim=True)[1]
                pred = torch.unbind(pred, dim=0)
                for j in range(TRAIN_BATCH_SIZE):
                    img = torch.squeeze(pred[j]).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
                    img = Image.fromarray(img.astype(np.uint8))
                    img.save(SAVE_PATH_VAL + "%s_pred.jpg" % (epoch * TRAIN_BATCH_SIZE + j))

                loss += loss_function(output, truth).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]
                pixels += pred.eq(truth.view_as(pred)).sum().item()
        loss /= count
        accuracy = 100. * int(pixels) / (count * TRAIN_BATCH_SIZE * 128 * 256)
        print("Loss = {}".format(loss))
        print("Accuracy = {}".format(accuracy))
        torch.save(model.state_dict(), '%s.pth'%accuracy)
        scheduler.step()

        ##TESTING##
    print("#############################TESTING###############################")
    test_dataset = DataLoader(tvtDatasetList(file_path=TEST_PATH, transforms=img_to_tensor), batch_size=TEST_BATCH_SIZE,\
                              shuffle=True, num_workers=NUM_WORKERS)

    model.eval()

    count = 0

    with torch.no_grad():
        for mini_batch in test_dataset:
            count += 1
            print("Printing Image {}".format(count))
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
                pred_img = torch.squeeze(pred_[j]).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
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
                truth_image.save(
                    SAVE_PATH_TEST_1 + "%s_data.jpg" % (TEST_BATCH_SIZE * count + j))  # red line on the original image
                pred_img.save(SAVE_PATH_TEST_2 + "%s_pred.jpg" % (TEST_BATCH_SIZE * count + j))  # prediction result

    # Model Evaluation
    loss = 0
    pixels = 0
    precision = 0.0
    recall = 0.0
    error = 0
    count = 0
    with torch.no_grad():
        for mini_batch in test_dataset:
            count += 1
            print("Evaluate Iteration {}".format(count))
            images = mini_batch['data'].to(device)
            truth = mini_batch['label'].type(torch.LongTensor).to(device)
            output = model(images)
            pred = output.max(1, keepdim=True)[1]
            pred_ = torch.unbind(pred, dim=0)
            truth_ = torch.unbind(truth, dim=0)
            img_ = []
            lab_ = []
            kernel = np.uint8(np.ones((3, 3)))
            for j in range(TEST_BATCH_SIZE):
                img = torch.squeeze(pred[j]).cpu().numpy() * 255
                lab = torch.squeeze(truth[j]).cpu().numpy() * 255
                img = img.astype(np.uint8)
                lab = lab.astype(np.uint8)
                label_precision = cv2.dilate(lab, kernel)
                pred_recall = cv2.dilate(img, kernel)
                img = img.astype(np.int32)
                lab = lab.astype(np.int32)
                label_precision = label_precision.astype(np.int32)
                pred_recall = pred_recall.astype(np.int32)
                a = len(np.nonzero(img * label_precision)[1])
                b = len(np.nonzero(img)[1])
                if b == 0:
                    error = error + 1
                    continue
                else:
                    precision += float(a / b)
                c = len(np.nonzero(pred_recall * lab)[1])
                d = len(np.nonzero(lab)[1])
                if d == 0:
                    error = error + 1
                    continue
                else:
                    recall += float(c / d)
                F1_measure = (2 * precision * recall) / (precision + recall)

                # accuracy
            loss += loss_function(output, truth).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            pixels += pred.eq(truth.view_as(pred)).sum().item()

    loss /= count
    accuracy = 100. * int(pixels) / (count * TEST_BATCH_SIZE * 128 * 256)
    print("Loss = {}".format(loss))
    print("Accuracy = {}".format(accuracy))
    if (not(count * TEST_BATCH_SIZE - error == 0)):
        precision = precision / (count * TEST_BATCH_SIZE - error)
        recall = recall / (count * TEST_BATCH_SIZE - error)
        F1_measure = F1_measure / (count * TEST_BATCH_SIZE - error)

        print("Precision = {}".format(precision))
        print("Recall = {}".format(recall))
        print("F1_measure = {}".format(F1_measure))

