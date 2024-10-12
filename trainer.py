import argparse
import logging
import os
from pickle import FALSE
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from osgeo import gdal
from sklearn.metrics import confusion_matrix

def trainer_data(args, model, moudelName):
    model=model.cuda()
    from datasets import  dataset
    logging.basicConfig(filename=args.output_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = dataset(root=args.root_path, type="train")
    db_test=dataset(root=args.test_path,type="test")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn,drop_last=False)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn,drop_last=False)                        
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer =optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)#torch.optim.Adam(model.parameters(), lr=0.000001, betas=(0.9, 0.999)) #optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(args.output_dir + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    #lr_scheduler = create_lr_scheduler(optimizer, len(trainloader), args.max_epochs, warmup=True)
    bestoverall_accuracy=0
    for epoch_num in iterator:
        model.train()
        if(epoch_num >4):
            optimizer =optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        for i,(image, label)  in enumerate(trainloader):

            image_batch, label_batch = image.cuda().float(), label.cuda()
            outputs = model(image_batch)
            # n, c, h, w = outputs.shape


            # outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            # label_batch = label_batch.view(-1)
            label_batch=label_batch.long().squeeze()
            loss_ce = ce_loss(outputs, label_batch[:])
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            #totallr=totallr+lr_
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1


            writer.add_scalar('info/total_loss',loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #torch.cuda.empty_cache()
        save_mode_path = os.path.join(args.output_dir,str(epoch_num)+ moudelName)
        torch.save(model.state_dict(), save_mode_path)
        overall_accuracy,f1,miou,conf_matrix_percentage=testmodule(testloader,model)
        logging.info('Overall Accuracy (OA): %f, Average F1 Score (F1): %f, Mean Intersection over Union (mIoU): %f, Confusion Matrix (Percentage):\n%s' % 
             (overall_accuracy, f1, miou, conf_matrix_percentage))

        #logging.info('Overall Accuracy (OA): %f, Average F1 Score (F1) : %f, Mean Intersection over Union (mIoU): %f, matrix_percentage: %f' % (overall_accuracy, f1, miou,conf_matrix_percentage))
        # 打印评估结果
        print(f"Overall Accuracy (OA): {overall_accuracy:.4f}")
        print(f"Average F1 Score (F1): {f1:.4f}")
        print(f"Mean Intersection over Union (mIoU): {miou:.4f}")       
        if(overall_accuracy>=bestoverall_accuracy):
            bestoverall_accuracy=overall_accuracy
            save_mode_path=os.path.join(args.output_dir,"bestMoudle.pth")
            torch.save(model.state_dict(), save_mode_path)
    writer.close()
    
    return "Training Finished!"

def testmodule(dataloader, model):
    model.eval()  # Set the model to evaluation mode
    all_labels = []  # List to store all ground truth labels
    all_predictions = []  # List to store all predicted labels
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for i, (image, label) in enumerate(dataloader):
            image_batch, label_batch = image, label  # Extract image and label batches
            image_batch = image_batch.cuda().float()  # Move image batch to GPU and convert to float
            label_batch = label_batch.cuda()  # Move label batch to GPU
            
            outputs = model(image_batch)  # Forward pass through the model
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # Get predicted class labels
            
            all_labels.append(label_batch.cpu().numpy())  # Store ground truth labels
            all_predictions.append(outputs.cpu().numpy())  # Store predicted labels
        
        # Concatenate all labels and predictions after the loop
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        # Remove dimensions of size 1
        all_labels = all_labels.squeeze()
        all_predictions = all_predictions.squeeze()

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(all_labels.flatten(), all_predictions.flatten())

        # Convert the confusion matrix to percentages
        conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum() * 100

        print("Confusion Matrix (Percentage):")
        print(conf_matrix_percentage)

        # Calculate Overall Accuracy (OA)
        overall_accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())

        # Calculate Average F1 Score (F1)
        f1 = f1_score(all_labels.flatten(), all_predictions.flatten(), average='macro')
        
        # Calculate Mean Intersection over Union (mIoU)
        miou = compute_iou(all_predictions, all_labels, num_classes=2)

        return overall_accuracy, f1, miou, conf_matrix_percentage  # Return evaluation metrics

 
def predict_synapse(args, model, weights_path):
    
    from datasets import dataset, RandomGenerator

    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = dataset(root=args.root_path, type='predict')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,drop_last=False,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict)
    
    model.eval()

    iter_num = 0
    max_epoch = args.max_epochs
    best_performance = 0.0
    with torch.no_grad():
        for i,(image,image_filenames) in enumerate(trainloader):
            image_batch = image
            image_batch = image_batch.cuda().float()
            outputs = model(image_batch)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            # n, c, h, w = outputs.shape

            # outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            # label_batch = label_batch.view(-1)
            #label_batch=label_batch.long().squeeze()
            shape=outputs.shape
            
            max_value = outputs.max()  
        
            # 打印最大值  
            print(max_value)
            iter_num = iter_num + 1
            # color
            # colors = [(255, 0, 0), (255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0)]

            # 
            for i, (image_batch, image_filenames) in enumerate(trainloader):
                    image_batch = image_batch.cuda().float()
                    outputs = model(image_batch)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    shape = outputs.shape
                    
                    max_value = outputs.max()
                    print(max_value)
                    iter_num += 1
                    
                    #colors = [(255, 0, 0), (255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0)]

                    for idx, (output, filename) in enumerate(zip(outputs, image_filenames)):
                        masked_image = np.zeros((224, 224, 3), dtype=np.uint8)
                        value = output.squeeze().cpu().numpy()

                        alpha = 0.5
                        red_color = np.array([255, 0, 0])

                        for c in range(3):
                            masked_image[:, :, c] = np.where(value == 1,
                                                            masked_image[:, :, c] * (1 - alpha) + alpha * red_color[c],
                                                            masked_image[:, :, c])

                        image = Image.fromarray(masked_image)
                        filename_without_extension = os.path.splitext(filename)[0]
                        image.save(f"./output_{filename_without_extension}.png")

    

def evaluate(args, model, weights_path):
    from datasets import dataset, RandomGenerator
    import random
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

    # Define the dataset and data loader
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = dataset(root=args.test_path, type='test')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False)

    # Use DataParallel if multiple GPUs are available
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # Load model weights
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict)
    model.eval()

    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for i, (image, label) in enumerate(trainloader):
            image_batch = image.float().cuda()
            label = label.cuda()

            # Forward pass
            outputs = model(image_batch)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            all_labels.append(label.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

        # Convert labels and predictions to numpy arrays
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(all_labels.flatten(), all_predictions.flatten())

        # Convert to percentages
        conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum() * 100

        print("Confusion Matrix (Percentage):")
        print(conf_matrix_percentage)

        # Remove dimensions of size 1
        all_labels = all_labels.squeeze()
        all_predictions = all_predictions.squeeze()

        # Calculate Overall Accuracy (OA)
        overall_accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())

        # Calculate Average F1 Score (F1)
        f1 = f1_score(all_labels.flatten(), all_predictions.flatten(), average='macro')
        
        # Assuming compute_iou function is defined elsewhere
        miou = compute_iou(all_predictions, all_labels, num_classes)

        # Print evaluation results
        print(f"Overall Accuracy (OA): {overall_accuracy:.4f}")
        print(f"Average F1 Score (F1): {f1:.4f}")
        print(f"Mean Intersection over Union (mIoU): {miou:.4f}")


# calculate（mIoU）
def compute_iou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  
        else:
            ious.append(float(intersection) / float(union))
    return np.nanmean(ious)  


def predict_totif(args, model, weights_path,outputpath):
    from datasets import dataset
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = dataset(root=args.predict_data_path, type='predict')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict)
    model.eval()

    iter_num = 0
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    for i, (image_batch, image_filenames, geotransform, projection) in enumerate(trainloader):
        image_batch = image_batch.cuda().float()
        outputs = model(image_batch)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

        max_value = outputs.max()
        print(max_value)
        iter_num += 1
        
        for idx, (output, filename, geo, proj) in enumerate(zip(outputs, image_filenames, geotransform, projection)):
            value = output.squeeze().cpu().numpy()

            # Create a new GeoTIFF file
            output_file =outputpath+ f"/output_{os.path.splitext(filename)[0]}.tif"
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(output_file, value.shape[1], value.shape[0], 1, gdal.GDT_Byte)
            dst_ds.SetGeoTransform(list(geo))
            dst_ds.SetProjection(proj)
            dst_ds.GetRasterBand(1).WriteArray(value)
            dst_ds.FlushCache()
            dst_ds = None
            
            print(f'Saved: {output_file}')

    print(f'All predictions completed.')