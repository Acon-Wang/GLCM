import utils_torch
import torch
import os
import shutil
import time
from torch.autograd import Variable
import torch.nn as nn
import sys
import classifier

def accuracy(outputs, labels):
    """
    Calculate accuracy with the output of network and real label
    :param outputs: the output of the network
    :param labels: the real label of the data
    :return:
    """
    prediction = torch.round(outputs)
    # one_hot_prediction = convert_one_hot(argmax.view(-1, 1).float())

    return (labels == prediction).float().mean()


def get_path(operating_system):
    """
    Return data path and number of workers(used in data loader) depends on operating system
    :param operating_system: windows or linux
    :return: train_cover_dir, train_stego_dir, test_cover_dir, test_stego_dir, num_workers
    """
    # windows path
    win_train_cover_dir = '..\\..\\paixubianma\\MIX\\0.1\\train\\cover'
    win_train_stego_dir = '..\\..\\paixubianma\\MIX\\0.1\\train\\stego0.1'
    win_test_cover_dir = '..\\..\\paixubianma\\MIX\\0.1\\test\\cover'
    win_test_stego_dir = '..\\..\\paixubianma\\MIX\\0.1\\test\\stego0.1'
    win_num_workers = 1
    #freq_CNV_divide
    # linux path
    linux_train_cover_dir = 'H:\\dataset\\freq_CNV_divide\\0.0\\Chinese\\train'
    linux_train_stego_dir = 'H:\\dataset\\freq_CNV_divide\\1.0\\Chinese\\train'
    linux_valid_cover_dir = 'H:\\dataset\\freq_CNV_divide\\0.0\\Chinese\\valid'
    linux_valid_stego_dir = 'H:\\dataset\\freq_CNV_divide\\1.0\\Chinese\\valid'
    linux_test_cover_dir = 'H:\\dataset\\freq_NPP_divide\\0.0\\Chinese\\test'
    linux_test_stego_dir = 'H:\\dataset\\freq_NPP_divide\\1.0\\Chinese\\test'
    linux_num_workers = 8

    if operating_system == 'windows':
        return win_train_cover_dir, win_train_stego_dir, \
               win_test_cover_dir, win_test_stego_dir, win_num_workers
    else:
        return linux_train_cover_dir, linux_train_stego_dir, \
               linux_valid_cover_dir, linux_valid_stego_dir, \
               linux_test_cover_dir, linux_test_stego_dir, linux_num_workers


def get_net(dir):
    """
    :return: the best net until now
    """
    stored_dir = dir
    best_model = os.listdir(stored_dir)[0]
    state_dict = torch.load(stored_dir+best_model)
    # net.load_state_dict(torch.load(stored_dir+best_model))
    return state_dict


def save_best(state, epoch, dir):
    """
    remove the model_best dir if exist and store the current best epoch
    :param state: current model state
    :param epoch: current best epoch number
    :param dir: model stored dir
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)   #删除原有文件重新建立文件夹
        os.mkdir(dir)

    filename = dir + 'model_best' + str(epoch) + '.pth'


    torch.save(state, filename)


def single_train(train_loader, f, epoch, net, loss_func, optimizer):
    """
    Training the network using all the data, feed the data into previous network if current FAE is not
    the first block in the architecture
    """
    net.train()   #net.train 和net.eval 切换模式
    train_loss = 0.     #定义loss和accuracy
    train_accuracy = 0.
    start = time.time()  #记录时间
    for batch_idx, data in enumerate(train_loader):
        # sys.stdout.write('\rCurrent batch index: %s' % batch_idx)
        # sys.stdout.flush()
        '''
        # full data
        train_images, train_labels = Variable(data['images'].view(self.batch_size, -1).float()), \
                         Variable(data['labels'].view(self.batch_size, 1))
        train_random_index, train_images_shuffled, train_labels_shuffled \
            = shuffle_data(train_images, train_labels)
        X = train_images_shuffled.cuda()            
        '''

        X, T = Variable(data['images'].cuda()).float(), Variable(data['labels'].cuda()).float()#数据加入GPU
        #from torch.autograd import Variable 存储数据和grad
        output = net(X)
        
        optimizer.zero_grad()  #梯度清零
        current_loss = loss_func(output, T)  #网络输出与标签求损失
        current_loss.backward()     #误差逆传播
        optimizer.step()        #更新 这几步是常用操作
        current_accuracy = accuracy(output, T)  #计算当前accuracy

        train_loss += current_loss.item()
        train_accuracy += current_accuracy.item()

        if ((batch_idx + 1) % 100) == 0:
            train_loss /= 100             #100个batch打印一次结果，随意定
            train_accuracy /= 100
            print('\nTrain epoch: ' + str(epoch) + ', ' + str(batch_idx+1) + '/' + str(len(train_loader)))
            print('Current_batch_loss: ' + str(train_loss))
            print('Current_batch_accuracy: ' + str(train_accuracy))
            f.write('\nTrain epoch: ' + str(epoch) + ', ' +
                         str(batch_idx+1) + '/' + str(len(train_loader)) + '\n')
            f.write('Current_batch_loss: ' + str(train_loss) + '\n')
            f.write('Current_batch_accuracy: ' + str(train_accuracy) + '\n')
            f.flush() #将缓冲区中的数据立刻写入文件，同时清空缓冲区
            train_loss = 0.
            train_accuracy = 0.
            end = time.time()
    # print('\nCurrent epoch training_time: ' + str(end - start))
    #f.write('\nCurrent epoch training_time: ' + str(end - start))

def single_valid(test_loader, f, epoch, net, loss_func):
    """
    Training the network using all the data, feed the data into previous network if current FAE is not
    the first block in the architecture
    """
    net.eval()
    test_loss = 0.
    test_accuracy = 0.
    start = time.time()
    for batch_idx, data in enumerate(test_loader):
        # sys.stdout.write('\rCurrent batch index: %s' % batch_idx)
        # sys.stdout.flush()
        '''
        # full data
        train_images, train_labels = Variable(data['images'].view(self.batch_size, -1).float()), \
                         Variable(data['labels'].view(self.batch_size, 1))
        train_random_index, train_images_shuffled, train_labels_shuffled \
            = shuffle_data(train_images, train_labels)
        X = train_images_shuffled.cuda()            
        '''
        # last three dim, used for CNV
        X, T = Variable(data['images'].cuda()).float(), Variable(data['labels'].cuda()).float()#数据加入GPU


        output = net(X)

        current_loss = loss_func(output, T)
        current_accuracy = accuracy(output, T)

        test_loss += current_loss.item()
        test_accuracy += current_accuracy.item()

    test_accuracy /= len(test_loader)
    test_loss /= len(test_loader)
    end = time.time()
    print('\nEpoch: ' + str(epoch) + ', valid loss: ' + str(test_loss))
    f.write('\nEpoch: ' + str(epoch) + ', valid loss: ' + str(test_loss))
    print('Epoch: ' + str(epoch) + ', valid acc: ' + str(test_accuracy))
    f.write('\nEpoch: ' + str(epoch) + ', valid acc: ' + str(test_accuracy) + '\n')
    print('Epoch: ' + str(epoch) + ', valid time: ' + str(end - start))
    f.write('Epoch: ' + str(epoch) + ', valid time: ' + str(end - start) + '\n')

    return test_accuracy

def single_eval(test_loader, f, epoch, net, loss_func):
    """
    Training the network using all the data, feed the data into previous network if current FAE is not
    the first block in the architecture
    """
    net.eval()
    test_loss = 0.
    test_accuracy = 0.
    start = time.time()
    for batch_idx, data in enumerate(test_loader):
        # sys.stdout.write('\rCurrent batch index: %s' % batch_idx)
        # sys.stdout.flush()
        '''
        # full data
        train_images, train_labels = Variable(data['images'].view(self.batch_size, -1).float()), \
                         Variable(data['labels'].view(self.batch_size, 1))
        train_random_index, train_images_shuffled, train_labels_shuffled \
            = shuffle_data(train_images, train_labels)
        X = train_images_shuffled.cuda()            
        '''
        # last three dim, used for CNV
        X, T = Variable(data['images'].cuda()).float(), Variable(data['labels'].cuda()).float()#数据加入GPU

        output = net(X)

        current_loss = loss_func(output, T)
        current_accuracy = accuracy(output, T)

        test_loss += current_loss.item()
        test_accuracy += current_accuracy.item()

    test_accuracy /= len(test_loader)
    test_loss /= len(test_loader)
    end = time.time()
    print('\nEpoch: ' + str(epoch) + ', evaluating loss: ' + str(test_loss))
    f.write('\nEpoch: ' + str(epoch) + ', evaluating loss: ' + str(test_loss))
    print('Epoch: ' + str(epoch) + ', evaluating acc: ' + str(test_accuracy))
    f.write('\nEpoch: ' + str(epoch) + ', evaluating acc: ' + str(test_accuracy) + '\n')
    print('Epoch: ' + str(epoch) + ', evaluating time: ' + str(end - start))
    f.write('Epoch: ' + str(epoch) + ', evaluating time: ' + str(end - start) + '\n')

    return test_accuracy


def main():
    """
    This function runs a single fast auto-encoder network
    """
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  ###指定此处为-1即可
    if __name__ == '__main__':
        batch_size = 128
        record_path = '0.1_50.txt'
        dir_1 = './model_best/'
        operating_system = 'linux'
        f = open(record_path, 'w')

        train_cover_dir, train_stego_dir, valid_cover_dir, valid_stego_dir, \
        test_cover_dir, test_stego_dir, num_workers = get_path(operating_system)
        train_loader = utils_torch.read_data(train_cover_dir, train_stego_dir, shuffle=True,
                                             batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        valid_loader = utils_torch.read_data(valid_cover_dir, valid_stego_dir, shuffle=True,
                                             batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        test_loader = utils_torch.read_data(test_cover_dir, test_stego_dir, shuffle=True,
                                             batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        print(len(valid_loader))
        net = classifier.Classifier().cuda()
        loss_function = nn.BCELoss().cuda()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

        best_accuracy = 0.
        bbest = 0.
        for epoch in range(200):#训练200次
            print("start")
            single_train(train_loader, f, epoch,  net, loss_function, optimizer)
            current_accuracy = single_valid(valid_loader, f, epoch,  net, loss_function)
            # cur_acc = single_eval(test_loader, f, epoch, net, loss_function)
            if current_accuracy >= best_accuracy or current_accuracy >= bbest:
                best_accuracy = current_accuracy

                save_best(net.state_dict(), epoch, dir_1)

                cur_acc=single_eval(test_loader, f, epoch,  net, loss_function)

                if cur_acc>bbest:
                    bbest=cur_acc
                    d=str(bbest)
                    print("最高准确率：")
                    print(d)

                f.write('Best epoch: ' + str(epoch) + '\n')


main()
