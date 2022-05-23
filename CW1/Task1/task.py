import torch
import torchvision
import torch.utils.data as Data
import time

# Python 3.9.7
#print(torch.__version__)
# torch 1.7.1.post2
#print(torchvision.__version__)
# torchvision 0.8.0a0

def polynomial_fun(w,x):
    # w: vector size (M+1)
    # x: scalar
    # y: scalar
    y = 0
    for m in range(len(w)):
        y += w[m]*(x**m)
    return y

def fit_polynomial_ls(x,t,M):
    # x: size N
    # t: target values, size N
    # M: degree
    m = torch.arange(M+1, dtype = torch.float32)
    y = torch.pow(x,m)
    w = torch.lstsq(t,y).solution
    return w[: M+1]

def fit_polynomial_sgd(x, t, M, lr, size):
    # x: size N
    # t: target values, size N
    # M: degree
    # lr: learning rate
    # size: batch size
    m = torch.arange(M+1, dtype = torch.float32)
    y = torch.pow(x,m) # forward

    ds_train = Data.DataLoader(Data.TensorDataset(y, t_training), batch_size = size, shuffle= True)

    model = torch.nn.Sequential(torch.nn.Linear(M+1,1))
    #model.weight.data.normal_(0, 0.1)
    model[0].weight.data.normal_(0, 0.1) # initial weight
    # model.apply(weigt_init)
    # m.weight.data.normal_(0, 0.02)
    # m.bias.data.zero_()
   
    loss_f = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    num_epochs = 40000
    
    for epoch in range(num_epochs): # num_epochs

        # train with batches of data
        for xb, yb in ds_train:
            pred = model(xb) # pred
            loss = loss_f(pred, yb) # loss
            loss.backward() # back propagation, compute grad
            opt.step() # update parameters
            opt.zero_grad() # reset the grad to zero
        if epoch % 400 == 399:
            print('epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


    return model[0].weight.data

w = torch.Tensor([1,2,3,4]).T # M = 3

# generate a training set and a test set
# U(0,1) *20 - 20 = U(0,20) - 20 = U(-20,20)

x_training = (40 * torch.rand(100) - 20).reshape(100,1)
y_training = polynomial_fun(w,x_training)
t_training = y_training + torch.normal(0,0.2, size=(100,1))

x_test = (40 * torch.rand(50) - 20).reshape(50,1)
y_test = polynomial_fun(w,x_test)
t_test = y_test + torch.normal(0,0.2, size=(50,1))

# compute the optimum weight vector using the training set
w_training_ls = fit_polynomial_ls(x_training,t_training,4)

# compute the predicted target values for both the training and test sets
y_pred_training_ls = polynomial_fun(w_training_ls,x_training)
y_pred_test_ls = polynomial_fun(w_training_ls,x_test)


print('The mean and standard deviation in difference between the observed training data and the underlying “true” polynomial curve are {:.4f} and {:.4f}'.format(torch.mean(t_training - y_training),torch.std(t_training - y_training)))
print('The mean and standard deviation in difference between the “LS-predicted” values and the underlying “true” polynomial curve are {:.4f} and {:.4f}'.format(torch.mean(y_pred_training_ls - y_training),torch.std(y_pred_training_ls - y_training)))

# compute the optimum weight vector using the training set
w_training_sgd = fit_polynomial_sgd(x_training, t_training, 4, 1e-10, 30)

# compute the predicted target values for both the training and test sets
y_pred_training_sgd = polynomial_fun(w_training_sgd,x_training)
y_pred_test_sgd = polynomial_fun(w_training_sgd,x_test)

print('The mean and standard deviation in difference between the “SGD-predicted” values and the underlying “true” polynomial curve are {:.4f} and {:.4f}'.format(torch.mean(y_pred_training_sgd - y_training),torch.std(y_pred_training_sgd - y_training)))

# Root Mean Squared Error
def RMSE(y_pred, y):
    rmse = torch.sqrt(torch.mean((y_pred-y)**2))
    return rmse

w = torch.Tensor([1,2,3,4,0]).T
print('The RMSEs of w for ls and sgd are {:.4f} and {:.4f}'.format(RMSE(w_training_ls, w), RMSE(w_training_sgd, w)))
print('The RMSEs of y for ls and sgd are {:.4f} and {:.4f}'.format(RMSE(y_pred_training_ls, y_training), RMSE(y_pred_training_sgd, y_training)))

time_start = time.time()
# compute the optimum weight vector using the test set
w_test_ls = fit_polynomial_ls(x_test,t_test,4)

# compute the predicted target values for both the training and test sets
y_pred_test_ls = polynomial_fun(w_test_ls,x_test)

time_end = time.time() 
time_ls= time_end - time_start
print('Time cost for ls is ', time_ls, 's.')

def fit_polynomial_sgd(x, t, M, lr, size):
    # x: size N
    # t: target values, size N
    # M: degree
    # lr: learning rate
    # size: batch size
    m = torch.arange(M+1, dtype = torch.float32)
    y = torch.pow(x,m) # forward

    ds_train = Data.DataLoader(Data.TensorDataset(y, t_test), batch_size = size, shuffle= True)

    model = torch.nn.Sequential(torch.nn.Linear(M+1,1))
    model[0].weight.data.normal_(0, 0.1) # initial weight


    
    loss_f = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    num_epochs = 40000
    
    for epoch in range(num_epochs): # num_epochs

        # train with batches of data
        for xb, yb in ds_train:
            pred = model(xb) # pred
            loss = loss_f(pred, yb) # loss
            loss.backward() # backpropagation, compute grad
            opt.step() # update parameters
            opt.zero_grad() # reset the grad to zero
        if epoch % 400 == 399:
            print('epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


    return model[0].weight.data

time_start = time.time()
# compute the optimum weight vector using the test set
w_test_sgd = fit_polynomial_sgd(x_test,t_test,4, 1e-10, 30)

# compute the predicted target values for both the training and test sets
y_pred_test_sgd = polynomial_fun(w_test_sgd,x_test)

time_end = time.time() 
time_sgd= time_end - time_start
print('Time cost for ls is ', time_sgd, 's.')


