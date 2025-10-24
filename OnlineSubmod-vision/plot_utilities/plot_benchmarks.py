baselines = ["gradmatch"]

filename = '../gradmatch_cifar10_0.5.log'


#plot accuracy vs speed-up

def best_acc_finder(file):
    for line in file:
        if 'INFO: Best Accuracy' in line:
            acc = (line.split(": "))
            print(acc[2])
def time_finder(file):
    for line in file:
        if 'INFO: Time' in line:
            acc = (line.split(": "))
            print(acc[2])
def test_acc_finder(file):
    for line in file:
        if 'INFO: Test Accuracy' in line:
            acc = (line.split(": "))
            print(acc[2])
def val_acc_finder(file):
    for line in file:
        if 'INFO: Validation Accuracy' in line:
            acc = (line.split(": "))
            print(acc[2])

source = open(filename,'rb')

best_acc = best_acc_finder(source)
