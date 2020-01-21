#include<stdio.h>
#include<stdlib.h>
#include"Eigen/Core"
#include"mnist.hpp"
#include"TwoLayerNet.h"
#include<map>
#include<vector>
#include<algorithm>
#include<random>
#include<iterator>
#include<iostream>
#include<string>

using namespace Eigen;
using namespace std;

int main(){
    srand((unsigned int) time(0));

    MatrixXd x_train=mnist::train_data();
    MatrixXd t_train=mnist::train_label();
    MatrixXd x_test=mnist::test_data();
    MatrixXd t_test=mnist::test_label();

    x_train.transposeInPlace();
    t_train.transposeInPlace();
    x_test.transposeInPlace();
    t_test.transposeInPlace();
    
    TwoLayerNet network(784,50,10);

    int iters_num=1200;
    int train_size=x_train.rows();
    int batch_size=100;
    double learning_rate = 0.1;
    vector<double> train_loss_list;
    vector<double> train_acc_list;
    vector<double> test_acc_list;

    int iter_per_epoch=max({train_size/batch_size,1});

    train_size=x_train.rows();

    for(int i=0;i<iters_num;i++){
        MatrixXd x_batch(batch_size,x_train.cols());
        MatrixXd t_batch(batch_size,t_train.cols());

        x_batch=x_train.middleRows((i%iter_per_epoch)*batch_size,batch_size);
        t_batch=t_train.middleRows((i%iter_per_epoch)*batch_size,batch_size);

        MatrixXd grad_W1;
        MatrixXd grad_b1;
        MatrixXd grad_W2;
        MatrixXd grad_b2;
        
        network.gradient(x_batch,t_batch,grad_W1,grad_b1,grad_W2,grad_b2,network);
        
        network.params["W1"] -= learning_rate * grad_W1;
        network.params["b1"] -= learning_rate * grad_b1;
        network.params["W2"] -= learning_rate * grad_W2;
        network.params["b2"] -= learning_rate * grad_b2;

        cout<<"iter:"<<i<<endl;

        if(i % iter_per_epoch == 0){
            cout<<"train_acc"<<endl;
            cout<<network.accuracy(x_train,t_train,network)<<endl;
            cout<<"test_acc"<<endl;
            cout<<network.accuracy(x_test,t_test,network)<<endl;
        }

    }

    

    return 0;
}