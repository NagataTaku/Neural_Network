#include<stdio.h>
#include<stdlib.h>
#include"Eigen/Core"
#include<iostream>
#include<random>
#include<time.h>
#include"layer_naive.h"
#include<map>

using namespace Eigen;
using namespace std;

class TwoLayerNet{
    public:
    map<string,MatrixXd> params;
    
    Affine__ layer_1;
    Relu layer_2;
    Affine__ layer_3;
    
    SoftmaxWithLoss lastlayer;

    double loss_d;


    TwoLayerNet(int input_size,int hidden_size,int output_size){
        double weight_init_std=0.01;
        params["W1"]=weight_init_std * MatrixXd::Random(input_size,hidden_size);
        params["b1"]=MatrixXd::Zero(1,hidden_size);
        params["W2"]=weight_init_std * MatrixXd::Random(hidden_size,output_size);
        params["b2"]=MatrixXd::Zero(1,output_size);


        Affine__ affine_1(params["W1"],params["b1"]);
        layer_1=affine_1;
        Relu relu_1;
        layer_2=relu_1;
        Affine__ affine_2(params["W2"],params["b2"]);
        layer_3=affine_2;
        


        SoftmaxWithLoss softmaxwithloss;
        lastlayer = softmaxwithloss;
    }

    void predict(MatrixXd &x,MatrixXd &y,TwoLayerNet &network){
        MatrixXd y1;
        layer_1.forward(x,y1,network.params["W1"],network.params["b1"]);
        MatrixXd y2;
        layer_2.forward(y1,y2);
        layer_3.forward(y2,y,network.params["W2"],network.params["b2"]);
    }

    double loss(MatrixXd &x,MatrixXd &t,TwoLayerNet &network){
        MatrixXd y;
        predict(x,y,network);
        return lastlayer.forward(y,t);
    }

    double accuracy(MatrixXd &x,MatrixXd &t,TwoLayerNet &network){
        MatrixXd y;
        predict(x,y,network);
        double correct=0;
        for(int i=0;i<y.rows();i++){
            MatrixXf::Index row_y,col_y;
            MatrixXf::Index row_t,col_t;
            y.row(i).maxCoeff(&col_y);
            t.row(i).maxCoeff(&col_t);
            if(col_y==col_t){
                correct++;
            }
        }

        return correct/t.rows();
    }

    void gradient(MatrixXd &x,MatrixXd &t,MatrixXd &grad_W1,MatrixXd &grad_b1,MatrixXd &grad_W2,MatrixXd &grad_b2,TwoLayerNet &network){
        //forward
        loss_d = loss(x,t,network);
        cout<<loss_d<<endl;
        
        //backward
        MatrixXd dout;

        lastlayer.backward(dout);
        layer_3.backward(dout);
        layer_2.backward(dout);
        layer_1.backward(dout);


        grad_W1=layer_1.dW;
        grad_b1=layer_1.db;
        grad_W2=layer_3.dW;
        grad_b2=layer_3.db;
        
    }
};
