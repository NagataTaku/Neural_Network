#include<stdio.h>
#include<stdlib.h>
#include"Eigen/Core"
#include<iostream>
#include<random>
#include<time.h>
#include"mnist.hpp"

using namespace Eigen;
using namespace std;

double cross_entropy_error(MatrixXd &y, MatrixXd &t){ //yが予測値、tが正解
    double delta=1e-7;
    y.array()+=delta;

    return -(y.array().log()*t.array()).sum();
}

void softmax(MatrixXd &a, MatrixXd &y){ //aが入力,yが出力

    VectorXd exp_a(a.cols());

    double c= a.maxCoeff();

    for(int i=0;i<a.cols();i++){
        exp_a(i)=exp(a(i)-c);
    }

    double sum_exp_a=exp_a.sum();

    y=exp_a.array()/sum_exp_a;

    y.transposeInPlace();
}

void sigmoid(MatrixXd &x,MatrixXd &y){
    for(int i=0;i<x.rows();i++){
        for(int j=0;j<x.cols();j++){
            y(i,j)=1/(1+exp(-x(i,j)));
        }
    }

}

void set_weight(MatrixXd& weight,int input_size,int output_size){
    srand(0);
    weight=MatrixXd::Random(input_size,output_size);
}

double loss_calc(MatrixXd &predict,MatrixXd &t){
    MatrixXd y;
    softmax(predict ,y);
    return cross_entropy_error(y,t);
}

//xは入力、yは出力
void predict(MatrixXd &x,MatrixXd &y,MatrixXd &t,MatrixXd &W1,MatrixXd &W2){
    int input_size=2;
    int hidden_size=3;
    int output_size=3;

    MatrixXd a1;

    a1=x*W1;

    MatrixXd z1 = MatrixXd::Zero(1,hidden_size);

    sigmoid(a1,z1);

    MatrixXd a2;
    a2=z1*W2;

    softmax(a2,y);
}

double f(MatrixXd &x,MatrixXd &y, MatrixXd &t, MatrixXd &W1,MatrixXd &W2){ //xは元データ、tは正解ラベル、weightは重みに使う
    predict(x,y,t,W1,W2);
    return cross_entropy_error(y,t);
}


void numerical_gradient(function<double(MatrixXd &, MatrixXd &, MatrixXd &,MatrixXd &,MatrixXd &)> f, MatrixXd &x, MatrixXd &y, MatrixXd &t, MatrixXd &W1,MatrixXd &W2, MatrixXd &grad_W1, MatrixXd &grad_W2){
    double h=1e-4;

    for(int i=0;i<W1.rows();i++){
        for(int j=0;j<W1.cols();j++){
            // 元の値を一旦入れておく
            double tmp_val=W1(i,j);

            // f(x+h)
            W1(i,j) = tmp_val + h;
            double fxh1 = f(x,y,t,W1,W2);

            // f(x-h)
            W1(i,j) = tmp_val - h;
            double fxh2 = f(x,y,t,W1,W2);

            // 微量な変化分をhで割って微分している
            grad_W1(i,j)=(fxh1-fxh2)/(2*h);

            // 元の値に戻す
            W1(i,j)=tmp_val;
        }
    }

    for(int i=0;i<W2.rows();i++){
        for(int j=0;j<W2.cols();j++){
            // 元の値を一旦入れておく
            double tmp_val=W2(i,j);

            // f(x+h)
            W2(i,j) = tmp_val + h;
            double fxh1 = f(x,y,t,W1,W2);

            // f(x-h)
            W2(i,j) = tmp_val - h;
            double fxh2 = f(x,y,t,W1,W2);

            // 微量な変化分をhで割って微分している
            grad_W2(i,j)=(fxh1-fxh2)/(2*h);

            // 元の値に戻す
            W2(i,j)=tmp_val;
        }
    }
}




void study(int input_size,int hidden_size,int output_size,MatrixXd &x,MatrixXd &t){
    MatrixXd W1(input_size,hidden_size);
    MatrixXd W2(hidden_size,output_size);

    set_weight(W1,input_size,hidden_size);
    set_weight(W2,hidden_size,output_size);
    
    MatrixXd grad_W1(input_size,hidden_size);
    MatrixXd grad_W2(hidden_size,output_size);

    MatrixXd y;


    int iter_num=10000;
    double learning_rate=1;

    for(int i=0;i<iter_num;i++){
        numerical_gradient(f,x,y,t,W1,W2,grad_W1,grad_W2);
        W1=W1.array()-learning_rate*grad_W1.array();
        W2=W2.array()-learning_rate*grad_W2.array();
    }

    predict(x,y,t,W1,W2);
    cout<<loss_calc(y,t)<<endl;
}


int main(){
    MatrixXd x; //サイズ指定しないと動かない・・・
    x=mnist::train_data();

    MatrixXd t; //サイズ指定しないと動かない・・・
    t=mnist::train_label();


    
    return 0;
}