#include<stdio.h>
#include<stdlib.h>
#include"Eigen/Core"
#include<iostream>
#include<random>
#include<time.h>

using namespace Eigen;
using namespace std;

void softmax(MatrixXd &a, MatrixXd &y){ //aが入力,yが出力
    MatrixXd exp_a(a.rows(),a.cols());
    y=MatrixXd::Constant(a.rows(),a.cols(),1);
    MatrixXd c(a.rows(),1);
    double sum_exp_a=0;

    c=a.rowwise().maxCoeff();

    for(int i=0;i<a.rows();i++){
        if ((a.row(i).array() == 0).all() == 1){
            a.row(i)=MatrixXd::Constant(1,a.cols(),1);
            cout<<"error"<<endl;
        }
        for(int j=0;j<a.cols();j++){
            exp_a(i,j)=exp(a(i,j)-c(i,0));
        }
        sum_exp_a = exp_a.row(i).sum();
        y.row(i) = exp_a.row(i) / sum_exp_a;
    }
}

void sigmoid(MatrixXd &x,MatrixXd &y){
    for(int i=0;i<x.cols();i++){
        for(int j=0;j<x.rows();j++){
            y(i,j)=1/(1+exp(-x(i,j)));
        }
    }

}

double cross_entropy_error(MatrixXd &y, MatrixXd &t){ //yが予測値、tが正解
    double delta=1e-7;
    y.array()+=delta;
    return -(y.array().log() * t.array()).sum();
}


void numerical_gradient(function<double(MatrixXd &, MatrixXd &, MatrixXd &)> f, MatrixXd &x, MatrixXd &t, MatrixXd &weight, MatrixXd &grad){
    double h=1e-7;

    for(int i=0;i<weight.rows();i++){
        for(int j=0;j<weight.cols();j++){
            // 元の値を一旦入れておく
            double tmp_val=weight(i,j);

            // f(x+h)
            weight(i,j) = tmp_val + h;
            double fxh1 = f(x,t,weight);

            // f(x-h)
            weight(i,j) = tmp_val - h;
            double fxh2 = f(x,t,weight);

            // 微量な変化分をhで割って微分している
            grad(i,j)=(fxh1-fxh2)/(2*h);

            // 元の値に戻す
            weight(i,j)=tmp_val;
        }
    }
}