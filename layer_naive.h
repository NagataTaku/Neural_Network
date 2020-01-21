#include<stdio.h>
#include<stdlib.h>
#include"Eigen/Core"
#include"functions.h"

using namespace Eigen;
using namespace std;

class MulLayer{
    public:

    double x;
    double y;

    double forward(double init_x,double init_y){
        x=init_x;
        y=init_y;
        double out=x*y;
        return out;
    }

    void backward(double dout,double &dx,double &dy){
        dx=dout*y;
        dy=dout*x;
    }
};

class AddLayer{
    public:

    double forward(double x,double y){
        double out=x+y;
        return out;
    }

    void backward(double dout,double &dx,double &dy){
        dx=dout*1;
        dy=dout*1;
    }

};

class Relu{
    public:
    MatrixXd mask;
    
    void forward(MatrixXd &x,MatrixXd &out){
        out=MatrixXd::Zero(x.rows(),x.cols());
        mask=MatrixXd::Zero(x.rows(),x.cols());

        for(int i=0;i<x.rows();i++){
            for(int j=0;j<x.cols();j++){
                if(x(i,j) > 0){
                    out(i,j)=x(i,j);
                    mask(i,j)=1;
                }
            }
        }
    }

    void backward(MatrixXd &dout){
        dout=dout.array()*mask.array();
    }
};

class Sigmoid{
    public:
    MatrixXd out;

    void forward(MatrixXd &x,MatrixXd &ret){
        for(int i=0;i<x.rows();i++){
            for(int j=0;j<x.cols();j++){
                ret(i,j)=1/(1+exp(-x(i,j)));
            }
        }
        out=ret;
    }

    void backward(MatrixXd &dout){
        MatrixXd dx;
        for(int i=0;i<dout.rows();i++){
            for(int j=0;j<dout.cols();j++){
                dx(i,j)=dout(i,j)*(1.0-out(i,j))*out(i,j);
            }
        }

        dout=dx;
    }
};

class Affine__{
    public:
    MatrixXd W;
    MatrixXd b;
    MatrixXd x;
    MatrixXd dW;
    MatrixXd db;
    
    Affine__(){
    }

    Affine__(MatrixXd &W_init,MatrixXd &b_init){
        W=W_init;
        b=b_init;
    }

    void forward(MatrixXd &x_forward,MatrixXd &out,MatrixXd &W1,MatrixXd &b1){
        W = W1;
        b = b1;
        x=x_forward;
        out=x*W;
        for(int i=0;i<x.rows();i++){
            out.row(i)=out.row(i)+b;
        }
    }

    void backward(MatrixXd &dout){
        MatrixXd dx;
        dx=dout*W.transpose();
        dW=x.transpose()*dout;

        db=MatrixXd::Zero(1,dout.cols());

        for(int i=0;i<dout.cols();i++){
            db(0,i)=dout.col(i).sum();
        }

        dout=dx;
    }

};

class SoftmaxWithLoss{
    public:
    double loss;
    MatrixXd y;
    MatrixXd t;

    double forward(MatrixXd &x,MatrixXd &t_init){
        t=t_init;
        softmax(x,y);
        loss=cross_entropy_error(y,t);
        return loss;
    }


    void backward(MatrixXd &dx){
        double batch_size=t.rows();
        dx=y-t;
        double inv=1.0/batch_size;
        dx=dx*inv;
    }

};
