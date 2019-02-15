function [x, y]=CorrSampleGenerator(type,n,dim,dependent, noise)
% Author: Cencheng Shen
% Generate sample data x and y for testing independence based on given
% distribution.
%
% Parameters:
% type specifies the type of distribution,
% n is the sample size, dim is the dimension
% dependent specifies whether the data are dependent or not, by default 1
% noise specifies the noise level, by default 0.
if nargin<4
    dependent=1; % by default generate dependent samples
end
if nargin<4
    noise=0; % default noise level
end

eps=mvnrnd(0,1,n); % Gaussian noise added to Y

% High-dimensional decay
A=ones(dim,1);
%A=A./(ceil(dim*rand(dim,1))); %random decay
for d=1:dim
    A(d)=A(d)/d; %fixed decay
end
d=dim;

% Generate x by uniform distribution first, which is the default distribution used by many types; store the weighted summation in xA.
x=unifrnd(-1,1,n,d);
xA=x*A;
% Generate x independently by uniform if the null hypothesis is true, i.e., x is independent of y.
if dependent==0
    x=unifrnd(-1,1,n,d);
end

switch type % In total 20 types of dependency + the type 0 outlier model
    case 1 %Linear
        y=xA+1*noise*eps;
    case 2 %Exponential
        x=unifrnd(0,3,n,d);
        y=exp(x*A)+10*noise*eps;
        if dependent==0
            x=unifrnd(0,3,n,d);
        end
    case 3 %Cubic
        y=128*(xA-1/3).^3+48*(xA-1/3).^2-12*(xA-1/3)+80*noise*eps;
    case 4 %Joint Normal; note that dim should be no more than 10 as the covariance matrix for dim>10 is no longer positive semi-definite
        rho=1/(d*2);
        cov1=[eye(d) rho*ones(d)];
        cov2=[rho*ones(d) eye(d)];
        covT=[cov1' cov2'];
        x=mvnrnd(zeros(n,2*d),covT,n);
        y=x(:,d+1:2*d)+0.5*noise*repmat(eps,1,d);
        if dependent==0
            x=mvnrnd(zeros(n,2*d),covT,n);
        end
        x=x(:,1:d);
    case 5 %Step Function
        if dim>1
            noise=1;
        end
        y=(xA>0)+1*noise*eps;
    case 6 %Quadratic
        y=(xA).^2+0.5*noise*eps;
    case 7 %W Shape
        y=4*( ( xA.^2 - 1/2 ).^2 + unifrnd(0,1,n,d)*A/500 )+0.5*noise*eps;
    case 9 %Uncorrelated Binomial
        if d>1
            noise=1;
        end
        x=binornd(1,0.5,n,d)+0.5*noise*mvnrnd(zeros(n,d),eye(d),n);
        y=(binornd(1,0.5,n,1)*2-1);
        y=x*A.*y+0.5*noise*eps;
        if dependent==0
            x=binornd(1,0.5,n,d)+0.5*noise*mvnrnd(zeros(n,d),eye(d),n);
        end
    case 10 %Log(X^2)
        x=mvnrnd(zeros(n, d),eye(d));
        y=log(x.^2)+3*noise*repmat(eps,1,d);
        if dependent==0
            x=mvnrnd(zeros(n, d),eye(d));
        end
    case 11 %Fourth root
        y=abs(xA).^(0.25)+noise/4*eps;
    case {8,16,17} %Circle & Ecllipse & Spiral
        if d>1
            noise=1;
        end
        cc=0.4;
        if type==16
            rx=ones(n,d);
        end
        if type==17
            rx=5*ones(n,d);
        end
        
        if type==8
            rx=unifrnd(0,5,n,1);
            ry=rx;
            rx=repmat(rx,1,d);
            z=rx;
        else
            z=unifrnd(-1,1,n,d);
            ry=ones(n,1);
        end
        x(:,1)=cos(z(:,1)*pi);
        for i=1:d-1;
            x(:,i+1)=x(:,i).*cos(z(:,i+1)*pi);
            x(:,i)=x(:,i).*sin(z(:,i+1)*pi);
        end
        x=rx.*x;
        y=ry.*sin(z(:,1)*pi);
        if type==8
            y=y+cc*(dim)*noise*mvnrnd(zeros(n, 1),eye(1));
        else
            x=x+cc*noise*rx.*mvnrnd(zeros(n, d),eye(d));
        end
        if dependent==0
            if type==8
                rx=unifrnd(0,5,n,1);
                rx=repmat(rx,1,d);
                z=rx;
            else
                z=unifrnd(-1,1,n,d);
            end
            x(:,1)=cos(z(:,1)*pi);
            for i=1:d-1;
                x(:,i+1)=x(:,i).*cos(z(:,i+1)*pi);
                x(:,i)=x(:,i).*sin(z(:,i+1)*pi);
            end
            x=rx.*x;
            if type==8
            else
                x=x+cc*noise*rx.*mvnrnd(zeros(n, d),eye(d));
            end
        end
    case {12,13} %Sine 1/2 & 1/8
        x=repmat(unifrnd(-1,1,n,1),1,d);
        if noise>0 || d>1
            x=x+0.02*(d)*mvnrnd(zeros(n,d),eye(d),n);
        end
        if type==12
            theta=4;cc=1;
        else
            theta=16;cc=0.5;
        end
        y=sin(theta*pi*x)+cc*noise*repmat(eps,1,d);
        if dependent==0
            x=repmat(unifrnd(-1,1,n,1),1,d);
            if noise>0 || d>1
                x=x+0.02*(d)*mvnrnd(zeros(n,d),eye(d),n);
            end
        end
    case {14,18} %Square & Diamond
        u=repmat(unifrnd(-1,1,n,1),1,d);
        v=repmat(unifrnd(-1,1,n,1),1,d);
        if type==14
            theta=-pi/8;
        else
            theta=-pi/4;
        end
        eps=0.05*(d)*mvnrnd(zeros(n,d),eye(d),n);
        x=u*cos(theta)+v*sin(theta)+eps;
        y=-u*sin(theta)+v*cos(theta);
        if dependent==0
            u=repmat(unifrnd(-1,1,n,1),1,d);
            v=repmat(unifrnd(-1,1,n,1),1,d);
            eps=0.05*(d)*mvnrnd(zeros(n,d),eye(d),n);
            x=u*cos(theta)+v*sin(theta)+eps;
        end
    case 15 %Two Parabolas
        y=( xA.^2  + 2*noise*unifrnd(0,1,n,1)).*(binornd(1,0.5,n,1)-0.5);
    case 19 %Multiplicative Noise
        x=mvnrnd(zeros(n, d),eye(d));
        y=mvnrnd(zeros(n, d),eye(d));
        y=x.*y;
        if dependent==0
            x=mvnrnd(zeros(n, d),eye(d));
        end
    case 20 %Independent clouds
        x=mvnrnd(zeros(n,d),eye(d),n)/3+(binornd(1,0.5,n,d)-0.5)*2;
        y=mvnrnd(zeros(n,d),eye(d),n)/3+(binornd(1,0.5,n,d)-0.5)*2;
    case 21 %Independent clouds
        n1=floor(n*noise);
        n2=n-n1;
        x1=unifrnd(-1,1,n2,1);
        y1=x1;
        if n1>0
        x2=mvnrnd(zeros(n1,d),eye(d),n1)/3+(binornd(1,0.5,n1,d)-0.5)*2;
        y2=mvnrnd(zeros(n1,d),eye(d),n1)/3+(binornd(1,0.5,n1,d)-0.5)*2;
        x=[x1;x2];
        y=[y1;y2];
        else
        x=x1;
        y=y1;
        end
end

%affine invariant
%x=x*cov(x)^(-0.5);
%y=y*cov(y)^(-0.5);