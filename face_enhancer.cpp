#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/photo.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
using namespace std;
using namespace cv;

struct Args {
    string in, out, sr="models/EDSR_x4.pb", proto="models/opencv_face_detector.prototxt",
           weights="models/opencv_face_detector.caffemodel", cascade="";
    int scale=4; float conf=0.5f; bool faceOnly=true; bool finalPass=true;
    double clip=-1, gclip=-1, sharp=-1, gsharp=-1, gamma=1.0; // optional manual overrides
};

static bool exists(const string& p){ return !p.empty() && ifstream(p).good(); }

static Args parse(int ac, char** av){
    if(ac<3){
        cerr<<"Usage: "<<av[0]<<" <in> <out> [--sr pb] [--scale 2|3|4|8] "
              "[--proto p] [--weights w] [--cascade xml] "
              "[--clip v] [--gclip v] [--sharp v] [--gsharp v] [--gamma v] "
              "[--no-face-only] [--no-final]\n"; exit(1);
    }
    Args a; a.in=av[1]; a.out=av[2];
    for(int i=3;i<ac;i++){ string k=av[i];
        if(k=="--sr"&&i+1<ac) a.sr=av[++i];
        else if(k=="--scale"&&i+1<ac) a.scale=max(2,min(8,atoi(av[++i])));
        else if(k=="--proto"&&i+1<ac) a.proto=av[++i];
        else if(k=="--weights"&&i+1<ac) a.weights=av[++i];
        else if(k=="--cascade"&&i+1<ac) a.cascade=av[++i];
        else if(k=="--clip"&&i+1<ac) a.clip=max(0.0,atof(av[++i]));
        else if(k=="--gclip"&&i+1<ac) a.gclip=max(0.0,atof(av[++i]));
        else if(k=="--sharp"&&i+1<ac) a.sharp=max(0.0,atof(av[++i]));
        else if(k=="--gsharp"&&i+1<ac) a.gsharp=max(0.0,atof(av[++i]));
        else if(k=="--gamma"&&i+1<ac) a.gamma=max(0.1,atof(av[++i]));
        else if(k=="--no-face-only") a.faceOnly=false;
        else if(k=="--no-final") a.finalPass=false;
    } return a;
}

static Rect clamp(Rect r, Size s){ return r & Rect(0,0,s.width,s.height); }

static Mat to8UC3(const Mat& m){
    if(m.empty()) return m;
    if(m.type()==CV_8UC3) return m;
    Mat r;
    if(m.channels()==3){
        double minv,maxv; minMaxLoc(m.reshape(1),&minv,&maxv);
        double sc=(m.depth()==CV_32F||m.depth()==CV_64F) && maxv<=1.0 ? 255.0 : 1.0;
        m.convertTo(r,CV_8UC3,sc);
    }else{
        Mat g; double minv,maxv; minMaxLoc(m,&minv,&maxv);
        double sc=(m.depth()==CV_32F||m.depth()==CV_64F)&&maxv<=1.0?255.0:1.0;
        m.convertTo(g,CV_8U,sc); cvtColor(g,r,COLOR_GRAY2BGR);
    }
    return r;
}

static Mat gammaCorrect(const Mat& src,double gamma){
    if(fabs(gamma-1.0)<1e-6) return to8UC3(src);
    Mat s=to8UC3(src), lut(1,256,CV_8U);
    for(int i=0;i<256;i++) lut.at<uchar>(i)=(uchar)cvRound(255.0*pow(i/255.0,1.0/gamma));
    Mat dst; LUT(s,lut,dst); return dst;
}

static void claheY(Mat& bgr,double clip){
    bgr=to8UC3(bgr);
    if(clip<=0) return;
    Mat ycrcb; cvtColor(bgr,ycrcb,COLOR_BGR2YCrCb);
    vector<Mat> ch; split(ycrcb,ch);
    Ptr<CLAHE> c=createCLAHE(clip,Size(8,8));
    Mat y; c->apply(ch[0],y); ch[0]=y; merge(ch,ycrcb);
    cvtColor(ycrcb,bgr,COLOR_YCrCb2BGR);
}

static Mat unsharp(const Mat& src,double sigma,double amt){
    Mat s=to8UC3(src), g,out; GaussianBlur(s,g,Size(),sigma,sigma);
    addWeighted(s,1.0+amt,g,-amt,0,out); return out;
}

static Mat bilateral8u3(const Mat& src,int d,double sc,double ss){
    Mat s=to8UC3(src), dst; bilateralFilter(s,dst,d,sc,ss); return dst;
}

static Rect biggestFaceDNN(const Mat& bgr,dnn::Net& net,float conf){
    Mat blob=dnn::blobFromImage(bgr,1.0,Size(300,300),Scalar(104,177,123),false,false);
    net.setInput(blob); Mat o=net.forward();
    Mat det(o.size[2],o.size[3],CV_32F,o.ptr<float>());
    Rect best; float bestA=-1.f;
    for(int i=0;i<det.rows;i++){
        float cf=det.at<float>(i,2); if(cf<conf) continue;
        Rect r(Point(int(det.at<float>(i,3)*bgr.cols),int(det.at<float>(i,4)*bgr.rows)),
               Point(int(det.at<float>(i,5)*bgr.cols),int(det.at<float>(i,6)*bgr.rows)));
        r=clamp(r,bgr.size()); if(r.area()>bestA){ bestA=(float)r.area(); best=r; }
    } return best;
}

static Rect biggestFaceCascade(const Mat& bgr,const string& xml){
    if(!exists(xml)) return Rect(); CascadeClassifier cc; if(!cc.load(xml)) return Rect();
    vector<Rect> v; Mat g; cvtColor(bgr,g,COLOR_BGR2GRAY); equalizeHist(g,g);
    cc.detectMultiScale(g,v,1.1,3,0|CASCADE_SCALE_IMAGE,Size(30,30));
    Rect best; size_t bestA=0; for(auto& r:v) if((size_t)r.area()>bestA){bestA=r.area(); best=r;}
    return best;
}

static Mat superResolve(const Mat& src,const string& model,int scale){
    using namespace dnn_superres; Mat up;
    if(exists(model)) try{
        DnnSuperResImpl sr; string name="edsr"; string low=model;
        transform(low.begin(),low.end(),low.begin(),::tolower);
        if(low.find("espcn")!=string::npos) name="espcn";
        else if(low.find("fsrcnn")!=string::npos) name="fsrcnn";
        else if(low.find("lapsrn")!=string::npos) name="lapsrn";
        sr.readModel(model); sr.setModel(name,scale); sr.upsample(src,up);
    }catch(...){}
    if(up.empty()) resize(src,up,Size(),scale,scale,INTER_CUBIC);
    return to8UC3(up);
}

static Mat feather(Size sz){
    Mat m(sz,CV_8UC1,Scalar(0));
    ellipse(m,Point(sz.width/2,sz.height/2),Size(int(sz.width*0.48),int(sz.height*0.58)),
            0,0,360,Scalar(255),-1,LINE_AA);
    GaussianBlur(m,m,Size(),5.0); return m;
}

struct Tuned {
    double clip, gclip, sharp, gsharp, gamma;
    float detSigmaS, detSigmaR;
    int bd; double bc, bs;
};

// Natural-only defaults, with optional overrides from flags
static Tuned naturalParams(const Args& a){
    Tuned p{1.2, 0.0, 0.35, 0.15, 1.0, 8.f, 0.08f, 7, 55, 55};
    if(a.clip>=0)   p.clip=a.clip;
    if(a.gclip>=0)  p.gclip=a.gclip;
    if(a.sharp>=0)  p.sharp=a.sharp;
    if(a.gsharp>=0) p.gsharp=a.gsharp;
    if(a.gamma!=1.0)p.gamma=a.gamma;
    return p;
}

int main(int ac, char** av){
    Args a=parse(ac,av);
    Tuned p=naturalParams(a);

    Mat img=imread(a.in,IMREAD_COLOR);
    if(img.empty()){ cerr<<"Failed to read input.\n"; return 2; }

    dnn::Net fd; bool haveFD=exists(a.proto)&&exists(a.weights);
    if(haveFD){ try{ fd=dnn::readNetFromCaffe(a.proto,a.weights);}catch(...){haveFD=false;} }

    Rect face; if(haveFD) face=biggestFaceDNN(img,fd,a.conf);
    if((face.width<=0||face.height<=0) && !a.cascade.empty()) face=biggestFaceCascade(img,a.cascade);

    Mat out=img.clone();

    if(a.faceOnly && face.area()>0){
        int px=int(face.width*0.35), py=int(face.height*0.45);
        Rect roi=clamp(Rect(face.x-px,face.y-py,face.width+2*px,face.height+2*py), out.size());
        Mat crop=out(roi).clone();

        Mat up=superResolve(crop,a.sr,a.scale);
        up=gammaCorrect(up,p.gamma);
        up=bilateral8u3(up,p.bd,p.bc,p.bs);
        detailEnhance(up,up,p.detSigmaS,p.detSigmaR);
        claheY(up,p.clip);
        up=unsharp(up,1.0,p.sharp);

        Mat down; resize(up,down,crop.size(),0,0,INTER_LANCZOS4);
        Mat mask=feather(down.size());
        Point center(roi.x+roi.width/2, roi.y+roi.height/2);
        Mat blended; seamlessClone(down,out,mask,center,blended,MIXED_CLONE);
        out=blended;
    }else{
        Mat up=superResolve(out,a.sr,min(a.scale,4));
        up=gammaCorrect(up,p.gamma);
        up=bilateral8u3(up,p.bd-1,max(40.0,p.bc-10),max(40.0,p.bs-10));
        detailEnhance(up,up,p.detSigmaS*0.9f,p.detSigmaR*0.9f);
        claheY(up,p.clip);
        up=unsharp(up,0.9,p.sharp*0.85);
        resize(up,out,img.size(),0,0,INTER_LANCZOS4);
    }

    Mat finalImg=out.clone();
    if(a.finalPass && p.gclip>0) claheY(finalImg,p.gclip);
    if(a.finalPass && p.gsharp>0) finalImg=unsharp(finalImg,0.8,p.gsharp);

    if(!imwrite(a.out,finalImg)){ cerr<<"Failed to write output.\n"; return 3; }
    cout<<"Saved: "<<a.out<<"\n";
    return 0;
}
