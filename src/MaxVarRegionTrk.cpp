#include "MaxVarRegionTrk.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>             /* malloc/free declarations */
#include <string.h>             /* memset declaration */

namespace OptFlowTrk{

MaxVarRegionImpl::MaxVarRegionImpl( const SceneOptFlow::Params &parameters )
{
    isInit = false;
    params =  parameters;
    imageSize = cv::Size(0, 0);
    grid.clear();
    bestgrid.clear();
    var = 10*10;
    memset(m_refVar, 0, sizeof(m_refVar));
    memset(m_sceneInfo, 0, sizeof(m_sceneInfo));
    m_bOptConf = false;
    memset(m_apce, 0, sizeof(m_apce));
}

bool MaxVarRegionImpl::initImpl( const Mat& image, const Rect2d& validBox )
{
	bool iRtn = true;
	CV_Assert(image.channels() == 1);

	if(image.size() != imageSize){
		buildGrid(image);
		imageSize = image.size();
	}

	extractMaxVarBlk(image);
	setRefImage(image);

	return iRtn;
}

bool MaxVarRegionImpl::updateImpl( const Mat& image, Point2f& mvPos )
{
	bool iRtn = true;
	CV_Assert(image.channels() == 1);

	int i, nsize = m_blk.size();

	#pragma omp parallel for
	for(i=0; i<nsize; i++){
		cv::Rect roi = m_blk[i].curPos;
		image(roi).copyTo(m_curSceneMap[i]);
	}

	iRtn = calSceneSimilar(image);
	if(iRtn){
		mvPos = m_curOptConf.pos;
	}else{
		mvPos= cv::Point2f(0.f, 0.f);
	}

	extractMaxVarBlk(image);
	setRefImage(image);

	return iRtn;
}

void MaxVarRegionImpl::getMaxVarRegionsImpl(std::vector<cv::Rect> &rcVector)
{
	int k, nsize = m_blk.size();
	rcVector.resize(nsize);
	for(k=0; k<nsize; k++){
		rcVector[k] = m_blk[k].refPos;
	}
}

static float SubPixelPeak(float left, float center, float right)
{
	float divisor = 2 * center - right - left;

	if (divisor == 0)
		return 0;

	return 0.5 * (right - left) / divisor;
}

static float _calAPCE(cv::Mat result, double minVal, double maxVal)
{
	 float *pdate,sumMean, diff, apce;
	sumMean = 0.f;
	pdate = result.ptr<float>(0,0);
	for(int i=0; i<result.cols*result.rows; i++){
		sumMean += (*pdate-(float)minVal)*(*pdate-(float)minVal);
		pdate++;
	}
	sumMean /=(result.cols*result.rows);
	diff = (float)maxVal-(float)minVal;
	apce =diff*diff/sumMean;
	return apce;
}

void MaxVarRegionImpl::compSceneSimilar(const cv::Mat _curImg, const cv::Mat _tmplModel, SCENE_INFO &posConf, SCENE_BLK &blkInfo, float &apce)
{
	cv::Mat	 result;
	double minVal, maxVal;
	 cv::Point minLoc,maxLoc;
	 cv::Point2f	MaxfLoc;

	posConf.pos.x = 0;
	posConf.pos.y = 0;

	cv::matchTemplate(_curImg, _tmplModel, result, CV_TM_CCOEFF_NORMED);

//	cv::GaussianBlur(result, result, cv::Size(7,7), 0, 0);

	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	MaxfLoc.x = (float)maxLoc.x;
	MaxfLoc.y = (float)maxLoc.y;

	if (maxLoc.x > 0 && maxLoc.x < result.cols-1) {
		 float *psrc0 = result.ptr<float>(maxLoc.y, maxLoc.x-1);
		 float *psrc1 = result.ptr<float>(maxLoc.y, maxLoc.x+1);
		 MaxfLoc.x += SubPixelPeak( *psrc0 , (float)maxVal,  *psrc1);
	 }

	 if (maxLoc.y > 0 && maxLoc.y < result.rows-1) {
		 float *psrc0 = result.ptr<float>(maxLoc.y-1, maxLoc.x);
		 float *psrc1 = result.ptr<float>(maxLoc.y+1, maxLoc.x);
		 MaxfLoc.y += SubPixelPeak(*psrc0, (float)maxVal, *psrc1);
	 }

	posConf.pos.x = MaxfLoc.x;
	posConf.pos.y = MaxfLoc.y;

	if(maxLoc.x == 0 && maxLoc.y == 0)
		maxVal = 0.0;
	posConf.opt = (float)maxVal;
	posConf.pos.x = posConf.pos.x + blkInfo.curPos.x - blkInfo.refPos.x;
	posConf.pos.y = posConf.pos.y + blkInfo.curPos.y  - blkInfo.refPos.y;
}

static float calEuclidean(SCENE_INFO conf1, SCENE_INFO conf2)
{
	return (float)(sqrt((conf1.pos.x-conf2.pos.x)*(conf1.pos.x-conf2.pos.x)+(conf1.pos.y-conf2.pos.y)*(conf1.pos.y-conf2.pos.y)));
}

#define GET_AVE_OPT_POS_3(optConf, I, J, K)	\
		optConf.opt = ((m_sceneInfo[I].opt+m_sceneInfo[J].opt+m_sceneInfo[K].opt)/3.0);		\
		optConf.pos.x = ((m_sceneInfo[I].pos.x+m_sceneInfo[J].pos.x+m_sceneInfo[K].pos.x)/3.0);	\
		optConf.pos.y = ((m_sceneInfo[I].pos.y+m_sceneInfo[J].pos.y+m_sceneInfo[K].pos.y)/3.0);

#define GET_AVE_OPT_POS_2(optConf, I, J)	\
		optConf.opt = (m_sceneInfo[I].opt+m_sceneInfo[J].opt)/2;		\
		optConf.pos.x = ((m_sceneInfo[I].pos.x+m_sceneInfo[J].pos.x)/2.0);	\
		optConf.pos.y = ((m_sceneInfo[I].pos.y+m_sceneInfo[J].pos.y)/2.0);

#define REGET_OPT_POS(optConf, I, J, maxIdx)	\
		if(m_refVar[I]<15 && m_refVar[J]<15 && m_refVar[maxIdx] > 25.0){	\
			optConf.opt = max_Conf.opt;	\
			optConf.pos = max_Conf.pos;	\
		}else if(m_refVar[I]<15 && m_refVar[J]<15 && m_refVar[maxIdx] < 15){	\
			bConf = false;	\
		}

bool MaxVarRegionImpl::calSceneSimilar(const cv::Mat image)
{
	int i, j;
	SCENE_INFO	max_Conf, aveConf, optConf;
	cv::Point2f	sumPt, optPt;
	int maxIdx = 0;
	float  euLen01, euLen02, euLen03,euLen12, euLen13, euLen23;
	float  euLen[MAX_VAR_BLK][MAX_VAR_BLK];
	float  euThred = 1.5;//2.0
	bool	bConf = true;
	float   optThred = 0.75;
	memset(euLen, 0, sizeof(euLen));
	int nsize = m_blk.size();
	memset(&optConf, 0x00, sizeof(optConf));

	memset(m_sceneInfo, 0x00, sizeof(m_sceneInfo));
	memset(m_apce, 0x00, sizeof(m_apce));
#pragma omp parallel for
	for(i=0; i<nsize; i++){
		compSceneSimilar(m_curSceneMap[i], m_refSceneMap[i],m_sceneInfo[i], m_blk[i], m_apce[i]);
	}

//	printf("%s:apce[0]=%.2f,apce[2]=%.2f,apce[2]=%.2f,apce[3]=%.2f \n",__func__, m_apce[0],m_apce[1],m_apce[2],m_apce[3]);

	aveConf.pos.x = aveConf.pos.y = 0;
	aveConf.opt = 0.f;
	max_Conf.opt = 0.f;
	for(i=0; i<MAX_VAR_BLK; i++){
		aveConf.opt +=m_sceneInfo[i].opt;
		aveConf.pos.x += m_sceneInfo[i].pos.x;
		aveConf.pos.y += m_sceneInfo[i].pos.y;
		if(m_sceneInfo[i].opt > max_Conf.opt){
			max_Conf.opt = m_sceneInfo[i].opt;
			max_Conf.pos = m_sceneInfo[i].pos;
			maxIdx = i;
		}
	}

	for(j=0; j<MAX_VAR_BLK; j++){
		for(i= j+1; i<MAX_VAR_BLK; i++){
			euLen[j][i] = calEuclidean(m_sceneInfo[i], m_sceneInfo[j]);
			euLen[i][j] = euLen[j][i];
		}
	}

	if( ((euLen[0][1]<euThred && euLen[0][2]<euThred) || (euLen[0][1]<euThred && euLen[1][2]<euThred)|| (euLen[0][2]<euThred && euLen[1][2]<euThred) )&&
			m_sceneInfo[0].opt > optThred  && m_sceneInfo[1].opt > optThred && m_sceneInfo[2].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 0, 1, 2);
	}else if( ((euLen[0][1]<euThred && euLen[0][3]<euThred) || (euLen[0][1]<euThred && euLen[1][3]<euThred)|| (euLen[0][2]<euThred && euLen[1][3]<euThred)) &&
			m_sceneInfo[0].opt > optThred  && m_sceneInfo[1].opt > optThred && m_sceneInfo[3].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 0, 1, 3);
	}else if( ((euLen[0][2]<euThred && euLen[0][3]<euThred) || (euLen[0][2]<euThred && euLen[2][3]<euThred)|| (euLen[0][3]<euThred && euLen[2][3]<euThred)) &&
			m_sceneInfo[0].opt > optThred  && m_sceneInfo[2].opt > optThred && m_sceneInfo[3].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 0, 2, 3);
	}else if( ((euLen[1][2]<euThred && euLen[1][3]<euThred)  || (euLen[1][2]<euThred && euLen[2][3]<euThred)|| (euLen[2][3]<euThred && euLen[1][3]<euThred)) &&
			m_sceneInfo[1].opt > optThred  && m_sceneInfo[2].opt > optThred && m_sceneInfo[3].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 1, 2, 3);
	}else if(euLen[0][1]<euThred && m_sceneInfo[0].opt > optThred  && m_sceneInfo[1].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 0, 1);
		REGET_OPT_POS(optConf, 0, 1, maxIdx);
	}else if(euLen[0][2]<euThred && m_sceneInfo[0].opt > optThred  && m_sceneInfo[2].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 0, 2);
		REGET_OPT_POS(optConf, 0, 2, maxIdx);
	}else if(euLen[0][3]<euThred && m_sceneInfo[0].opt > optThred  && m_sceneInfo[3].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 0, 3);
		REGET_OPT_POS(optConf, 0, 3, maxIdx);
	}else if(euLen[1][2]<euThred && m_sceneInfo[1].opt > optThred  && m_sceneInfo[2].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 1, 2);
		REGET_OPT_POS(optConf, 1, 2, maxIdx);
	}else if(euLen[1][3]<euThred && m_sceneInfo[1].opt > optThred  && m_sceneInfo[3].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 1, 3);
		REGET_OPT_POS(optConf, 1, 3, maxIdx);
	}else if(euLen[2][3]<euThred && m_sceneInfo[2].opt > optThred  && m_sceneInfo[3].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 2, 3);
		REGET_OPT_POS(optConf, 2, 3, maxIdx);
	}else{
		optConf.opt = max_Conf.opt;
		optConf.pos = max_Conf.pos;
		if(m_refVar[maxIdx] < 30.0)
			bConf = false;
		for(i=0, j=0; i<MAX_VAR_BLK; i++){
			if(m_refVar[i]>25.0)
				j++;
		}
		if(j>2){
			bConf = false;
		}
	}

	m_bOptConf = bConf;
	m_curOptConf = optConf;

	return m_bOptConf;
}

void MaxVarRegionImpl::buildGrid(const cv::Mat &image)
{
	cv::Size	boxsize;
	int width = image.cols;
	int height = image.rows;

	if(width<=768){ //720x576
		boxsize.width = 72;
		boxsize.height = 60;
	}else if(width<=1024){//1024x768
		boxsize.width = 100;
		boxsize.height = 72;
	}else if(width<=1280){//1280x1024
		boxsize.width = 120;
		boxsize.height = 90;
	}else{//1920x1080
		boxsize.width = 156;
		boxsize.height = 96;
	}
	BoundingBox bbox;
	BoundingBox	centBox;
	centBox.x = (width-boxsize.width*3)/2;//width/6;
	centBox.y = height/2-boxsize.height*0.5;
	centBox.width = boxsize.width*3;//width*2/3;
	centBox.height = boxsize.height;
	grid.clear();
	for (int y=height/6;y<image.rows-height/6;y+=boxsize.height/3){
		for (int x=width/6;x<image.cols-width/6;x+=boxsize.width/3){
			if((y+boxsize.height)>height ||(x+boxsize.width)>width)
				continue;
			bbox.x = (x&(~1));
			bbox.y = (y&(~1));
			bbox.width = (boxsize.width&(~1));
			bbox.height = (boxsize.height&(~1));
			bbox.var = 0.0;
			if(bbOverlap(centBox, bbox)>0.1)
				continue;
			grid.push_back(bbox);
		}
	}
}

float MaxVarRegionImpl::bbOverlap(const BoundingBox& box1,const BoundingBox& box2)
{
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }

	float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
	float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

double MaxVarRegionImpl::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum)
{
	double brs = sum.at<int>(box.y+box.height,box.x+box.width);
	double bls = sum.at<int>(box.y+box.height,box.x);
	double trs = sum.at<int>(box.y,box.x+box.width);
	double tls = sum.at<int>(box.y,box.x);
	double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
	double blsq = sqsum.at<double>(box.y+box.height,box.x);
	double trsq = sqsum.at<double>(box.y,box.x+box.width);
	double tlsq = sqsum.at<double>(box.y,box.x);
	double mean = (brs+tls-trs-bls)/((double)box.area());
	double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
	return sqmean-mean*mean;
}

static float _calDistance(BoundingBox rc1, BoundingBox rc2)
{
	float dist;
	Point2f pt1, pt2;
	pt1.x = rc1.x + rc1.width/2;		pt1.y = rc1.y + rc1.height/2;
	pt2.x = rc2.x + rc2.width/2;		pt2.y = rc2.y + rc2.height/2;

	dist = sqrt( (pt1.x-pt2.x)*(pt1.x-pt2.x) + (pt1.y-pt2.y)*(pt1.y-pt2.y));

	return dist;
}

static bool bNearestBox(const std::vector<BoundingBox> &tmpBox, BoundingBox box)
{
	int i, nsize = tmpBox.size();
	float dist, length;
	bool iRtn = true;
	if(nsize == 0)
		return iRtn;
	length = sqrt((float)box.width*box.width+box.height*box.height);
	for(i=0; i<nsize; i++){
		dist = _calDistance(box, tmpBox[i]);
		if(dist < (length*1.2)){
			iRtn = false;
			break;
		}
	}
	return iRtn;
}

void MaxVarRegionImpl::extractMaxVarBlk(const cv::Mat &image)
{
	int i, j, k, count, nsize;
	double val;

	assert(image.channels() == 1);
	integral(image,iisum,iisqsum);

	bestgrid.clear();
	for (i=0;i<grid.size();i++){//FIXME: BottleNeck
		val = getVar(grid[i],iisum,iisqsum);
		grid[i].var = val;
		if (val>=var){
		  bestgrid.push_back(grid[i]);
		}
	}
	if(bestgrid.size() > MAX_VAR_BLK){
		std::sort(bestgrid.begin(), bestgrid.end(), BComparator(bestgrid));//Descending order
	}

	if(bestgrid.size() == 0){
		m_blk.clear();
		return;
	}
	int itern = std::min((int)bestgrid.size(), MAX_VAR_BLK);

	std::vector<BoundingBox>	tmpBox;
	BoundingBox oldBox;
	tmpBox.clear();
	for(k=0,count=0;k<bestgrid.size();k++){
		oldBox = bestgrid[k];
		if(bNearestBox(tmpBox, oldBox)){
			 tmpBox.push_back(oldBox);
			 count++;
		}
		if(count >= itern)
			break;
	}

	nsize = tmpBox.size();
	m_blk.resize(nsize);
	for(k=0; k<nsize; k++){
		m_blk[k].refPos = tmpBox[k];
		m_blk[k].curPos.x = (int)(m_blk[k].refPos.x - m_blk[k].refPos.width*0.25);
		m_blk[k].curPos.y = (int)(m_blk[k].refPos.y - m_blk[k].refPos.height*0.25);
		m_blk[k].curPos.width = (int)(m_blk[k].refPos.width*1.5);
		m_blk[k].curPos.height = (int)(m_blk[k].refPos.height*1.5);
		if(m_blk[k].curPos.x<0) m_blk[k].curPos.x = 0;
		if(m_blk[k].curPos.y<0) m_blk[k].curPos.y = 0;
		if((m_blk[k].curPos.x+m_blk[k].curPos.width)>image.cols)		m_blk[k].curPos.x = image.cols- m_blk[k].curPos.width;
		if((m_blk[k].curPos.y+m_blk[k].curPos.height)>image.rows)	m_blk[k].curPos.y = image.rows- m_blk[k].curPos.height;
		m_blk[k].curPos.x &=(~1);
		m_blk[k].curPos.y &=(~1);
		m_blk[k].curPos.width &=(~1);
		m_blk[k].curPos.height &=(~1);
	}

}

void MaxVarRegionImpl::setRefImage(cv::Mat image)
{
//	image.copyTo(image_ref);
	int i, nsize = m_blk.size();

	memset(m_refVar, 0x00, sizeof(m_refVar));
#pragma omp parallel for
	for(i=0; i<nsize; i++){
		cv::Rect roi = m_blk[i].refPos;
		image(roi).copyTo(m_refSceneMap[i]);
		 Scalar stdev, mean;
		 meanStdDev(m_refSceneMap[i],mean,stdev);
		 m_refVar[i] = stdev.val[0];
	}
}

}
