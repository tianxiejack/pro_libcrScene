#include "sceneProc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <assert.h>

namespace OptFlowTrk{

SceneProc::SceneProc()
{
	m_ScenePtr = cv::Ptr<SceneOptFlow>();
	uncsentedKalmanFilter = cv::Ptr<UnscentedKalmanFilter>();
	model = cv::Ptr<UkfSystemModel>();
	standardSize = cv::Size(1920,1080);
	m_bakTS = 0;
	m_accNoise  = 0.0;//1e-3;
	sceneStatus = INIT_SCENE;

	m_bakDetlta.x = m_bakDetlta. y = 0.f;
	m_filterVel.x= m_filterVel.y = 0.f;
	m_instanVel.x = m_instanVel.y = 0.f;
	m_sumDelta.x = m_sumDelta.y = 0.f;
	m_instanConfd = 0.f;
}

SceneProc::~SceneProc()
{
	m_sceneState.clear();

}

UInt32 dbg_getCurTimeInUsec()
{
  static int isInit = false;
  static UInt32 initTime=0;
  struct timeval tv;

  if(isInit==false)
  {
      isInit = true;

      if (gettimeofday(&tv, NULL) < 0)
        return 0;

      initTime = (UInt32)(tv.tv_sec * 1000u * 1000 + tv.tv_usec);
  }

  if (gettimeofday(&tv, NULL) < 0)
    return 0;

  return (UInt32)(tv.tv_sec * 1000u * 1000u + tv.tv_usec)-initTime;
}

void SceneProc::SetStandardSize(cv::Size sz)
{
	standardSize = sz;
}

void SceneProc::InitSceneLock(const cv::Mat image)
{
	SceneState		tpState;
	m_sceneState.clear();

	tpState.ts = *((UInt32*)image.data);
	tpState.mv = cv::Point2f(0.f, 0.f);
	tpState.confidence = 1.0;
	tpState.deltaT = 20000;//us

	if(m_ScenePtr.empty()){
		SceneOptFlow::Params parameters;
		parameters.fpMode = GOOD_FEATPOINT;
		parameters.motionModel_ = LINEAR_SIMILARITY;
		parameters.ransacParams_ = RansacParams::linearSimilarityMotionStd();
		m_ScenePtr = SceneOptFlow::createInterface(parameters);

		assert(!m_ScenePtr.empty());
	}
	Rect2d	boundingRect;
	boundingRect.x = boundingRect.y = 0;
	boundingRect.width = image.cols;	boundingRect.height = image.rows;
	m_ScenePtr->init(image, boundingRect);
	m_sceneState.push_back(tpState);
	m_bakTS = tpState.ts;

	if( !uncsentedKalmanFilter.empty()){
		uncsentedKalmanFilter.release();
		model.release();
	}
	m_rng = cv::RNG(dbg_getCurTimeInUsec());

	InitUKFilter(0.f, 0.f);
	m_sumDelta.x = m_sumDelta.y = 0.f;
	m_filterVel.x = m_filterVel.y = 0.f;
	m_instanVel.x= m_instanVel.y = 0.f;
	m_bakDetlta.x= m_bakDetlta.y = 0.f;
	m_instanConfd = tpState.confidence;

	sceneStatus = RUN_SCENE;

}

void SceneProc::CalSceneLock(const cv::Mat image)
{
	Point2f mvPos;
	SceneState		tpState;
	float fx, fy;
	tpState.ts = *((UInt32*)image.data);
	assert(!m_ScenePtr.empty());
	assert(!image.empty());
	assert(tpState.ts>m_bakTS);

	bool iRtn = m_ScenePtr->update(image, mvPos);

	fx = standardSize.width*1.0/image.cols;
	fy = standardSize.height*1.0/image.rows;

	if(m_sceneState.size() >= MAX_SCENE_FRAMES){
		StateVectorIterator it = m_sceneState.begin();
		m_sceneState.erase(it);
	}
	if(iRtn){
		tpState.confidence =1.0;
	}else{
		tpState.confidence =0.f;
	}
	tpState.mv.x = mvPos.x*fx;
	tpState.mv.y = mvPos.y*fy;
	tpState.deltaT = tpState.ts - m_bakTS;
	m_sceneState.push_back(tpState);

	//printf("%s:timestamp=%d, deltaT=%d\n",__func__, tpState.ts, tpState.deltaT);

	m_bakTS = tpState.ts;

	m_instanVel.x = tpState.mv.x;
	m_instanVel.y = tpState.mv.y;
	m_instanConfd = tpState.confidence;
	AnalyseSceneLock();
}

class ObjectTrackerModel: public UkfSystemModel
{
public:
    void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1)
    {
    	x_kplus1.at<double>(0,0) = x_k.at<double>(0,0)+u_k.at<double>(0,0)*x_k.at<double>(2,0)+u_k.at<double>(0,1)*v_k.at<double>(0,0);
    	x_kplus1.at<double>(1,0) = x_k.at<double>(1,0)+u_k.at<double>(1,0)*x_k.at<double>(3,0)+u_k.at<double>(1,1)*v_k.at<double>(1,0);
    	x_kplus1.at<double>(2,0) = x_k.at<double>(2,0)+u_k.at<double>(2,0)*v_k.at<double>(2,0);
    	x_kplus1.at<double>(3,0) = x_k.at<double>(3,0)+u_k.at<double>(3,0)*v_k.at<double>(3,0);
    }
    void measurementFunction(const Mat& x_k, const Mat& n_k, Mat& z_k)
    {
    	z_k.at<double>(0,0) = x_k.at<double>(0, 0)+n_k.at<double>(0,0);
    	z_k.at<double>(1,0) = x_k.at<double>(1, 0)+n_k.at<double>(1,0);
    }
};

void SceneProc::InitUKFilter(float x, float y)
{
	const double alpha = 1.0;
	const double beta = 2.0;
	const double kappa = -2.0;

	const double mse_treshold = 0.5;

	int MP = 2;
	int DP = 4;
	int CP = 0;
	int type = CV_64F;

	if(uncsentedKalmanFilter.empty()){
		model = cv::Ptr<UkfSystemModel>( new ObjectTrackerModel() );
		AugmentedUnscentedKalmanFilterParams params( DP, MP, CP, 0, 0, model );

		Mat processNoiseCov = Mat::zeros( DP, DP, type );
		processNoiseCov.at<double>(0, 0) = 0.1;//0.00001;//x
		processNoiseCov.at<double>(1, 1) = 0.1;//0.00001;//y
		processNoiseCov.at<double>(2, 2) = 0.01;//vx
		processNoiseCov.at<double>(3, 3) = 0.01;//vy
//		Mat processNoiseCovSqrt = Mat::zeros( DP, DP, type );
//		sqrt( processNoiseCov, processNoiseCovSqrt );

		Mat measurementNoiseCov = Mat::zeros( MP, MP, type );
		measurementNoiseCov.at<double>(0, 0) = 0.25;//0.0025;//x
		measurementNoiseCov.at<double>(1, 1) = 0.25;//0.0025;//y
//		Mat measurementNoiseCovSqrt = Mat::zeros( MP, MP, type );
//		sqrt( measurementNoiseCov, measurementNoiseCovSqrt );

		Mat P =  1e-6 *Mat::eye( DP, DP, type );//

		Mat state( DP, 1, type );
		state.at<double>(0, 0) = (double)x;
		state.at<double>(1, 0) = (double)y;
		state.at<double>(2, 0) = 0.0;
		state.at<double>(3, 0) = 0.0;

		Mat initState = state.clone();

		params.errorCovInit = P.clone();
		params.measurementNoiseCov = measurementNoiseCov.clone();
		params.processNoiseCov = processNoiseCov.clone();
		params.stateInit = initState.clone();

		params.alpha = alpha;
		params.beta = beta;
		params.k = kappa;

		uncsentedKalmanFilter = createAugmentedUnscentedKalmanFilter( params );
	}
}

void  SceneProc::AnalyseSceneLock()
{
	int MP = 2;
	int DP = 4;
	int CP = 0;
	int type = CV_64F;
	int nsize = m_sceneState.size();
	if( nsize<2){// 可能需要冗余几帧，待验证
		m_filterVel.x = m_filterVel.y = 0.f;
		return;
	}else if(nsize == 2){
		m_sceneState[0].deltaT = m_sceneState[1].deltaT;
	}
	SceneState		curState;
	curState = m_sceneState[nsize-1];

	if(curState.confidence ==0){
		cv::Mat state = uncsentedKalmanFilter->getState();
		m_filterVel.x = (float)(state.at<double>(0,0))-m_bakDetlta.x;
		m_filterVel.y = (float)(state.at<double>(1,0))-m_bakDetlta.y;
		return;
	}

	double deltat = curState.deltaT/1000000.0;
	double deltat2 = deltat*deltat;

	if(!uncsentedKalmanFilter.empty()){
		cv::Mat u_k  = cv::Mat(DP, MP, type);
//		cv::Mat q_a =  cv::Mat(MP, 1, type);
//		m_rng.fill( q_a, RNG::NORMAL, Scalar::all(0),  Scalar::all(1) );
//		q_a *= m_accNoise;

		cv::Mat state = uncsentedKalmanFilter->getState();
		cv::Mat measure = cv::Mat(MP, 1, type);

		u_k.at<double>(0, 0) = deltat; //px
		u_k.at<double>(1, 0) = deltat;//py
		u_k.at<double>(2, 0) = deltat;//vx
		u_k.at<double>(3, 0) = deltat;//vy

		u_k.at<double>(0, 1) = 0.5*deltat2; //px
		u_k.at<double>(1, 1) = 0.5*deltat2;//py
		u_k.at<double>(2, 1) = 0;//vx
		u_k.at<double>(3, 1) = 0;//vy

		m_sumDelta.x += curState.mv.x;
		m_sumDelta.y += curState.mv.y;

		measure.at<double>(0,0) = (double)m_sumDelta.x;
		measure.at<double>(1,0) = (double)m_sumDelta.y;

		uncsentedKalmanFilter->predict(u_k);
		cv::Mat correctStateUKF = uncsentedKalmanFilter->correct(measure);

		m_filterVel.x = (float)(correctStateUKF.at<double>(0,0))-m_bakDetlta.x;
		m_filterVel.y = (float)(correctStateUKF.at<double>(1,0))-m_bakDetlta.y;

		m_bakDetlta.x =  (float)(correctStateUKF.at<double>(0,0));
		m_bakDetlta.y =  (float)(correctStateUKF.at<double>(1,0));
	}else{
		printf("%s:UKF is not Inited \n",__func__);
		assert(0);
	}
}

}
