#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/ranking_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RankingLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  
  CHECK_EQ(bottom[3]->channels(), 1);
  CHECK_EQ(bottom[3]->height(), 1);
  CHECK_EQ(bottom[3]->width(), 1);

  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  //diff2_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  //diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_mine_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void RankingLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  
//  caffe_sub(
//	count,
//	bottom[2]->cpu_data(),  // Ya -- ground-truth label
//	bottom[3]->cpu_data(),  // Yb -- ground-truth label
//	diff2_.mutable_cpu_data());  // Ya_i-Yb_i

  //const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.ranking_loss_param().margin();
  //int legacy_version =
  //  this->layer_param_.ranking_loss_param().version();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
      //if (static_cast<int>(bottom[2]->cpu_data()[i])) {  //if first is larger than second by >margin --- a>b+margin
      if ( bottom[2]->cpu_data()[i] > bottom[3]->cpu_data()[i] + margin ) {  //if first is larger (a-b)
          dist_mine_.mutable_cpu_data()[i] =diff_.cpu_data()[i];
      }
      //else if (diff2_cpu_data()[i] < -1*margin){ // if first is less than second by >margin  -- a<b-margin
      else if ( bottom[2]->cpu_data()[i] < bottom[3]->cpu_data()[i] - margin ) {  //if first is larger (a-b)
          dist_mine_.mutable_cpu_data()[i] = -diff_.cpu_data()[i];
      }
      else { // if two ground-truth are with no big difference, then do not penalize their relative ranking
          dist_mine_.mutable_cpu_data()[i] = Dtype(0.0); // -diff_.cpu_data()[i];
      }
      loss += std::max(margin - dist_mine_.cpu_data()[i], Dtype(0.0)); //margin = 0.3
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void RankingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.ranking_loss_param().margin();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      //const Dtype sign = (i == 0) ? 1 : -1; // a bug here!!! --- should comment this line and add the other line as below
      //const Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());

      int num = bottom[i]->num();
      int channels = bottom[i]->channels(); // 1
      for (int j = 0; j < num; ++j) {

	Dtype sign = (bottom[2]->cpu_data()[j] > bottom[3]->cpu_data()[j]) ? 1 : -1; // a bug here!!! --- should add this line
        Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());

        Dtype* bout = bottom[i]->mutable_cpu_diff();
        Dtype mdist(0.0);
        mdist = margin - dist_mine_.cpu_data()[j];
        if (mdist > Dtype(0.0)) {
            //if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // first is larger
            if ( bottom[2]->cpu_data()[j] > bottom[3]->cpu_data()[j] + margin ) {  //if first is larger (a-b)
                bout[j] = -alpha;
            }
      	    else if ( bottom[2]->cpu_data()[j] < bottom[3]->cpu_data()[j] - margin ) {  //if first is larger (a-b)
		bout[j] = alpha;
	    }
            else {  // second is larger
                bout[j] = Dtype(0); //alpha;
            }
        }else {
            caffe_set(channels, Dtype(0), bout + (j*channels)); //bout[j] = 0 should work
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RankingLossLayer);
#endif

INSTANTIATE_CLASS(RankingLossLayer);
REGISTER_LAYER_CLASS(RankingLoss);

}  // namespace caffe
