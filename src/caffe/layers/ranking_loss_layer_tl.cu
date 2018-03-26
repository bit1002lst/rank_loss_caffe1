#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RealRankingLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_sub(
      count,
      bottom[2]->gpu_data(),  // G(a)
      bottom[3]->gpu_data(),  // G(b)
      diff2_.mutable_gpu_data());  // G(a)_i-G(b)_i
//caffe_gpu_axpby(
//      count,
//      Dtype(0.0125),
//      diff2_.cpu_data(),
//      Dtype(0),
//      rescaled_.mutable_cpu_data());

  Dtype margin = this->layer_param_.real_ranking_loss_param().margin();
    //bool legacy_version =
  //    this->layer_param_.real_ranking_loss_param().legacy_version();
  Dtype loss(0.0);
  //LOG(INFO) << "HI! ";
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (diff2_.cpu_data()[i] > 0) {  // if first is larger (a-b)
        loss += std::max(margin - diff_.cpu_data()[i], Dtype(0.0));
        //dist_mine_.mutable_gpu_data()[i]=diff_.cpu_data()[i];
        //LOG(INFO) << "BYE! ";
    }
    else {
        //LOG(INFO) << margin << " " << -diff_.cpu_data()[i];
        loss += std::max(margin + diff_.cpu_data()[i], Dtype(0.0));
        //dist_mine_.mutable_gpu_data()[i]= -diff_.cpu_data()[i];
        //LOG(INFO) << "BYE! ";
    }
//LOG(INFO) <<  "G(a)-G(b): " << diff2_.cpu_data()[i] << "; a-b: " << diff_.cpu_data()[i];
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const Dtype alpha,
    const Dtype* y, const Dtype* diff, Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_mine
    Dtype mdist(0.0);
    if (y[n] > 0.0) {  // first is larger
        mdist = (margin - diff[n]);
        if (mdist > 0.0) {
            bottom_diff[i] = -alpha;
        } else {
            bottom_diff[i] = 0;
        }
    }
    else {
        mdist = (margin + diff[n]);
        if (mdist > 0.0) {
            bottom_diff[i] = alpha;
        }
        else {
        bottom_diff[i] = 0;
        }
    }
  }
}

template <typename Dtype>
void RealRankingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const int count = bottom[0]->count();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {

      const int channels = bottom[0]->channels(); //1
      Dtype margin = this->layer_param_.real_ranking_loss_param().margin();
      //const bool legacy_version =
      //    this->layer_param_.real_ranking_loss_param().legacy_version();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, alpha,
          diff2_.gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),
          bottom[i]->mutable_gpu_diff());

      CUDA_POST_KERNEL_CHECK;
    }
  }
//int i = 0;
//for (int j = 0; j < count; ++j) {
//LOG(INFO) <<  "G(a)-G(b): " << diff2_.cpu_data()[j] << "; a-b: " << diff_.cpu_data()[j] << "; diff: " << bottom[i]->cpu_diff()[j];
//}
}

INSTANTIATE_LAYER_GPU_FUNCS(RealRankingLossLayer);

}  // namespace caffe
