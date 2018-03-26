#ifndef CAFFE_REAL_RANKING_LOSS_LAYER_HPP_
#define CAFFE_REAL_RANKING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
    /**
     * @brief Computes the pair ranking loss @f$
     *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left(y\right) d +
     *              \left(1-y\right) \max \left(margin-d, 0\right)^2
     *          @f$ where @f$
     *          d = \left| \left| a_n - b_n \right| \right|_2 @f$. This can be
     *          used to train siamese networks.
     *
     * @param bottom input Blob vector (length 4)
     *   -# @f$ (N \times 1 \times 1 \times 1) @f$
     *      the features @f$ a \in [-\infty, +\infty]@f$
     *   -# @f$ (N \times 1 \times 1 \times 1) @f$
     *      the features @f$ b \in [-\infty, +\infty]@f$
     *   -# @f$ (N \times 1 \times 1 \times 1) @f$
     *      the binary (1:G(a)>G(b), 0:G(a)<G(b)) @f$ s \in [0, 1]@f$
     * @param top output Blob vector (length 1)
     *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
     *      the computed ranking loss: @f$ E =
     *          \frac{1}{2N} \sum\limits_{n=1}^N \left(y\right) d +
     *          \left(1-y\right) \max \left(margin-((+ data)-(- data)), 0\right)
     *          @f$ where @f$.
     */
    template <typename Dtype>
class RealRankingLossLayer : public LossLayer<Dtype> {
public:
    explicit RealRankingLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param), diff_() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        
    virtual inline int ExactNumBottomBlobs() const { return 4; }
    virtual inline const char* type() const { return "RealRankingLoss"; }
        /**
         * Unlike most loss layers, in the RealRankingLossLayer we can backpropagate
         * to the first two inputs.
         */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
        return bottom_index != 2;
        }
        
    protected:
        /// @copydoc RealRankingLossLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        
    Blob<Dtype> diff_;  // cached for backward pass
    Blob<Dtype> dist_mine_;  // cached for backward pass
    Blob<Dtype> diff2_;  // tmp storage for gpu forward pass
    Blob<Dtype> rescaled_;
    Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
};
}// namespace caffe
#endif  // CAFFE_REAL_RANKING_LOSS_LAYER_HPP_
    