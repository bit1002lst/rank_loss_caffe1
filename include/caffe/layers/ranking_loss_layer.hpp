#ifndef CAFFE_RANKING_LOSS_LAYER_HPP_
#define CAFFE_RANKING_LOSS_LAYER_HPP_

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
    class RankingLossLayer : public LossLayer<Dtype> {
    public:
        explicit RankingLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        
        virtual inline int ExactNumBottomBlobs() const { return 4; }
        virtual inline const char* type() const { return "RankingLoss"; }
        /**
         * Unlike most loss layers, in the RankingLossLayer we can backpropagate
         * to the first two inputs.
         */
        virtual inline bool AllowForceBackward(const int bottom_index) const {
            return bottom_index != 2;
        }
        
    protected:
        /// @copydoc RankingLossLayer
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        
        /**
         * @brief Computes the Ranking error gradient w.r.t. the inputs.
         *
         * Computes the gradients with respect to the two input vectors (bottom[0] and
         * bottom[1]), but not the similarity label (bottom[2]).
         *
         * @param top output Blob vector (length 1), providing the error gradient with
         *      respect to the outputs
         *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
         *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
         *      as @f$ \lambda @f$ is the coefficient of this layer's output
         *      @f$\ell_i@f$ in the overall Net loss
         *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
         *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
         *      (*Assuming that this top Blob is not used as a bottom (input) by any
         *      other layer of the Net.)
         * @param propagate_down see Layer::Backward.
         * @param bottom input Blob vector (length 2)
         *   -# @f$ (N \times C \times 1 \times 1) @f$
         *      the features @f$a@f$; Backward fills their diff with
         *      gradients if propagate_down[0]
         *   -# @f$ (N \times C \times 1 \times 1) @f$
         *      the features @f$b@f$; Backward fills their diff with gradients if
         *      propagate_down[1]
         */
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        
        Blob<Dtype> diff_;  // cached for backward pass
        //Blob<Dtype> diff2_;  // cached for ground-truth label difference to determine valid pairs
        Blob<Dtype> dist_mine_;  // cached for backward pass
        //Blob<Dtype> diff_sq_;  // tmp storage for gpu forward pass
        Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
    };
    
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
#endif  // CAFFE_RANKING_LOSS_LAYER_HPP_
    